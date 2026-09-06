"""Transport, hardware and safety tests against a mocked ESP32 (§26).

No robot required. FakeSession stands in for requests.Session and records
every call, so ordering and retry behaviour are directly assertable.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.config import ServoLimits, Settings  # noqa: E402
from minibot.events.bus import Event, EventBus  # noqa: E402
from minibot.robot.esp32_client import Esp32Client, Esp32Unavailable  # noqa: E402
from minibot.robot.hardware import Esp32RobotHardware, RobotStatus  # noqa: E402

STATUS = {"camera": True, "oled": True, "servos": True, "audio": True,
          "mic": True, "speaking": False, "listening": False, "volume": 10,
          "emotion": "neutral", "pan": 90, "tilt": 90, "rssi": -55,
          "heap": 180000, "psram": 8000000}


class FakeResponse:
    def __init__(self, status=200, body=b"ok", js=None):
        self.status_code = status
        self.content = body
        self._js = js
        self.text = body.decode(errors="replace")

    def json(self):
        return self._js if self._js is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            e = requests.HTTPError(f"{self.status_code}")
            e.response = self
            raise e


class FakeSession:
    """Records calls; `plan` maps a path to a list of responses/exceptions."""

    def __init__(self, plan=None, delay=0.0):
        self.calls: list[tuple[str, str]] = []
        self.plan = plan or {}
        self.delay = delay
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def request(self, method, url, params=None, data=None, headers=None,
                timeout=None):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append((method, path))
            if self.delay:
                time.sleep(self.delay)
            queued = self.plan.get(path)
            if queued:
                item = queued.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item
            if path == "/status":
                return FakeResponse(js=dict(STATUS))
            if path == "/capture":
                return FakeResponse(body=b"\xff\xd8jpegdata\xff\xd9")
            if path == "/level":
                return FakeResponse(js={"rms": 40, "peak": 90})
            if path == "/say":
                return FakeResponse(js={"ok": True, "bytes": len(data or b"")})
            return FakeResponse()
        finally:
            with self._lock:
                self.concurrent -= 1


def make_client(plan=None, delay=0.0, retries=2):
    c = Esp32Client("http://bot.local", timeout=1.0, retries=retries,
                    min_interval=0.0)
    c._s = FakeSession(plan, delay)
    return c


class TestEsp32Client:
    def test_status_parsed(self):
        c = make_client()
        assert c.status()["pan"] == 90

    def test_retries_on_server_error_then_succeeds(self):
        plan = {"/look": [FakeResponse(status=500), FakeResponse(body=b"ok")]}
        c = make_client(plan)
        assert c.look(90, 90) == "ok"
        assert c.stats["retries"] == 1

    def test_503_is_not_retried(self):
        """503 from this firmware means a subsystem failed at boot — those
        flags never change at runtime, so retrying only wastes the device's
        single request thread."""
        plan = {"/look": [FakeResponse(status=503)] * 5}
        c = make_client(plan)
        with pytest.raises(requests.HTTPError):
            c.look(90, 90)
        assert c._s.calls.count(("GET", "/look")) == 1

    def test_404_is_not_retried(self):
        plan = {"/look": [FakeResponse(status=404)] * 5}
        c = make_client(plan)
        with pytest.raises(requests.HTTPError):
            c.look(90, 90)
        assert c._s.calls.count(("GET", "/look")) == 1

    def test_500_is_retried(self):
        """'capture failed' / 'no memory' can be a transient hiccup, unlike 503."""
        plan = {"/capture": [FakeResponse(status=500),
                             FakeResponse(body=b"\xff\xd8ok\xff\xd9")]}
        c = make_client(plan)
        assert c.capture().startswith(b"\xff\xd8")
        assert c._s.calls.count(("GET", "/capture")) == 2

    def test_exhausted_retries_raise_unavailable(self):
        plan = {"/status": [requests.ConnectionError("down")] * 5}
        c = make_client(plan)
        with pytest.raises(Esp32Unavailable):
            c.status()
        assert c.online is False

    def test_recovery_marks_online_again(self):
        seen = []
        plan = {"/status": [requests.ConnectionError("down")] * 3}
        c = make_client(plan)
        c._on_state_change = seen.append
        with pytest.raises(Esp32Unavailable):
            c.status()
        c.status()
        assert seen == [False, True]

    def test_requests_are_serialized(self):
        """The firmware services one request at a time inside loop(); parallel
        calls must never overlap on the wire."""
        c = make_client(delay=0.02)
        threads = [threading.Thread(target=c.status) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c._s.max_concurrent == 1

    def test_rate_limit_spaces_requests(self):
        c = Esp32Client("http://bot.local", timeout=1.0, retries=0,
                        min_interval=0.05)
        c._s = FakeSession()
        t0 = time.monotonic()
        for _ in range(4):
            c.status()
        assert time.monotonic() - t0 >= 0.15 - 1e-6

    def test_unknown_expression_falls_back_to_neutral(self):
        c = make_client()
        c.face("definitely-not-a-face")
        assert ("GET", "/set") in c._s.calls

    def test_record_omits_vad_params_when_disabled(self):
        c = make_client()
        c.record(max_ms=2000)
        assert c._s.calls[-1] == ("GET", "/mic")

    def test_play_skips_empty_audio(self):
        c = make_client()
        assert c.play(b"") is None
        assert c._s.calls == []


class TestHardware:
    def test_status_maps_to_dataclass(self):
        hw = Esp32RobotHardware(make_client())
        st = hw.get_status()
        assert isinstance(st, RobotStatus)
        assert st.camera and st.servos and st.pan == 90

    def test_status_survives_unknown_fields(self):
        """New firmware fields must not crash an older Mac build."""
        plan = {"/status": [FakeResponse(js={**STATUS, "brand_new_field": 7})]}
        hw = Esp32RobotHardware(make_client(plan))
        assert hw.get_status().raw["brand_new_field"] == 7

    def test_offline_status_keeps_last_known_and_flags_it(self):
        plan = {"/status": [requests.ConnectionError("down")] * 5}
        hw = Esp32RobotHardware(make_client(plan))
        assert hw.get_status().online is False

    def test_rejects_unknown_expression(self):
        hw = Esp32RobotHardware(make_client())
        with pytest.raises(ValueError):
            hw.set_expression("curious")

    def test_camera_lock_blocks_concurrent_capture(self):
        """Mirrors the firmware's capturing/setHead handshake."""
        hw = Esp32RobotHardware(make_client(delay=0.05))
        seen = []

        def grab():
            hw.capture_image()
            seen.append(hw.is_camera_busy())

        ts = [threading.Thread(target=grab) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert hw.is_camera_busy() is False
        assert len(seen) == 3

    def test_head_position_tracked(self):
        hw = Esp32RobotHardware(make_client())
        hw.set_head_position(120, 70)
        assert hw.head_position() == (120, 70)


class TestServoBoundsAndTools:
    """§22: the model is an untrusted source of requested actions."""

    def _agent(self):
        from minibot.agent.robot_agent import RobotAgent
        from minibot.ai.provider import AIProvider

        class NullAI(AIProvider):
            name = "null"
            async def connect(self): ...
            async def disconnect(self): ...
            def register_tools(self, tools, handler): ...
            async def send_audio(self, pcm, rate): ...
            async def send_frame(self, jpeg): self.frame = jpeg
            async def send_text(self, text): ...
            async def complete_turn(self, timeout=90.0): ...
            async def interrupt(self): ...

        class NullSpeech:
            def synthesize(self, text): ...
            def stop(self): ...
            def interrupt(self): ...
            def is_speaking(self): return False

        hw = Esp32RobotHardware(make_client())
        return RobotAgent(Settings(), hw, NullAI(), NullSpeech(), EventBus())

    @pytest.mark.parametrize("pan,tilt,exp_pan,exp_tilt", [
        (10000, 10000, 165, 150),     # the spec's explicit example
        (-500, -500, 15, 30),
        (90, 90, 90, 90),
        (165, 150, 165, 150),
        (14, 29, 15, 30),
    ])
    def test_look_at_clamps(self, pan, tilt, exp_pan, exp_tilt):
        agent = self._agent()
        out = asyncio.run(agent.tools.dispatch("look_at", {"pan": pan, "tilt": tilt}))
        assert (out["pan"], out["tilt"]) == (exp_pan, exp_tilt)

    def test_clamp_matches_firmware_defines(self):
        ino = (Path(__file__).resolve().parent.parent / "ai_mini_bot.ino").read_text()
        s = ServoLimits()
        for define, value in [("PAN_MIN", s.pan_min), ("PAN_MAX", s.pan_max),
                              ("TILT_MIN", s.tilt_min), ("TILT_MAX", s.tilt_max)]:
            assert f"#define {define}" in ino
            line = next(l for l in ino.splitlines()
                        if l.startswith(f"#define {define}"))
            assert int(line.split()[2]) == value, f"{define} drifted from firmware"

    def test_look_at_with_no_args_holds_position(self):
        agent = self._agent()
        agent.hw.set_head_position(120, 70)
        out = asyncio.run(agent.tools.dispatch("look_at", {}))
        assert (out["pan"], out["tilt"]) == (120, 70)

    def test_unknown_tool_returns_error_not_raise(self):
        agent = self._agent()
        out = asyncio.run(agent.tools.dispatch("set_pwm", {"ch": 3, "us": 2400}))
        assert out["ok"] is False and "unknown tool" in out["error"]

    def test_no_hardware_primitives_are_exposed(self):
        """Gemini must never see PCA9685, PWM, I2C or GPIO (§2)."""
        agent = self._agent()
        names = {t.name for t in agent.tools.specs}
        banned = {"set_pwm", "servo_write", "i2c_write", "set_pca9685_channel"}
        assert not (names & banned)
        blob = " ".join(t.description.lower() for t in agent.tools.specs)
        for word in ("pca9685", "pwm", "i2c", "gpio"):
            assert word not in blob

    def test_bad_expression_returns_error_not_raise(self):
        agent = self._agent()
        out = asyncio.run(agent.tools.dispatch("set_face", {"emotion": "curious"}))
        assert out["ok"] is False

    def test_take_photo_reports_camera_failure_honestly(self):
        """§27: never claim an action succeeded until it actually did."""
        from minibot.agent.robot_agent import RobotAgent
        agent = self._agent()
        plan = {"/capture": [FakeResponse(status=503)] * 5}
        agent.hw = Esp32RobotHardware(make_client(plan))
        out = asyncio.run(agent.tools.dispatch("take_photo", {}))
        assert out["ok"] is False and "camera" in out["error"].lower()


class TestEventBus:
    def test_sync_and_async_handlers_both_run(self):
        bus, seen = EventBus(), []
        bus.subscribe(Event.PERSON_DETECTED, lambda m: seen.append("sync"))

        async def ahandler(m):
            seen.append("async")

        bus.subscribe(Event.PERSON_DETECTED, ahandler)
        asyncio.run(bus.publish(Event.PERSON_DETECTED))
        assert sorted(seen) == ["async", "sync"]

    def test_failing_handler_does_not_stop_the_others(self):
        bus, seen = EventBus(), []

        def boom(m):
            raise RuntimeError("nope")

        bus.subscribe(Event.SPEECH_STARTED, boom)
        bus.subscribe(Event.SPEECH_STARTED, lambda m: seen.append(1))
        asyncio.run(bus.publish(Event.SPEECH_STARTED))
        assert seen == [1]

    def test_unsubscribe(self):
        bus, seen = EventBus(), []
        off = bus.subscribe(Event.SPEECH_STOPPED, lambda m: seen.append(1))
        off()
        asyncio.run(bus.publish(Event.SPEECH_STOPPED))
        assert seen == []

    def test_payload_delivered(self):
        bus, got = EventBus(), {}
        bus.subscribe(Event.MEMORY_RETRIEVED, lambda m: got.update(m.payload))
        asyncio.run(bus.publish(Event.MEMORY_RETRIEVED, count=3, best=0.89))
        assert got == {"count": 3, "best": 0.89}


class TestConfig:
    def test_legacy_env_names_still_work(self, monkeypatch):
        monkeypatch.setenv("BOT_URL", "http://10.0.0.5")
        monkeypatch.setenv("ELEVEN_VOICE_ID", "legacy-voice")
        s = Settings.load()
        assert s.esp32_base_url == "http://10.0.0.5"
        assert s.elevenlabs_voice_id == "legacy-voice"

    def test_new_names_take_precedence(self, monkeypatch):
        monkeypatch.setenv("BOT_URL", "http://legacy")
        monkeypatch.setenv("ESP32_BASE_URL", "http://new")
        assert Settings.load().esp32_base_url == "http://new"

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("ESP32_BASE_URL", "http://bot/")
        assert Settings.load().esp32_base_url == "http://bot"

    def test_bad_numbers_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("SERVO_MAX_PAN", "not-a-number")
        assert Settings.load().servo.pan_max == 165

    def test_camera_fps_never_exceeds_gemini_ceiling(self):
        c = Settings().camera
        for fps in (c.idle_fps, c.conversation_fps, c.visual_query_fps,
                    c.tracking_fps):
            assert fps <= c.max_fps == 1.0

    def test_no_secrets_have_defaults(self):
        s = Settings()
        assert s.openai_api_key == s.gemini_api_key == s.elevenlabs_api_key == ""


class TestMicSelection:
    """AUDIO_INPUT picks which microphone RobotAgent listens on (2026-09-03).

    The computer's own mic is the default and the fallback (2026-09-05); only
    an explicit AUDIO_INPUT=bot moves listening onto the robot's MAX4466.

    wait_quiet() must stay on the ESP32 regardless of the mic in use — it
    gates on the ROBOT's own speaker, needed either way so the mic doesn't
    re-trigger the bot on its own voice carrying across the room.
    """

    def _agent(self, audio_input="mac"):
        from minibot.agent.robot_agent import RobotAgent
        from minibot.ai.provider import AIProvider

        class NullAI(AIProvider):
            name = "null"
            async def connect(self): ...
            async def disconnect(self): ...
            def register_tools(self, tools, handler): ...
            async def send_audio(self, pcm, rate): ...
            async def send_frame(self, jpeg): ...
            async def send_text(self, text): ...
            async def complete_turn(self, timeout=90.0): ...
            async def interrupt(self): ...

        class NullSpeech:
            def synthesize(self, text): ...
            def stop(self): ...
            def interrupt(self): ...
            def is_speaking(self): return False

        hw = Esp32RobotHardware(make_client())
        s = Settings(audio_input=audio_input)
        return RobotAgent(s, hw, NullAI(), NullSpeech(), EventBus())

    def test_bot_mode_uses_the_esp32_client_as_mic(self):
        agent = self._agent("bot")
        mic = asyncio.run(agent._build_mic())
        assert mic is agent.hw_client

    @staticmethod
    def _fake_sounddevice(monkeypatch):
        import sys as _sys
        import types as _types
        fake_sd = _types.ModuleType("sounddevice")
        fake_sd.query_devices = lambda device=None, kind=None: {"name": "fake"}

        class _FakeInputStream:
            def __init__(self, **kw): pass
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        fake_sd.InputStream = _FakeInputStream
        monkeypatch.setitem(_sys.modules, "sounddevice", fake_sd)
        _sys.modules.pop("minibot.audio.mac_mic", None)

    def test_mac_mode_builds_a_mac_mic_source(self, monkeypatch):
        self._fake_sounddevice(monkeypatch)
        agent = self._agent("mac")
        mic = asyncio.run(agent._build_mic())
        assert type(mic).__name__ == "MacMicSource"
        assert agent.vad is True

    def test_the_computer_mic_is_the_default(self, monkeypatch):
        """Settings() with nothing set listens on this computer, not the robot."""
        self._fake_sounddevice(monkeypatch)
        assert Settings().audio_input == "mac"
        agent = self._agent()
        mic = asyncio.run(agent._build_mic())
        assert type(mic).__name__ == "MacMicSource"

    def test_unknown_mode_warns_and_falls_back_to_the_computer_mic(self, monkeypatch):
        """A typo leaves the robot listening on the mic that is always there,
        rather than silently moving to one nobody asked for."""
        self._fake_sounddevice(monkeypatch)
        agent = self._agent("bluetooth-headset")
        mic = asyncio.run(agent._build_mic())
        assert type(mic).__name__ == "MacMicSource"
        assert mic is not agent.hw_client
