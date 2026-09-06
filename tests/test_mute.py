"""Mute mode.

Mute has one hole in it by construction: something has to hear "mute off", so
the robot cannot literally stop listening. What it can do is stop ACTING — no
speech, no movement, no tools, nothing reaching X — and stop uploading anything
longer than the wake phrase itself. These tests pin that distinction, because
"muted" quietly meaning "still posting to X" is the failure that matters.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.agent.robot_agent import (MUTED_WAKE_MAX_MS,  # noqa: E402
                                       MUTED_WAKE_PROMPT, RobotAgent)
from minibot.audio.dsp import pcm_to_wav  # noqa: E402
from minibot.config import Settings  # noqa: E402
from minibot.events.bus import EventBus  # noqa: E402


class FakeClient:
    def __init__(self):
        self.beeps = []

    def beep(self, f, ms):
        self.beeps.append((f, ms))
        return "ok"

    def wait_quiet(self, timeout=30):
        pass


class FakeHardware:
    def __init__(self):
        self.client = FakeClient()
        self.faces = []
        self.moves = []

    def set_expression(self, e):
        self.faces.append(e)

    def set_head_position(self, p, t):
        self.moves.append((p, t))

    def head_position(self):
        return (90, 90)

    def capture_image(self):
        return b"\xff\xd8photo"


class Reply:
    def __init__(self, text=""):
        self.text = text
        self.error = ""
        self.interrupted = False
        self.tool_calls = []


class FakeAI:
    """Records what actually reached the model."""

    name = "fake"

    def __init__(self):
        self.audio = []
        self.texts = []

    async def connect(self): ...
    async def disconnect(self): ...
    def register_tools(self, tools, handler): self.handler = handler
    async def send_audio(self, pcm, rate): self.audio.append((pcm, rate))
    async def send_frame(self, jpeg): ...
    async def send_text(self, text, *, turn_complete=True):
        self.texts.append((text, turn_complete))
    async def complete_turn(self, timeout=90.0): return Reply()
    async def interrupt(self): ...


class FakeSpeech:
    def __init__(self):
        self.said = []

    def synthesize(self, text): self.said.append(text)
    def stop(self): ...
    def interrupt(self): ...
    def is_speaking(self): return False


def build_agent() -> RobotAgent:
    return RobotAgent(Settings(), FakeHardware(), FakeAI(), FakeSpeech(),
                      EventBus())


def run(coro):
    return asyncio.run(coro)


def silence(ms: int) -> bytes:
    return pcm_to_wav(b"\x00\x00" * int(16000 * ms / 1000), 16000)


# -- entering and leaving -------------------------------------------

def test_mute_sleeps_the_face_and_chirps_so_you_know_it_took():
    agent = build_agent()
    result = run(agent._dispatch("mute", {}))
    assert result["ok"] and result["muted"] is True
    assert agent.muted is True
    assert agent.hw.faces[-1] == "sleep"
    # Falling pair — a beep is not talking, and it answers "did that work?"
    assert len(agent.hw.client.beeps) == 2
    assert agent.hw.client.beeps[0][0] > agent.hw.client.beeps[1][0]


def test_unmute_comes_back_with_a_rising_chirp_and_a_normal_face():
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    result = run(agent._dispatch("unmute", {}))
    assert result["ok"] and result["muted"] is False
    assert agent.muted is False
    assert agent.hw.faces[-1] == "neutral"
    rising = agent.hw.client.beeps[2:]
    assert rising[0][0] < rising[1][0]


def test_muting_twice_changes_nothing():
    """mute is deliberately NOT on the allowed list: a muted robot refuses it
    like everything else, and stays muted. Fail-closed beats a special case."""
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    beeps = len(agent.hw.client.beeps)
    again = run(agent._dispatch("mute", {}))
    assert again["ok"] is False and again["muted"] is True
    assert agent.muted is True
    assert len(agent.hw.client.beeps) == beeps      # no second announcement


# -- what a muted robot refuses -------------------------------------

def test_every_tool_but_unmute_refuses_while_muted():
    """The gate is at the dispatcher, so a tool added later is muted by
    default rather than quietly exempt."""
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    for name in ("set_face", "look_at", "take_photo", "remember", "recall",
                 "forget", "mute"):
        result = run(agent._dispatch(name, {"emotion": "happy", "pan": 20,
                                            "fact": "x", "query": "x",
                                            "memory_id": "1"}))
        assert result["ok"] is False, f"{name} ran while muted"
        assert result["muted"] is True
    assert agent.hw.moves == []                     # nothing physical happened


def test_a_muted_robot_cannot_post_to_x():
    """The one that would actually matter. Mute has to reach the account, not
    just the mouth."""
    import tests.test_x_account as x_tests

    account, session = x_tests.make_account(require_confirm=False)
    agent = RobotAgent(Settings(), FakeHardware(), FakeAI(), FakeSpeech(),
                       EventBus(), x=account)
    run(agent._dispatch("mute", {}))
    result = run(agent._dispatch("post_to_x", {"text": "hello"}))
    assert result["ok"] is False and result["muted"] is True
    assert session.calls == []


def test_unmute_is_the_one_thing_that_still_works():
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    assert run(agent._dispatch("unmute", {}))["ok"] is True
    assert agent.muted is False


# -- what a muted robot says ----------------------------------------

def test_a_muted_robot_does_not_speak_even_if_the_model_replies():
    agent = build_agent()
    run(agent._dispatch("mute", {}))

    class Reply:
        text, error, interrupted, tool_calls = "I am still here!", "", False, []

    async def complete_turn(timeout=90.0):
        return Reply()

    agent.ai.complete_turn = complete_turn
    run(agent._turn())
    assert agent.speech.said == []


def test_the_reply_to_mute_off_is_spoken():
    """unmute happens DURING the turn, so by the time the reply is handled the
    robot is no longer muted and its "I'm back" must come out."""
    agent = build_agent()
    run(agent._dispatch("mute", {}))

    class Reply:
        text, error, interrupted, tool_calls = "Back.", "", False, []

    async def complete_turn(timeout=90.0):
        await agent._dispatch("unmute", {})        # what the model does mid-turn
        return Reply()

    agent.ai.complete_turn = complete_turn
    run(agent._turn())
    assert agent.speech.said == ["Back."]


# -- what leaves the machine ----------------------------------------

def test_a_short_clip_is_sent_so_mute_off_can_be_heard():
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    run(agent.respond_to_audio(silence(800)))
    assert len(agent.ai.audio) == 1
    # Framed every time: "you are muted" ages out of a compressed session.
    assert agent.ai.texts[-1] == (MUTED_WAKE_PROMPT, False)


def test_a_long_clip_is_never_sent_while_muted(monkeypatch):
    """A conversation held in front of a muted robot dies on this machine.
    Uploading the room while the person believes they muted it would be the
    wrong reading of the word."""
    agent = build_agent()
    agent.room = type("R", (), {"endpoint": 0.0})()
    agent.mic = object()
    run(agent._dispatch("mute", {}))

    long_clip = silence(MUTED_WAKE_MAX_MS + 4000)
    monkeypatch.setattr("minibot.agent.robot_agent.capture_turn",
                        lambda *a, **k: long_clip)
    monkeypatch.setattr("minibot.agent.robot_agent.wait_for_voice",
                        lambda *a, **k: None)
    monkeypatch.setattr("minibot.agent.robot_agent.speech_ms",
                        lambda *a, **k: 5000.0)
    monkeypatch.setattr("minibot.agent.robot_agent.trim_tail",
                        lambda wav, *a, **k: wav)

    run(agent.listen_and_respond())
    assert agent.ai.audio == []                    # nothing left the Mac
    assert agent.ai.texts == []


def test_an_unmuted_robot_sends_long_clips_normally(monkeypatch):
    agent = build_agent()
    agent.room = type("R", (), {"endpoint": 0.0})()
    agent.mic = object()

    long_clip = silence(MUTED_WAKE_MAX_MS + 4000)
    monkeypatch.setattr("minibot.agent.robot_agent.capture_turn",
                        lambda *a, **k: long_clip)
    monkeypatch.setattr("minibot.agent.robot_agent.wait_for_voice",
                        lambda *a, **k: None)
    monkeypatch.setattr("minibot.agent.robot_agent.speech_ms",
                        lambda *a, **k: 5000.0)
    monkeypatch.setattr("minibot.agent.robot_agent.trim_tail",
                        lambda wav, *a, **k: wav)

    class Reply:
        text, error, interrupted, tool_calls = "", "", False, []

    async def complete_turn(timeout=90.0):
        return Reply()

    agent.ai.complete_turn = complete_turn
    run(agent.listen_and_respond())
    assert len(agent.ai.audio) == 1
    assert agent.ai.texts == []                    # no wake framing when active


def test_the_face_stays_asleep_while_muted():
    """Blinking through listening/thinking at someone you are ignoring is a
    lie about what the robot is doing."""
    agent = build_agent()
    run(agent._dispatch("mute", {}))
    before = list(agent.hw.faces)
    run(agent._face("listening"))
    run(agent._face("thinking"))
    assert agent.hw.faces == before
