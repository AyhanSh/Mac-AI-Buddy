"""MacMicSource tests, entirely mocked — no real audio device involved.

A live device needs a one-time macOS microphone permission grant that only a
human can click through; the running Bash tool that first exercised this
hung indefinitely on that system dialog (2026-09-03). These tests stand in
for the parts that don't need the human in the loop: framing, endpointing
math, and the duck-typed interface contract with Room/capture_turn.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import types as pytypes
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.audio.dsp import MIC_FRAME_MS  # noqa: E402


def _install_fake_sounddevice(frames: list[np.ndarray], device_name="Fake Mic"):
    """A minimal stand-in for the `sounddevice` module. Delivers `frames` to
    the InputStream callback one at a time on a background thread, spaced by
    MIC_FRAME_MS — real CoreAudio delivers progressively as audio arrives, not
    all at once, and MacMicSource's persistent-stream design depends on that:
    an early version fired every frame synchronously inside .start(), which
    hid the exact bug under test here (reading a stream's first, always-empty
    "priming" frame) by never producing one in the first place."""
    fake = pytypes.ModuleType("sounddevice")

    def query_devices(device=None, kind=None):
        return {"name": device_name}

    class InputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, device,
                    callback):
            self.callback = callback
            self._thread = None
            self._stop = threading.Event()

        def start(self):
            def feed():
                for f in frames:
                    if self._stop.is_set():
                        return
                    time.sleep(MIC_FRAME_MS / 1000.0)
                    self.callback(f.reshape(-1, 1), len(f), None, None)
            self._thread = threading.Thread(target=feed, daemon=True)
            self._thread.start()

        def stop(self):
            self._stop.set()

        def close(self):
            pass

    fake.query_devices = query_devices
    fake.InputStream = InputStream
    sys.modules["sounddevice"] = fake
    return fake


@pytest.fixture(autouse=True)
def _clean_sounddevice_module():
    sys.modules.pop("sounddevice", None)
    sys.modules.pop("minibot.audio.mac_mic", None)
    yield
    sys.modules.pop("sounddevice", None)


def _frame(rms_level: float, n: int = 256) -> np.ndarray:
    """A frame of int16 samples with roughly the given RMS about zero."""
    return (np.full(n, rms_level, dtype=np.float64)
            * np.sign(np.sin(np.arange(n)))).astype(np.int16)


class TestImportGuard:
    def test_missing_sounddevice_raises_clear_error(self):
        sys.modules.pop("sounddevice", None)
        sys.modules["sounddevice"] = None  # force ImportError on `import`

        import builtins
        real_import = builtins.__import__

        def blocking_import(name, *a, **kw):
            if name == "sounddevice":
                raise ImportError("no module")
            return real_import(name, *a, **kw)

        builtins.__import__ = blocking_import
        try:
            from minibot.audio.mac_mic import MacMicUnavailable, _require_sounddevice
            with pytest.raises(MacMicUnavailable, match="pip install sounddevice"):
                _require_sounddevice()
        finally:
            builtins.__import__ = real_import
            sys.modules.pop("sounddevice", None)


class TestLevel:
    def test_reports_rms_over_the_combined_window(self):
        # More than LEVEL_FRAMES supplied so the call never falls back to the
        # 1s starvation timeout waiting for a frame that isn't coming.
        _install_fake_sounddevice([_frame(500.0)] * 8)
        from minibot.audio.mac_mic import MacMicSource
        m = MacMicSource()
        out = m.level()
        assert out["rms"] == pytest.approx(500.0, rel=0.05)
        assert "peak" in out

    def test_no_frame_available_returns_zero_not_raise(self):
        _install_fake_sounddevice([])   # callback never fires
        from minibot.audio.mac_mic import MacMicSource
        m = MacMicSource()
        assert m.level() == {"rms": 0, "peak": 0}


class TestRecordEndpointing:
    """Same endpointing contract as Esp32Client.record(): stop after
    silence_ms once speech has been heard, or after lead_ms if it never
    starts, capped at max_ms. capture_turn()/Room don't know which is which."""

    def test_stops_after_silence_following_speech(self):
        # 16ms frames: 3 quiet, 3 loud (speech), then quiet until it should stop
        frames = [_frame(20)] * 3 + [_frame(500)] * 3 + [_frame(20)] * 20
        _install_fake_sounddevice(frames)
        from minibot.audio.mac_mic import MacMicSource
        from minibot.audio.dsp import wav_to_pcm

        m = MacMicSource()
        wav = m.record(max_ms=5000, silence_ms=160, thresh=100, lead_ms=1000)
        pcm, rate = wav_to_pcm(wav)
        got_ms = len(pcm) / 2 / rate * 1000
        # 3 quiet + 3 loud + ~10 quiet frames (160ms/16ms) before stopping
        assert 200 < got_ms < 400

    def test_gives_up_if_nobody_speaks(self):
        frames = [_frame(20)] * 200   # never crosses thresh
        _install_fake_sounddevice(frames)
        from minibot.audio.mac_mic import MacMicSource
        from minibot.audio.dsp import wav_to_pcm

        m = MacMicSource()
        wav = m.record(max_ms=5000, silence_ms=160, thresh=100, lead_ms=100)
        pcm, rate = wav_to_pcm(wav)
        got_ms = len(pcm) / 2 / rate * 1000
        assert got_ms <= 130   # stopped near lead_ms, not the 5s cap

    def test_capped_at_max_ms_even_with_continuous_speech(self):
        frames = [_frame(500)] * 100
        _install_fake_sounddevice(frames)
        from minibot.audio.mac_mic import MacMicSource
        from minibot.audio.dsp import wav_to_pcm

        m = MacMicSource()
        wav = m.record(max_ms=200, silence_ms=1000, thresh=100, lead_ms=1000)
        pcm, rate = wav_to_pcm(wav)
        got_ms = len(pcm) / 2 / rate * 1000
        assert got_ms <= 220

    def test_silence_ms_zero_records_full_span_unconditionally(self):
        frames = [_frame(20)] * 30
        _install_fake_sounddevice(frames)
        from minibot.audio.mac_mic import MacMicSource
        from minibot.audio.dsp import wav_to_pcm

        m = MacMicSource()
        wav = m.record(max_ms=500, silence_ms=0)
        pcm, rate = wav_to_pcm(wav)
        got_ms = len(pcm) / 2 / rate * 1000
        assert got_ms == pytest.approx(480, abs=40)   # ~30 frames * 16ms


class TestDuckTypedInterface:
    """calibrate()/wait_for_voice()/capture_turn() only call .level() and
    .record(...) — a MacMicSource must be a drop-in for Esp32Client there."""

    def test_calibrate_accepts_a_mac_mic(self):
        # calibrate(n=5) makes 5 level() calls, each consuming LEVEL_FRAMES —
        # supply comfortably more than 5*LEVEL_FRAMES so none of them starve.
        _install_fake_sounddevice([_frame(50)] * 40)
        from minibot.audio.mac_mic import MacMicSource
        from minibot.audio.vad import calibrate

        m = MacMicSource()
        room = calibrate(m, n=5)
        assert room.floor > 0

    def test_wave_bytes_are_valid_wav(self):
        _install_fake_sounddevice([_frame(100)] * 10)
        from minibot.audio.mac_mic import MacMicSource

        m = MacMicSource()
        wav = m.record(max_ms=200, silence_ms=0)
        assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


class TestStreamPriming:
    """Regression for the live session of 2026-09-05: calibration measured
    floor=0 against a mic that was actually producing real audio. Root cause
    was opening a fresh CoreAudio stream per .level() call and always reading
    its first frame — which is reliably a zero/near-zero "priming" buffer
    before the hardware settles. Confirmed by pulling raw sounddevice output
    directly: real frames were nonzero, but the per-call restart pattern threw
    the stream away and read only that first instant, every time.
    """

    def test_one_stream_is_opened_for_the_whole_object_lifetime(self, monkeypatch):
        opened = []
        # 3 level() calls * LEVEL_FRAMES each, with margin so none starve.
        fake = _install_fake_sounddevice([_frame(50)] * 20)
        real_input_stream = fake.InputStream

        def counting_input_stream(*a, **kw):
            s = real_input_stream(*a, **kw)
            opened.append(s)
            return s

        fake.InputStream = counting_input_stream
        from minibot.audio.mac_mic import MacMicSource

        m = MacMicSource()
        m.level()
        m.level()
        m.level()
        assert len(opened) == 1, "level() must not reopen the stream per call"

    def test_priming_frame_does_not_corrupt_the_reading(self):
        """Even if the very first delivered frame is silent (as real hardware
        produces), later real frames must not be starved by a per-call
        stream restart that keeps re-reading only that first instant."""
        frames = [_frame(0)] + [_frame(400)] * 10
        _install_fake_sounddevice(frames)
        from minibot.audio.mac_mic import MacMicSource

        m = MacMicSource()
        time.sleep(0.1)   # let a few real frames land in the queue
        out = m.level()
        assert out["rms"] > 100, "must read a real frame, not the stuck priming one"
