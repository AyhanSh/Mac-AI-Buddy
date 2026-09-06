"""Phase 3 gate: the refactor must not change behaviour.

These tests import BOTH the original mac_realtime.py and the new minibot
package and assert the moved functions produce byte-identical results. That is
the only real proof that "moved verbatim" is true rather than aspirational.

mac_realtime.py stays in the tree as the rollback path, so these keep running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mac_realtime as old  # noqa: E402

from minibot.audio import capture as new_capture  # noqa: E402
from minibot.audio import dsp as new_dsp  # noqa: E402
from minibot.audio import vad as new_vad  # noqa: E402


def _tone(ms: int, freq: float = 220.0, amp: float = 6000.0,
          rate: int = 16000) -> bytes:
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    return (np.sin(2 * np.pi * freq * t) * amp).astype("<i2").tobytes()


def _silence(ms: int, rate: int = 16000) -> bytes:
    return np.zeros(int(rate * ms / 1000), dtype="<i2").tobytes()


class TestDspEquivalence:
    @pytest.mark.parametrize("src,dst", [(16000, 24000), (24000, 16000),
                                         (16000, 16000), (16000, 8000)])
    def test_resample(self, src, dst):
        pcm = _tone(120, rate=src)
        assert new_dsp.resample(pcm, src, dst) == old.resample(pcm, src, dst)

    def test_resample_empty(self):
        assert new_dsp.resample(b"", 16000, 24000) == old.resample(b"", 16000, 24000)

    def test_wav_roundtrip(self):
        pcm = _tone(200)
        wav_new = new_dsp.pcm_to_wav(pcm, 16000)
        assert wav_new == old.pcm_to_wav(pcm, 16000)
        assert new_dsp.wav_to_pcm(wav_new) == old.wav_to_pcm(wav_new) == (pcm, 16000)

    def test_frame_rms(self):
        pcm = _tone(300) + _silence(200)
        np.testing.assert_array_equal(new_dsp.frame_rms(pcm, 16000),
                                      old.frame_rms(pcm, 16000))

    def test_frame_rms_shorter_than_one_frame(self):
        pcm = _tone(2)
        assert len(new_dsp.frame_rms(pcm, 16000)) == len(old.frame_rms(pcm, 16000))

    def test_constants_match_firmware(self):
        assert new_dsp.BOT_RATE == old.BOT_RATE == 16000
        assert new_dsp.MIC_FRAME_MS == old.MIC_FRAME_MS == 16
        assert new_dsp.WAV_GAIN == old.WAV_GAIN == 16


class TestRoomEquivalence:
    @pytest.mark.parametrize("floor", [10.0, 60.0, 200.0, 900.0])
    def test_floor_still_matches(self, floor):
        """The floor estimator is unchanged — only the thresholds moved."""
        assert new_vad.Room(floor).floor == old.Room(floor).floor

    @pytest.mark.parametrize("floor", [10.0, 60.0, 200.0, 900.0])
    def test_thresholds_deliberately_lower_than_the_original(self, floor):
        """A DELIBERATE break of the Phase 3 equivalence gate (2026-09-03).

        The original onset of floor x 3.0 came from mac_realtime.py and was
        never reachable on this hardware: measured live, the floor idles near
        78 and speech peaks around 191, so onset wanted 234 and not one sample
        in 42 crossed it. The robot sat there looking dead.

        Lowering it is the fix, and it is safe in this direction because
        has_speech() discards a false trigger before any API call, while a
        missed trigger is indistinguishable from broken hardware.
        """
        new, before = new_vad.Room(floor), old.Room(floor)
        assert new.onset < before.onset
        assert new.endpoint < before.endpoint

    def test_speech_measured_on_this_robot_actually_triggers(self):
        """Guards the regression directly: the numbers seen on the bench."""
        r = new_vad.Room(78.0)
        assert 191 > r.onset, "measured speech peak must cross onset"
        assert 133 < r.onset, "measured room noise must not cross onset"

    def test_onset_above_endpoint(self):
        """Hysteresis: starting a turn must cost more than continuing one, or
        sentences end on their own quiet consonants."""
        for floor in (5.0, 50.0, 500.0):
            r = new_vad.Room(floor)
            assert r.onset > r.endpoint > r.floor

    def test_loud_readings_do_not_raise_the_floor(self):
        r = new_vad.Room(50.0)
        before = r.floor
        for _ in range(40):
            r.observe(5000.0)
        assert r.floor == before

    def test_quiet_readings_track(self):
        r = new_vad.Room(50.0)
        for _ in range(40):
            r.observe(20.0)
        assert r.floor == pytest.approx(20.0)


class TestCaptureEquivalence:
    def _wav(self):
        return new_dsp.pcm_to_wav(_silence(150) + _tone(500) + _silence(1000), 16000)

    def test_has_speech(self):
        wav = self._wav()
        r_new, r_old = new_vad.Room(20.0), old.Room(20.0)
        assert new_capture.has_speech(wav, r_new) == old.has_speech(wav, r_old) is True

    def test_silence_is_not_speech(self):
        wav = new_dsp.pcm_to_wav(_silence(1500), 16000)
        r_new, r_old = new_vad.Room(20.0), old.Room(20.0)
        assert new_capture.has_speech(wav, r_new) == old.has_speech(wav, r_old) is False

    def test_trim_tail_identical(self):
        wav = self._wav()
        r_new, r_old = new_vad.Room(20.0), old.Room(20.0)
        assert new_capture.trim_tail(wav, r_new) == old.trim_tail(wav, r_old)

    def test_trim_tail_shortens(self):
        wav = self._wav()
        out = new_capture.trim_tail(wav, new_vad.Room(20.0))
        assert len(out) < len(wav)

    def test_trim_tail_keeps_silence_untouched(self):
        """Nothing loud means nothing to anchor a trim to; return it whole."""
        wav = new_dsp.pcm_to_wav(_silence(800), 16000)
        assert new_capture.trim_tail(wav, new_vad.Room(20.0)) == wav


class TestExpressionSet:
    def test_matches_firmware_enum(self):
        """The 15 names in ai_mini_bot.ino's emotionName(), in enum order."""
        from minibot.robot.esp32_client import EMOTIONS
        assert EMOTIONS == old.EMOTIONS
        assert EMOTIONS == [
            "neutral", "happy", "sad", "angry", "surprised", "thinking",
            "listening", "talking", "sleep", "searching", "loading",
            "scanning", "wifi", "memory", "saving",
        ]

    def test_firmware_source_agrees(self):
        """Guards against the .ino and the Mac drifting apart."""
        ino = (Path(__file__).resolve().parent.parent / "ai_mini_bot.ino").read_text()
        from minibot.robot.esp32_client import EMOTIONS
        for name in EMOTIONS:
            assert f'"{name}"' in ino, f"{name} missing from firmware"
