"""Audio DSP primitives.

Moved verbatim from mac_realtime.py — this code is tuned against the actual
hardware and is deliberately not rewritten.
"""

from __future__ import annotations

import io
import wave

import numpy as np

BOT_RATE = 16000    # AUDIO_RATE / MIC_RATE in the firmware
MIC_FRAME_MS = 16   # matches MIC_FRAME / MIC_RATE in the firmware

# recordMic() strips DC and multiplies 12-bit samples by 16 on the way out, so
# /level readings (raw ADC counts) need this factor to compare against the WAV.
#
# This is a property of the ESP32 firmware, NOT of audio in general: a source
# whose .level() and .record() report the same units has a gain of 1. Read it
# off the source as `source_gain(mic)` rather than importing it directly, or
# the Mac microphone gets judged against a threshold 16x too high and every
# turn is thrown away as a false trigger. See MacMicSource.wav_gain.
WAV_GAIN = 16


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    """Linear interpolation, good enough for speech."""
    if src == dst or not pcm:
        return pcm
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = int(len(a) * dst / src)
    if n_out < 1:
        return b""
    xi = np.linspace(0, len(a) - 1, n_out)
    out = np.interp(xi, np.arange(len(a)), a)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


def frame_rms(pcm: bytes, rate: int) -> np.ndarray:
    """RMS per MIC_FRAME_MS window, each measured about its own mean so a
    drifting bias cannot read as loudness. Same shape as the firmware's VAD."""
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = max(1, int(rate * MIC_FRAME_MS / 1000))
    if len(a) < n:
        return np.zeros(0, np.float32)
    f = a[: len(a) - len(a) % n].reshape(-1, n)
    return np.sqrt(((f - f.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
