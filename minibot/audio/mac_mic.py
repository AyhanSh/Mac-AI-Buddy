"""Microphone input from the Mac itself, as an alternative to the robot's mic.

Duck-types the same interface Esp32Client already exposes — .level() and
.record(max_ms, silence_ms, thresh, lead_ms) — so calibrate(), wait_for_voice()
and capture_turn() in audio/vad.py and audio/capture.py work unmodified. Room's
onset/endpoint math never sees raw ADC counts vs. Core Audio samples; it just
calibrates to whatever .level() reports, so the same code adapts to either
source automatically.

Endpointing mirrors the firmware's recordMic(): per-frame RMS about the
frame's own mean, wait up to lead_ms for speech to start, then stop after
silence_ms of continuous quiet.

The stream is opened ONCE in __init__ and kept running for the object's whole
life. An earlier version opened a fresh CoreAudio InputStream for every single
.level() call (needed ~10x during calibration alone, then continuously during
voice-wait polling). The first frame after a stream starts is reliably a
zero/near-zero "priming" buffer before the hardware settles, and reading
exactly that frame, every call, is what produced a live calibration of
floor=0 — the mic had real audio, the code was just always sampling the
one guaranteed-silent instant. Verified against raw sounddevice output.

sounddevice is an optional dependency — only required when AUDIO_INPUT=mac.
"""

from __future__ import annotations

import queue
from collections import deque

import numpy as np

from ..obs.logger import AUDIO
from .dsp import BOT_RATE, MIC_FRAME_MS, pcm_to_wav

FRAME_SAMPLES = int(BOT_RATE * MIC_FRAME_MS / 1000)   # 256 @ 16kHz
LEVEL_FRAMES = 4   # frames combined per .level() reading (~64ms), see level()


class MacMicUnavailable(RuntimeError):
    pass


def _require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as e:
        raise MacMicUnavailable(
            "AUDIO_INPUT=mac needs the 'sounddevice' package: "
            "pip install sounddevice") from e
    return sd


class MacMicSource:
    """The MacBook's own microphone, captured via Core Audio.

    One InputStream, opened at construction and left running. .level() reads
    the freshest queued frame (discarding any backlog, so an idle period
    before someone speaks never makes the meter lag). .record() drains
    frames from the same live queue in order, so nothing is skipped mid-turn.
    """

    # Both .level() and .record() report the same Core Audio int16 samples, so
    # there is no gain between them — unlike the ESP32, where /mic amplifies by
    # 16 and /level does not. Getting this wrong is not subtle: has_speech()
    # then wants speech 16x above the endpoint threshold, throws every real
    # turn away as a false trigger, and the robot listens without ever
    # answering. Measured live: room tone ~50 RMS, speech peaking ~4000, and a
    # gate of endpoint x 16 = 864 that ordinary conversation never clears.
    wav_gain = 1

    def __init__(self, rate: int = BOT_RATE, device: int | str | None = None):
        self.sd = _require_sounddevice()
        self.rate = rate
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()

        try:
            info = self.sd.query_devices(device, kind="input")
            AUDIO.info(f"mac mic: {info['name']} @ {rate}Hz")
            self._stream = self.sd.InputStream(
                samplerate=rate, channels=1, dtype="int16",
                blocksize=FRAME_SAMPLES, device=device, callback=self._on_audio)
            self._stream.start()
        except Exception as e:
            raise MacMicUnavailable(f"could not open input stream: {e}") from e

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            AUDIO.debug(f"mac mic status: {status}")
        self._q.put(indata[:, 0].copy())

    # -- duck-typed to match Esp32Client -----------------------------
    def level(self) -> dict:
        """RMS of the most recent frame. Drains any backlog first — after an
        idle stretch several frames queue up, and the meter should report
        NOW, not the oldest thing still waiting to be read.

        Combines LEVEL_FRAMES consecutive frames into one measurement rather
        than reading a single 16ms window. A real acoustic mic's RMS over one
        such window swings wildly moment to moment — measured live: 12 to 121
        across 60 single-frame reads in the same few seconds of room tone. A
        10-sample calibration built from single-frame reads can land on an
        unlucky run of low draws and compute a near-zero floor even though the
        room is not remotely that quiet. Concatenating a few frames first
        (~64ms) — the same idea as the firmware's own /level, which averages
        over a 20ms window rather than one instant — smooths that out; measured
        min never dropped below 30 over the same span.
        """
        self._drain()
        frames = []
        for _ in range(LEVEL_FRAMES):
            try:
                frames.append(self._q.get(timeout=1.0))
            except queue.Empty:
                break
        if not frames:
            return {"rms": 0, "peak": 0}

        window = np.concatenate(frames)
        dc = window.mean()
        centered = window - dc
        rms = float(np.sqrt((centered ** 2).mean()))
        peak = float(np.abs(centered).max())
        return {"rms": rms, "peak": peak}

    def record(self, max_ms: int = 3500, silence_ms: int = 0, thresh: float = 0,
               lead_ms: int = 1500) -> bytes:
        """Endpointed recording, same semantics as Esp32Client.record().

        Drops any frames queued from before this call (idle chatter picked up
        while nothing was listening) so the recording starts from now, not
        from whatever the mic happened to be capturing during the wait.

        The endpoint decision is taken over the same ~64 ms window .level()
        uses, not over one 16 ms frame. Single-frame RMS on a real acoustic mic
        swings far enough on room tone alone (measured: 17 to 132 with a
        threshold of 54) that an isolated loud frame keeps resetting the quiet
        counter — the recording then never ends early and every turn runs the
        full max_ms. See the note on level() for the same measurement.
        """
        self._drain()
        max_samples = int(self.rate * max_ms / 1000)
        buf: list[np.ndarray] = []
        recent: deque[np.ndarray] = deque(maxlen=LEVEL_FRAMES)
        n = 0
        speech = False
        quiet_ms = 0.0
        elapsed_ms = 0.0

        while n < max_samples:
            try:
                frame = self._q.get(timeout=1.0)
            except queue.Empty:
                break
            buf.append(frame)
            n += len(frame)
            elapsed_ms += MIC_FRAME_MS

            if silence_ms:
                recent.append(frame)
                window = np.concatenate(recent)
                dc = window.mean()
                rms = float(np.sqrt(((window - dc) ** 2).mean()))
                if rms > thresh:
                    speech = True
                    quiet_ms = 0.0
                elif speech:
                    quiet_ms += MIC_FRAME_MS
                    if quiet_ms >= silence_ms:
                        break
                elif elapsed_ms >= lead_ms:
                    break   # nobody started talking

        pcm = np.concatenate(buf) if buf else np.zeros(0, dtype=np.int16)
        return pcm_to_wav(pcm.astype("<i2").tobytes(), self.rate)

    def _drain(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
