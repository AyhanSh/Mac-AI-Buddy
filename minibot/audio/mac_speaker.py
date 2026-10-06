"""The robot's voice through this computer's speakers instead of its own.

Duck-types the two calls ElevenLabsProvider makes on the robot — play_audio()
and stop_audio() — plus the wait_quiet() the conversation loop uses so it does
not start listening while the robot is still talking (and hear itself).

One output stream is opened at construction and kept running, fed from a
buffer. play_audio() only appends, so it returns at once and streamed speech
plays back to back without gaps, the same contract as the robot's /say.
"""

from __future__ import annotations

import threading
import time

from ..obs.logger import AUDIO
from .dsp import BOT_RATE
from .mac_mic import MacMicUnavailable, _require_sounddevice, resolve_device

# Core Audio still has this much queued in the device after our buffer runs
# dry. Listening any sooner would catch the last syllable.
TAIL_S = 0.25


class MacSpeaker:
    def __init__(self, rate: int = BOT_RATE, device: int | str | None = None):
        self.sd = _require_sounddevice()
        self.rate = rate
        self.device = resolve_device(self.sd, device, "output")
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._last_audio = 0.0
        try:
            info = self.sd.query_devices(self.device, kind="output")
            AUDIO.info(f"speaker: {info['name']}")
            self._stream = self.sd.RawOutputStream(
                samplerate=rate, channels=1, dtype="int16",
                device=self.device, callback=self._on_audio)
            self._stream.start()
        except Exception as e:
            raise MacMicUnavailable(f"could not open the speaker: {e}") from e

    def _on_audio(self, outdata, frames, time_info, status) -> None:
        n = len(outdata)
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
        if chunk:
            self._last_audio = time.monotonic()
        outdata[:len(chunk)] = chunk
        if len(chunk) < n:
            outdata[len(chunk):] = b"\0" * (n - len(chunk))

    # -- what ElevenLabsProvider calls --------------------------------
    def play_audio(self, pcm16k: bytes) -> None:
        """Raw int16 LE mono at 16 kHz. Queues and returns, like /say."""
        with self._lock:
            self._buf += pcm16k

    def stop_audio(self) -> None:
        with self._lock:
            self._buf.clear()

    # -- what the conversation loop calls -----------------------------
    def is_playing(self) -> bool:
        with self._lock:
            if self._buf:
                return True
        return time.monotonic() - self._last_audio < TAIL_S

    def wait_quiet(self, timeout: float = 60) -> None:
        t0 = time.monotonic()
        while self.is_playing() and time.monotonic() - t0 < timeout:
            time.sleep(0.05)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
