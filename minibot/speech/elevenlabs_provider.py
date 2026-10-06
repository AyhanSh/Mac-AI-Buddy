"""ElevenLabs cloned-voice synthesis (§14).

Streaming logic moved verbatim from mac_realtime.py. pcm_16000 lands exactly on
the firmware's 16 kHz, so nothing is resampled on this path, and chunks are
pushed to the robot as they arrive so it starts speaking before synthesis
finishes.

Added: a cancel flag so stop()/interrupt() can abandon an utterance mid-stream
instead of waiting for the whole thing to synthesize.
"""

from __future__ import annotations

import re
import threading

import requests

from ..obs.logger import SPEECH
from ..robot.hardware import RobotHardware
from .provider import SpeechProvider

# A desk robot answering in one or two sentences never needs more than this.
# Well past that means something went wrong upstream — observed live: a
# reasoning model leaked its chain of thought into the reply and the robot
# began reading a JSON fragment and the text of a safety policy out loud.
SPEAKABLE_WARN_CHARS = 600
_CODE_FENCE = re.compile(r"```.*?```", re.S)
# Emoji, pictographs, and the joiners/variation selectors that glue them
# together. Told not to, the local model still ends a line with "🤖" now and
# then, and a voice reading "robot face" out loud is worse than silence.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF"
    "\uFE0F\u200D\u20E3]+")


def speakable(text: str) -> str:
    """Last check before anything reaches the speaker.

    Strips fenced code blocks, which are never speech, and flags text far
    longer than a desk robot should ever say. Nothing is truncated — a real
    long answer is still delivered in full — but the log makes a leak obvious
    instead of leaving you listening to it.
    """
    cleaned = _CODE_FENCE.sub(" ", text).strip()
    if cleaned != text.strip():
        SPEECH.warn("stripped a code block from the reply before speaking")
    cleaned = " ".join(_EMOJI.sub(" ", cleaned).split())
    if len(cleaned) > SPEAKABLE_WARN_CHARS:
        SPEECH.warn(f"reply is {len(cleaned)} chars — unusually long for a "
                    f"spoken answer; check for leaked model reasoning")
    return cleaned

ELEVEN_URL = "https://api.elevenlabs.io/v1"
# Only 44.1 kHz PCM needs a Pro subscription; 16 kHz is on every plan and is
# exactly what the robot wants.
ELEVEN_FORMAT = "pcm_16000"
ELEVEN_SETTINGS = {
    "stability": 0.45,          # lower = more expressive, more variable
    "similarity_boost": 0.80,   # how tightly it hugs your clone
    "style": 0.30,              # 0 for a neutral read, higher for character
    "use_speaker_boost": True,
}


def list_voices(api_key: str) -> list[dict]:
    r = requests.get(f"{ELEVEN_URL}/voices",
                     headers={"xi-api-key": api_key}, timeout=15)
    r.raise_for_status()
    return r.json().get("voices", [])


class ElevenLabsProvider(SpeechProvider):
    def __init__(self, hardware: RobotHardware, api_key: str, voice_id: str,
                 model: str = "eleven_flash_v2_5", chunk_bytes: int = 16000):
        self.hw = hardware
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.chunk_bytes = chunk_bytes      # ~500 ms at 16 kHz mono
        # The first piece goes to the robot at half size so it starts talking
        # sooner. ElevenLabs streams far faster than real time, so the next
        # full chunk lands well before this one finishes playing.
        self.first_chunk_bytes = chunk_bytes // 2
        # One kept-alive connection: a fresh one costs a TLS handshake per
        # reply, and with sentence streaming there are several per reply.
        self._http = requests.Session()
        self._cancel = threading.Event()
        self._speaking = threading.Event()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.voice_id)

    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self) -> None:
        self._cancel.set()
        try:
            self.hw.stop_audio()
        except Exception as e:
            SPEECH.warn(f"stop failed: {e!r}")

    def interrupt(self) -> None:
        SPEECH.info("interrupted")
        self.stop()

    def synthesize(self, text: str) -> None:
        text = speakable(text or "")
        if not text:
            return
        if not self.configured:
            SPEECH.error("ElevenLabs not configured (need key + voice id)")
            return

        self._cancel.clear()
        self._speaking.set()
        try:
            with SPEECH.timed("synthesized") as t:
                sent = self._stream(text)
                t["note"] = f'"{text[:48]}" {sent}B'
        except Exception as e:
            SPEECH.error(f"synthesis failed: {e!r}")
            # Don't leave the robot mute — a chirp makes the failure audible.
            try:
                self.hw.play_audio(b"")
            except Exception:
                pass
        finally:
            self._speaking.clear()

    def _stream(self, text: str) -> int:
        url = f"{ELEVEN_URL}/text-to-speech/{self.voice_id}/stream"
        body = {"text": text, "model_id": self.model,
                "voice_settings": ELEVEN_SETTINGS}
        sent = 0
        with self._http.post(url, json=body, stream=True, timeout=60,
                           headers={"xi-api-key": self.api_key,
                                    "Content-Type": "application/json"},
                           params={"output_format": ELEVEN_FORMAT}) as r:
            if r.status_code != 200:
                raise RuntimeError(f"elevenlabs {r.status_code}: {r.text[:200]}")
            buf = bytearray()
            for part in r.iter_content(chunk_size=4096):
                if self._cancel.is_set():
                    SPEECH.info("stream abandoned mid-utterance")
                    break
                if not part:
                    continue
                buf += part
                while len(buf) >= (size := self.chunk_bytes if sent
                                   else self.first_chunk_bytes):
                    if self._cancel.is_set():
                        break
                    self.hw.play_audio(bytes(buf[:size]))
                    sent += size
                    del buf[:size]
            if buf and not self._cancel.is_set():
                self.hw.play_audio(bytes(buf))
                sent += len(buf)
        return sent
