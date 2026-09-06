"""Turn capture from the robot's microphone.

Moved verbatim from mac_realtime.py. Endpointing happens on the ESP32 because
/mic owns the core for the whole recording and nothing can reach in to stop it
partway; record_chunked() is the fallback for firmware without that support.
"""

from __future__ import annotations

import numpy as np

from ..obs.logger import AUDIO
from .dsp import BOT_RATE, MIC_FRAME_MS, WAV_GAIN, frame_rms, pcm_to_wav, wav_to_pcm
from .vad import Room


def source_gain(client) -> float:
    """How much louder a source's WAV is than its own .level() readings.

    The ESP32 amplifies on the way out of /mic and so needs WAV_GAIN; a source
    that records and meters in the same units (MacMicSource) declares 1. Any
    duck-typed source that doesn't say gets the firmware's value, which is what
    every caller assumed before AUDIO_INPUT existed.
    """
    return float(getattr(client, "wav_gain", WAV_GAIN))


def duration_ms(wav: bytes) -> float:
    """How long the clip is, in milliseconds — regardless of what is in it."""
    pcm, rate = wav_to_pcm(wav)
    return len(pcm) / 2 / rate * 1000


def speech_ms(wav: bytes, room: Room, gain: float = WAV_GAIN) -> float:
    """Milliseconds of the clip that are above the endpoint threshold."""
    pcm, rate = wav_to_pcm(wav)
    r = frame_rms(pcm, rate)
    return float((r > room.endpoint * gain).sum()) * MIC_FRAME_MS


def has_speech(wav: bytes, room: Room, min_ms: int = 120,
               gain: float = WAV_GAIN) -> bool:
    """Did anything actually get said? Guards against a door slam opening a
    turn and burning an API round trip on a second of room tone."""
    return speech_ms(wav, room, gain) >= min_ms


def trim_tail(wav: bytes, room: Room, keep_ms: int = 250,
              gain: float = WAV_GAIN) -> bytes:
    """Drop the dead air the endpointer leaves behind. The model does not need
    to sit through the silence that ended your sentence, and it is charged for
    the audio either way."""
    pcm, rate = wav_to_pcm(wav)
    r = frame_rms(pcm, rate)
    loud = np.nonzero(r > room.endpoint * gain)[0]
    if not len(loud):
        return wav
    keep = int((loud[-1] + 1) * MIC_FRAME_MS + keep_ms) * rate // 1000
    return pcm_to_wav(pcm[: keep * 2], rate)


def record_chunked(client, room: Room, max_ms: int, silence_ms: int,
                   chunk_ms: int = 900) -> bytes:
    """Endpointing for firmware that cannot do it itself. /mic only records a
    fixed span, so take it in chunks and stop once the tail goes quiet.

    The mic is deaf for a WiFi round trip between chunks, so syllables land in
    the seams. Reflash for the endpointing build if transcripts read badly."""
    gate = room.endpoint * source_gain(client)
    need = max(1, int(round(silence_ms / MIC_FRAME_MS)))
    pcm, rate, took = bytearray(), BOT_RATE, 0
    while took < max_ms:
        chunk, rate = wav_to_pcm(client.record(max_ms=min(chunk_ms, max_ms - took)))
        pcm += chunk
        took += chunk_ms
        r = frame_rms(bytes(pcm), rate)
        if len(r) >= need and (r > gate).any() and (r[-need:] <= gate).all():
            break
    return pcm_to_wav(bytes(pcm), rate)


def capture_turn(client, room: Room, vad: bool, max_ms: int, silence_ms: int,
                 lead_ms: int) -> bytes:
    """Everything the user just said, ending silence_ms after they stop."""
    if vad:
        wav = client.record(max_ms=max_ms, silence_ms=silence_ms,
                            thresh=int(room.endpoint), lead_ms=lead_ms)
    else:
        wav = record_chunked(client, room, max_ms, silence_ms)
    pcm, rate = wav_to_pcm(wav)
    AUDIO.info(f"captured {len(pcm) / 2 / rate:.2f}s @ {rate}Hz")
    return wav
