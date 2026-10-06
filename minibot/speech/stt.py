"""Local speech-to-text.

The cloud providers transcribe audio themselves. A local LLM takes text, so
with AI_PROVIDER=ollama the robot's ears move onto this machine too: nothing
you say leaves the laptop on its way to the model.

mlx-whisper runs Whisper on the Apple GPU through MLX. large-v3-turbo
transcribes a 5 s turn in well under a second on Apple Silicon, which keeps it
out of the critical path next to the LLM itself.

MLX binds its GPU stream to the thread that first touched it, and calling in
from another thread raises "There is no Stream(gpu, 0) in current thread". So
every load and every transcription runs on one dedicated worker thread, never
on the shared executor the rest of the robot uses.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..audio.dsp import resample
from ..obs.logger import AI

WHISPER_RATE = 16000

# Whisper's well-known hallucinations on near-silence: it was trained on
# subtitled video, so a quiet clip comes back as the credits. The agent already
# drops clips without speech, but a cough or a chair can still get through, and
# answering "Thanks for watching!" is worse than answering nothing.
_HALLUCINATIONS = re.compile(
    r"^\W*(thanks? (you )?(so much )?for watching|thank you\.?|you|"
    r"subtitles by.*|please subscribe.*|\.+|"
    r"(and )?(i'?ll )?see you (in the )?next (time|video|one)\.?|"
    r"don'?t forget to (like and )?subscribe.*)\W*$", re.I)

# Below this mean log-probability Whisper is guessing. Whisper's own
# logprob_threshold is the same -1.0. Measured on this machine: real English
# and Russian speech -0.16 to -0.40; room noise decoded as multilingual
# gibberish -8.7.
MIN_AVG_LOGPROB = -1.0
# Above this, the text is one phrase looping ("I'm like, I'm like, …"), which
# is what Whisper makes of music — and it is confident while doing it, so the
# log-probability alone lets it through. Whisper's own threshold, also 2.4.
MAX_COMPRESSION_RATIO = 2.4

# Letters of these scripts belong to these languages. Latin is always allowed:
# brand names ("YouTube") turn up in Latin inside Russian sentences.
_SCRIPTS = {
    "CYRILLIC": {"ru", "uk", "be", "bg", "sr", "mk", "kk"},
    "HIRAGANA": {"ja"}, "KATAKANA": {"ja"}, "CJK": {"ja", "zh"},
    "HANGUL": {"ko"}, "ARABIC": {"ar", "fa", "ur"}, "GREEK": {"el"},
    "HEBREW": {"he"}, "DEVANAGARI": {"hi"}, "THAI": {"th"},
}


def foreign_script(text: str, languages: set[str]) -> str | None:
    """The first script in `text` that none of `languages` is written in.
    Video audio and room noise come back as a salad of scripts ("Motoええええ"),
    which no single language the person speaks would produce."""
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script, langs in _SCRIPTS.items():
            if name.startswith(script) and not (langs & languages):
                return script.lower()
    return None


# How a short, accented "Mac" comes back when it is said as a greeting. Only
# the greeting position is rewritten ("Hey Matt, …" → "Hey Mac, …"): there the
# word is plainly the robot's name, while "Matt" mid-sentence may be a person.
_GREETING_NAME = re.compile(
    r"^(\W*(?:hey|hi|hello|ok|okay|yo)[\s,]+)(?:matt|max|mack|mark|mike|mag|"
    r"man|mak|mac)\b", re.I)


def fix_name(text: str, name: str = "Mac") -> str:
    """'Hey Matt, open YouTube' → 'Hey Mac, open YouTube'."""
    return _GREETING_NAME.sub(lambda m: m.group(1) + name, text, count=1)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


class SpeechToText(ABC):
    @abstractmethod
    async def load(self) -> None:
        """Load the model ahead of the first turn, so the first reply is not
        the one that pays for it."""

    @abstractmethod
    async def transcribe(self, pcm: bytes, rate: int) -> str:
        """Mono int16 LE audio in, text out. Empty string for no speech."""


class MlxWhisperSTT(SpeechToText):
    def __init__(self, model: str = "mlx-community/whisper-large-v3-turbo",
                 language: str | None = None,
                 prompt: str = "Mac, YouTube.",
                 languages: set[str] | None = None):
        self.model = model
        # Primes Whisper's vocabulary with the robot's name. Without it a
        # short, accented "Mac" came back as "Matt" or "man" (observed
        # 2026-10-06), and the robot then corrected the person on it.
        self.prompt = prompt or None
        # None lets Whisper detect it per turn, at the cost of the odd
        # misfire on very short clips. Pin it (STT_LANGUAGE=en) if that bites.
        self.language = language or None
        # The languages the person actually speaks. Anything Whisper decides
        # is another language is sound that is not them — a video playing, a
        # chair, the robot itself — observed: Japanese "see you in the next
        # video" while YouTube was open. Empty = accept every language.
        self.languages = {l.strip().lower() for l in (languages or ()) if l.strip()}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    async def _on_worker(self, fn, *a):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *a)

    async def load(self) -> None:
        with AI.timed(f"stt model loaded ({self.model})"):
            await self._on_worker(self._warm)

    def _warm(self) -> None:
        # There is no public "load" call; one transcription of silence pulls
        # the weights down on first run and into memory on every run.
        self._transcribe(np.zeros(WHISPER_RATE // 2, dtype=np.float32))

    async def transcribe(self, pcm: bytes, rate: int) -> str:
        if rate != WHISPER_RATE:
            pcm = resample(pcm, rate, WHISPER_RATE)
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        text, lang, logprob, ratio = await self._on_worker(self._transcribe, audio)
        reason = self._reject(text, lang, logprob, ratio)
        if reason:
            AI.info(f"ignored {text[:80]!r} — {reason}")
            return ""
        # A primed Whisper can echo its own prompt back on a clip with no
        # real words in it; that is not the person saying it.
        if self.prompt and _norm(text) == _norm(self.prompt):
            AI.debug(f"dropped an echo of the stt prompt: {text!r}")
            return ""
        if _HALLUCINATIONS.match(text):
            AI.debug(f"dropped likely whisper hallucination: {text!r}")
            return ""
        return fix_name(text)

    def _reject(self, text: str, lang: str, logprob: float,
                ratio: float = 0.0) -> str | None:
        """Why this transcript is not the person talking, or None if it is."""
        if not text:
            return None
        if ratio > MAX_COMPRESSION_RATIO:
            return f"one phrase repeating (compression {ratio:.1f}), probably music"
        if logprob < MIN_AVG_LOGPROB:
            return f"low confidence ({logprob:.2f}), probably not speech"
        if self.languages:
            if lang and lang not in self.languages:
                return f"heard as {lang!r}, not one of {sorted(self.languages)}"
            script = foreign_script(text, self.languages)
            if script:
                return f"contains {script} script — noise or a video, not speech"
        return None

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str, float, float]:
        """(text, detected language, mean segment log-probability, highest
        segment compression ratio)."""
        import mlx_whisper   # deferred: only needed with AI_PROVIDER=ollama
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.model, language=self.language,
            # Each turn stands alone. Conditioning on the previous text is how
            # Whisper gets stuck repeating one line across unrelated turns.
            condition_on_previous_text=False, initial_prompt=self.prompt,
            verbose=None)
        text = " ".join(str(result.get("text", "")).split())
        segs = result.get("segments") or []
        logprob = (sum(float(sg.get("avg_logprob", 0.0)) for sg in segs) / len(segs)
                   if segs else 0.0)
        ratio = max((float(sg.get("compression_ratio", 0.0)) for sg in segs),
                    default=0.0)
        return text, str(result.get("language") or ""), logprob, ratio
