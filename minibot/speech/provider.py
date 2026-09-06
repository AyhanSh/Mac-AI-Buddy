"""Speech synthesis interface (§14)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SpeechProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> None:
        """Speak `text`. Blocks until the audio has been handed to the robot,
        not until playback finishes."""

    @abstractmethod
    def stop(self) -> None:
        """Stop current playback and drop anything queued."""

    @abstractmethod
    def interrupt(self) -> None:
        """Barge-in: stop immediately and abandon the utterance."""

    @abstractmethod
    def is_speaking(self) -> bool: ...
