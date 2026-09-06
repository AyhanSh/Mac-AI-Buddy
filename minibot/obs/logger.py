"""Structured logging (§20).

Produces the channel-tagged lines the spec asks for:

    [AI]     tool_call: look_at pan=25 tilt=-4
    [ACTION] approved
    [ROBOT]  POST /say -> 200 (31ms)
    [MEMORY] query="what drink does user like"
    [MEMORY] retrieved=3 best_score=0.89

Never log raw audio or image bytes — only sizes and durations.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_CHANNEL_WIDTH = 7


class _Formatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        channel = getattr(record, "channel", "APP")
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        return f"{stamp} [{channel:<{_CHANNEL_WIDTH}}] {record.getMessage()}"


def setup(level: str = "INFO") -> None:
    root = logging.getLogger("minibot")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(_Formatter())
    root.addHandler(h)
    root.propagate = False


class Channel:
    """A named log channel. `log = Channel("ROBOT")` then `log.info(...)`."""

    __slots__ = ("_name", "_log")

    def __init__(self, name: str):
        self._name = name
        self._log = logging.getLogger(f"minibot.{name.lower()}")

    def _emit(self, level: int, msg: str, *args) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, msg, *args, extra={"channel": self._name})

    def debug(self, msg: str, *a) -> None: self._emit(logging.DEBUG, msg, *a)
    def info(self, msg: str, *a) -> None: self._emit(logging.INFO, msg, *a)
    def warn(self, msg: str, *a) -> None: self._emit(logging.WARNING, msg, *a)
    def error(self, msg: str, *a) -> None: self._emit(logging.ERROR, msg, *a)

    @contextmanager
    def timed(self, label: str, level: int = logging.INFO):
        """Times a block and appends the latency, since every layer in the
        spec's observability list is measured in milliseconds."""
        t0 = time.perf_counter()
        holder: dict[str, object] = {}
        try:
            yield holder
        finally:
            ms = (time.perf_counter() - t0) * 1000
            note = holder.get("note")
            suffix = f" {note}" if note else ""
            self._emit(level, f"{label}{suffix} ({ms:.0f}ms)")


AI = Channel("AI")
ACTION = Channel("ACTION")
ROBOT = Channel("ROBOT")
MEMORY = Channel("MEMORY")
VISION = Channel("VISION")
CAMERA = Channel("CAMERA")
SPEECH = Channel("SPEECH")
STATE = Channel("STATE")
AUDIO = Channel("AUDIO")
SOCIAL = Channel("SOCIAL")
APP = Channel("APP")
