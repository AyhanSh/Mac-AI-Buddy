"""Which mentions have already been handled.

Autopilot runs unattended, so "have I answered this already?" cannot live in
memory: a restart, a crash, or a laptop lid closing would otherwise make the
robot answer the same person again, and again, every time it woke up. Being
replied to twice by a robot is the most visible way this feature can embarrass
its owner, so the record is on disk and is written BEFORE the reply goes out.

That ordering is deliberate and it is a trade. Writing first means a mention
can be lost — marked handled, then the network drops and the reply never
lands. Writing after would mean a mention can be answered twice. For something
public and irreversible, silence is the better failure.

A remembered set of ids is not enough on its own, which was learned the hard
way (2026-09-09). The mentions endpoint returns "the newest N right now", so
anything outside that window on a first run is never recorded, and when the
window later shifts — someone deletes a tweet, the API pages differently — an
old mention surfaces looking brand new and gets answered. The duplicate guard
does not catch it either: the model writes fresh wording every time, so the
same person receives an endless series of different replies to one tweet.

So the real defence is the high-water mark. X ids are snowflakes, ordered by
time, which gives a total order over every mention that can ever exist:
anything at or below the mark is old, whether or not it is in the set, and
whether or not this machine has ever seen it. The set then handles the only
remaining case — ids above the mark, handled during this window.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Enough history that a mention cannot come back around, bounded so an
# unattended process cannot grow a file forever.
MAX_REMEMBERED = 2000


def _as_int(mention_id: str) -> int | None:
    """Snowflake ids sort by time, but only if they really are numbers."""
    try:
        return int(str(mention_id))
    except (TypeError, ValueError):
        return None


class SeenStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ids: list[str] = []
        self._set: set[str] = set()
        self._high_water: int = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        high = 0
        if isinstance(data, dict):
            high = _as_int(data.get("high_water", 0)) or 0
            data = data.get("replied", [])
        if isinstance(data, list):
            self._ids = [str(i) for i in data][-MAX_REMEMBERED:]
            self._set = set(self._ids)
        # A file written before high_water existed still protects its ids: take
        # the mark from the highest one it recorded rather than starting at 0,
        # which would make every old mention look new all over again.
        self._high_water = max([high] + [i for i in map(_as_int, self._ids)
                                         if i is not None] or [high])

    @property
    def empty(self) -> bool:
        """True the very first time autopilot runs against this account."""
        return not self._set and not self._high_water

    @property
    def high_water(self) -> int:
        return self._high_water

    def is_old(self, mention_id: str) -> bool:
        """Already handled, or from before this store started watching.

        The mark is what makes an old mention drifting back into the API's
        window harmless: it is below the line, so it is old by definition.
        """
        if str(mention_id) in self._set:
            return True
        n = _as_int(mention_id)
        return n is not None and n <= self._high_water

    def __contains__(self, mention_id: str) -> bool:
        return self.is_old(mention_id)

    def add(self, *mention_ids: str) -> None:
        for mention_id in mention_ids:
            mention_id = str(mention_id)
            n = _as_int(mention_id)
            if n is not None and n > self._high_water:
                self._high_water = n
            if mention_id in self._set:
                continue
            self._set.add(mention_id)
            self._ids.append(mention_id)
        if len(self._ids) > MAX_REMEMBERED:
            dropped, self._ids = self._ids[:-MAX_REMEMBERED], self._ids[-MAX_REMEMBERED:]
            self._set.difference_update(dropped)
        self._flush()

    def _flush(self) -> None:
        """Written atomically: a half-written file read on the next boot would
        look like an empty history, and an empty history replies to everything."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"high_water": str(self._high_water),
                           "replied": self._ids}, f)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
