"""Room noise model and voice-onset detection.

Moved verbatim from mac_realtime.py. The thresholds here were tuned against
the MAX4466 and its gain pot; do not "simplify" them without re-measuring.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import requests

from ..obs.logger import AUDIO


class Room:
    """Running estimate of the noise floor, in raw ADC counts, and the two
    thresholds taken from it. Onset sits above endpoint deliberately: it should
    cost more energy to start a turn than to keep one going, so a sentence does
    not end on its own quiet consonants.

    The floor tracks the room rather than being measured once, because the fans
    and the servos and the camera all change what quiet sounds like.

    Multipliers were loosened on 2026-09-03 after a live session where the
    robot never heard anything. Measured: floor 78, speech peaking at 191. The
    old onset of floor x 3.0 wanted 234, so not one sample in 42 crossed it —
    x 3.0 only works when the floor is low, and this MAX4466 idles high. A
    false trigger is cheap because has_speech() throws the clip away before any
    API call, whereas a missed trigger looks like a dead robot.
    """

    def __init__(self, floor: float = 60.0,
                 onset_mult: float = 2.0, onset_margin: float = 35.0,
                 endpoint_mult: float = 1.5, endpoint_margin: float = 18.0):
        self.hist: deque[float] = deque([floor] * 8, maxlen=40)
        self.onset_mult = onset_mult
        self.onset_margin = onset_margin
        self.endpoint_mult = endpoint_mult
        self.endpoint_margin = endpoint_margin

    @property
    def floor(self) -> float:
        return float(np.median(self.hist))

    @property
    def onset(self) -> float:
        f = self.floor
        return max(f * self.onset_mult, f + self.onset_margin)

    @property
    def endpoint(self) -> float:
        f = self.floor
        return max(f * self.endpoint_mult, f + self.endpoint_margin)

    def observe(self, rms: float) -> None:
        # Only quiet readings update the floor, or the first loud sentence
        # would teach it that shouting is normal.
        if rms < self.onset:
            self.hist.append(float(rms))


def calibrate(client, n: int = 10, **room_kw) -> Room:
    """`client` is anything exposing .level() -> {"rms": ...}."""
    seen: list[float] = []
    for _ in range(n):
        try:
            seen.append(client.level()["rms"])
        except requests.RequestException:
            pass
    room = Room(**room_kw)
    room.hist.clear()
    room.hist.extend(float(s) for s in (seen or [60.0]))
    AUDIO.info(f"calibrated floor={room.floor:.0f} onset={room.onset:.0f} "
               f"endpoint={room.endpoint:.0f}")
    return room


def wait_for_voice(client, room: Room, consecutive: int = 2,
                   poll: float = 0.02) -> None:
    """Block until sustained sound. Each /level call already takes ~20 ms on
    the device plus a round trip, so two hits is roughly 150 ms of evidence —
    enough to ignore a single click, short enough that little is clipped off
    the front of your first word before /mic takes over."""
    hits = 0
    while True:
        try:
            rms = client.level()["rms"]
        except requests.RequestException:
            time.sleep(0.4)
            continue
        room.observe(rms)
        hits = hits + 1 if rms > room.onset else 0
        if hits >= consecutive:
            return
        time.sleep(poll)
