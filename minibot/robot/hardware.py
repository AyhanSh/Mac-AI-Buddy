"""Hardware abstraction layer (§4).

`RobotHardware` is the capability surface the rest of the robot programs
against. `Esp32RobotHardware` is the only implementation that knows an ESP32
exists; swapping in a simulator or a different board means implementing this
interface and nothing else changes.

Note the busy flags. The firmware's setHead() already busy-waits while a
capture is in flight, so the Mac mirrors that state rather than discovering it
by timing out — the scheduler uses it to order work instead of racing.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..obs.logger import CAMERA, ROBOT
from .esp32_client import EMOTIONS, Esp32Client, Esp32Unavailable


@dataclass
class RobotStatus:
    camera: bool = False
    oled: bool = False
    servos: bool = False
    audio: bool = False
    mic: bool = False
    speaking: bool = False
    listening: bool = False
    volume: int = 0
    emotion: str = "neutral"
    pan: int = 90
    tilt: int = 90
    rssi: int = 0
    heap: int = 0
    psram: int = 0
    online: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict) -> "RobotStatus":
        known = {f for f in cls.__dataclass_fields__ if f not in ("online", "raw")}
        return cls(**{k: v for k, v in d.items() if k in known}, raw=d)


class RobotHardware(ABC):
    """What the robot can physically do. No transport details leak through."""

    @abstractmethod
    def set_expression(self, expression: str) -> None: ...

    @abstractmethod
    def set_head_position(self, pan: int, tilt: int) -> None: ...

    @abstractmethod
    def center_head(self) -> None: ...

    @abstractmethod
    def capture_image(self) -> bytes: ...

    @abstractmethod
    def play_audio(self, pcm16k: bytes) -> None: ...

    @abstractmethod
    def stop_audio(self) -> None: ...

    @abstractmethod
    def get_status(self) -> RobotStatus: ...

    @abstractmethod
    def is_camera_busy(self) -> bool: ...

    @abstractmethod
    def is_servo_busy(self) -> bool: ...

    @abstractmethod
    def head_position(self) -> tuple[int, int]: ...


class Esp32RobotHardware(RobotHardware):
    def __init__(self, client: Esp32Client):
        self.client = client
        self._camera_lock = threading.Lock()
        self._servo_busy = False
        self._pan = 90
        self._tilt = 90
        self._last_status = RobotStatus()

    # -- expression ------------------------------------------------
    def set_expression(self, expression: str) -> None:
        if expression not in EMOTIONS:
            raise ValueError(f"unknown expression {expression!r}")
        self.client.face(expression)

    # -- head ------------------------------------------------------
    def set_head_position(self, pan: int, tilt: int) -> None:
        # Bounds are enforced upstream by the scheduler; this is the last line
        # of defence before the wire, and the firmware clamps again after that.
        self._servo_busy = True
        try:
            self.client.look(pan, tilt)
            self._pan, self._tilt = pan, tilt
        finally:
            self._servo_busy = False

    def center_head(self) -> None:
        self._servo_busy = True
        try:
            self.client.center()
            self._pan, self._tilt = 90, 90
        finally:
            self._servo_busy = False

    def head_position(self) -> tuple[int, int]:
        return self._pan, self._tilt

    # -- camera ----------------------------------------------------
    def capture_image(self) -> bytes:
        """Holds the camera lock for the whole capture so servo work queues
        behind it, mirroring the firmware's own capturing/setHead handshake."""
        with self._camera_lock:
            with CAMERA.timed("frame captured") as t:
                data = self.client.capture()
                t["note"] = f"{len(data)}B"
            return data

    # -- audio -----------------------------------------------------
    def play_audio(self, pcm16k: bytes) -> None:
        self.client.play(pcm16k)

    def stop_audio(self) -> None:
        self.client.stop()

    # -- state -----------------------------------------------------
    def get_status(self) -> RobotStatus:
        try:
            st = RobotStatus.from_json(self.client.status())
            st.online = True
            self._pan, self._tilt = st.pan, st.tilt
            self._last_status = st
        except Esp32Unavailable:
            self._last_status.online = False
        except Exception as e:
            ROBOT.warn(f"status read failed: {e!r}")
        return self._last_status

    def is_camera_busy(self) -> bool:
        return self._camera_lock.locked()

    def is_servo_busy(self) -> bool:
        return self._servo_busy
