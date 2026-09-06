"""Event bus (§18).

Async pub/sub so services can talk without importing each other. Handlers may
be sync or async; both are awaited correctly.

A handler that raises is logged and skipped — one bad subscriber must never
take down the robot's control loop.
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from ..obs.logger import APP


class Event(str, Enum):
    USER_SPEECH_STARTED = "UserSpeechStarted"
    USER_SPEECH_ENDED = "UserSpeechEnded"
    TRANSCRIPT_RECEIVED = "TranscriptReceived"
    AI_RESPONSE_STARTED = "AIResponseStarted"
    AI_RESPONSE_COMPLETED = "AIResponseCompleted"
    TOOL_CALL_RECEIVED = "ToolCallReceived"
    CAMERA_FRAME_AVAILABLE = "CameraFrameAvailable"
    VISUAL_CONTEXT_UPDATED = "VisualContextUpdated"
    PERSON_DETECTED = "PersonDetected"
    PERSON_LOST = "PersonLost"
    ROBOT_ACTION_STARTED = "RobotActionStarted"
    ROBOT_ACTION_COMPLETED = "RobotActionCompleted"
    ROBOT_ACTION_REJECTED = "RobotActionRejected"
    SPEECH_STARTED = "SpeechStarted"
    SPEECH_STOPPED = "SpeechStopped"
    MEMORY_STORED = "MemoryStored"
    MEMORY_RETRIEVED = "MemoryRetrieved"
    X_POST_PUBLISHED = "XPostPublished"
    ESP32_DISCONNECTED = "Esp32Disconnected"
    ESP32_RECONNECTED = "Esp32Reconnected"
    STATE_CHANGED = "StateChanged"


@dataclass
class Message:
    event: Event
    payload: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Message], Awaitable[None] | None]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[Event, list[Handler]] = defaultdict(list)

    def subscribe(self, event: Event, handler: Handler) -> Callable[[], None]:
        self._subs[event].append(handler)
        return lambda: self.unsubscribe(event, handler)

    def unsubscribe(self, event: Event, handler: Handler) -> None:
        try:
            self._subs[event].remove(handler)
        except ValueError:
            pass

    async def publish(self, event: Event, **payload: Any) -> None:
        msg = Message(event, payload)
        for handler in list(self._subs.get(event, ())):
            try:
                result = handler(msg)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as e:  # a bad subscriber must not stop the robot
                APP.error(f"handler for {event.value} failed: {e!r}")
