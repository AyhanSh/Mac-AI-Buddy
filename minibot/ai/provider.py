"""AI provider interface (§1).

The rest of the robot programs against this and never imports a vendor SDK.
Both implementations emit TEXT only; speech is always synthesized by the
SpeechProvider, so the voice is the same whichever brain is driving.

Tool calls are resolved inside the provider because both APIs require the
result to go back on the same live session. The provider invokes the registered
handler and loops until the model produces a final text answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence


@dataclass(frozen=True)
class ToolSpec:
    """A semantic robot capability offered to the model.

    `blocking` maps to Gemini's `behavior: "BLOCKING"`: the model waits for the
    robot to actually finish before continuing, which is what lets the rule
    "never claim a physical action succeeded until confirmed" hold.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    blocking: bool = True


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class AIResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    interrupted: bool = False
    error: str | None = None


# handler(name, arguments) -> JSON-serializable result
ToolHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

# A tool result may carry raw JPEG bytes under this key. The provider lifts it
# out and attaches it to the tool result by whatever mechanism its API offers,
# then removes the key before the rest of the dict is serialized as JSON.
#
# This exists because a camera tool's *result* is the picture. Pushing the frame
# in as out-of-band video instead leaves the model blocked on a tool result that
# never contains what it asked for.
IMAGE_RESULT_KEY = "_image"


class AIProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    def register_tools(self, tools: Sequence[ToolSpec],
                       handler: ToolHandler) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes, rate: int) -> None:
        """Queue mono int16 LE audio as user input."""

    @abstractmethod
    async def send_frame(self, jpeg: bytes) -> None:
        """Queue a camera frame as visual context."""

    @abstractmethod
    async def send_text(self, text: str, *, turn_complete: bool = True) -> None:
        """Queue a text user turn.

        turn_complete=False appends the text as context without asking the
        model to respond yet — used to inject retrieved memory ahead of the
        user's actual message (§9), the same way a tool-returned photo is
        attached ahead of the model's next turn.
        """

    @abstractmethod
    async def complete_turn(self, timeout: float = 90.0) -> AIResponse:
        """Close the current input turn, run tool calls to completion, and
        return the model's final text."""

    @abstractmethod
    async def interrupt(self) -> None:
        """Abandon the in-flight response."""
