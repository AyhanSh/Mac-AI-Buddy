"""Tool registry (§2).

Semantic robot capabilities offered to the model. Every tool describes WHAT the
robot should do; deterministic code below decides HOW. Nothing here exposes
PWM, PCA9685 channels, I2C or GPIO — those stay inside the hardware layer.

Phase 3 registers the three tools the OpenAI path already had, so behaviour is
unchanged. The registry is the extension point for the rest of §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..ai.provider import ToolSpec
from ..obs.logger import AI

Executor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    spec: ToolSpec
    execute: Executor


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, spec: ToolSpec, execute: Executor) -> None:
        self._tools[spec.name] = Tool(spec, execute)

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Single entry point for every model-requested action.

        Model output is untrusted (§22): an unknown tool name or a failing
        executor returns a structured error rather than raising into the
        provider's event loop.
        """
        pretty = " ".join(f"{k}={v}" for k, v in args.items())
        AI.info(f"tool_call: {name} {pretty}".rstrip())
        tool = self._tools.get(name)
        if tool is None:
            AI.warn(f"unknown tool {name!r}")
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            return await tool.execute(args)
        except Exception as e:
            AI.error(f"tool {name} failed: {e!r}")
            return {"ok": False, "error": str(e)}
