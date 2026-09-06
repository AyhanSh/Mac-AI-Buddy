"""OpenAI Realtime provider (§24 — kept switchable during migration).

Port of `Session` from mac_realtime.py, including the response-lifecycle fixes:
one response in flight at a time, tool outputs submitted as they complete, and
a single follow-up issued from response.done rather than one per tool call.

Behaviour change from the original: this runs in the text-output mode that the
`--voice eleven` path already used, so speech always goes through the
SpeechProvider. The model's own voice is not used.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Sequence

import websockets

from ..audio.dsp import resample
from ..obs.logger import AI
from .provider import (
    IMAGE_RESULT_KEY, AIProvider, AIResponse, ToolCall, ToolHandler, ToolSpec,
)

API_RATE = 24000    # Realtime API pcm16 is 24 kHz mono


class OpenAIRealtimeProvider(AIProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, instructions: str,
                 max_tool_rounds: int = 4):
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        self.max_tool_rounds = max_tool_rounds

        self.ws = None
        self._tools: list[ToolSpec] = []
        self._handler: ToolHandler | None = None
        self._pump: asyncio.Task | None = None

        self._text = ""
        self._turn_done = asyncio.Event()
        self._response = AIResponse()
        # Optimistic: set when we ask for a response, not when the server
        # confirms, because two creates sent back to back would both look
        # inactive otherwise.
        self._active = False
        self._tool_rounds = 0

    # -- lifecycle -------------------------------------------------
    async def connect(self) -> None:
        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        self.ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=None,
        )
        await self._send({"type": "session.update", "session": self._session()})
        self._pump = asyncio.create_task(self._run_pump())
        AI.info(f"connected: openai {self.model} (text out)")

    async def disconnect(self) -> None:
        if self._pump:
            self._pump.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    def _session(self) -> dict:
        return {
            "type": "realtime",
            "model": self.model,
            "instructions": self.instructions,
            "tools": [
                {"type": "function", "name": t.name, "description": t.description,
                 "parameters": t.parameters}
                for t in self._tools
            ],
            "tool_choice": "auto",
            "output_modalities": ["text"],
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": API_RATE},
                                "turn_detection": None}},
        }

    def register_tools(self, tools: Sequence[ToolSpec], handler: ToolHandler) -> None:
        self._tools = list(tools)
        self._handler = handler

    async def _send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj))

    # -- input -----------------------------------------------------
    async def send_audio(self, pcm: bytes, rate: int) -> None:
        pcm24 = resample(pcm, rate, API_RATE)
        await self._send({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm24).decode()})
        await self._send({"type": "input_audio_buffer.commit"})

    async def send_frame(self, jpeg: bytes) -> None:
        b64 = base64.b64encode(jpeg).decode()
        await self._send({
            "type": "conversation.item.create",
            "item": {"role": "user", "type": "message",
                     "content": [{"type": "input_image",
                                  "image_url": f"data:image/jpeg;base64,{b64}"}]},
        })

    async def send_text(self, text: str, *, turn_complete: bool = True) -> None:
        # turn_complete is meaningless here: appending an item never triggers
        # generation on its own in this API — only an explicit response.create
        # does, and that happens separately in complete_turn(). So every call
        # is already "non-completing" until the caller asks for a response.
        await self._send({
            "type": "conversation.item.create",
            "item": {"role": "user", "type": "message",
                     "content": [{"type": "input_text", "text": text}]},
        })

    # -- turn ------------------------------------------------------
    async def _create_response(self) -> bool:
        """One response at a time, or the API rejects the second."""
        if self._active:
            return False
        self._active = True
        await self._send({"type": "response.create"})
        return True

    async def complete_turn(self, timeout: float = 90.0) -> AIResponse:
        self._text = ""
        self._response = AIResponse()
        self._tool_rounds = 0
        self._turn_done.clear()
        await self._create_response()
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout)
        except asyncio.TimeoutError:
            AI.warn("turn timed out")
            self._response.error = "timeout"
        self._response.text = self._response.text or self._text
        return self._response

    async def interrupt(self) -> None:
        if self._active:
            await self._send({"type": "response.cancel"})
            self._active = False
        self._response.interrupted = True
        self._turn_done.set()

    # -- event pump ------------------------------------------------
    async def _run_pump(self) -> None:
        try:
            async for raw in self.ws:
                await self._on_event(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            AI.error(f"socket closed: {e!r}")
            self._turn_done.set()

    async def _on_event(self, ev: dict) -> None:
        t = ev.get("type", "")

        if t.endswith("output_text.delta") and "delta" in ev:
            self._text += ev["delta"]

        elif t.endswith("output_text.done"):
            said = ev.get("text") or self._text
            self._text = ""
            if said:
                self._response.text = said

        elif t.endswith("input_audio_transcription.completed"):
            AI.info(f"you: {ev.get('transcript','')}")

        elif t == "response.created":
            self._active = True

        elif t == "response.function_call_arguments.done":
            try:
                args = json.loads(ev.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            await self._dispatch(ToolCall(ev.get("name", ""), args,
                                          ev.get("call_id", "")))

        elif t == "response.done":
            await self._on_response_done(ev)

        elif t == "error":
            code = (ev.get("error") or {}).get("code")
            AI.error(f"api error: {json.dumps(ev.get('error', ev))[:300]}")
            # A rejected create means nothing new is in flight, and the
            # response already running will end the turn on its own.
            if code != "conversation_already_has_active_response":
                self._response.error = str(code)
                self._turn_done.set()

    async def _dispatch(self, call: ToolCall) -> None:
        self._response.tool_calls.append(call)
        result: dict = {"ok": False, "error": "no handler"}
        if self._handler:
            try:
                result = await self._handler(call.name, call.arguments)
            except Exception as e:
                result = {"ok": False, "error": str(e)}
        # A function result is text only in this API, so an image coming back
        # from a tool has to follow as its own user message.
        image = result.pop(IMAGE_RESULT_KEY, None) if result else None
        await self._send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call.call_id,
                     "output": json.dumps(result)},
        })
        if image:
            await self.send_frame(image)

    async def _on_response_done(self, ev: dict) -> None:
        self._active = False
        status = ev.get("response", {}).get("status")
        outputs = ev.get("response", {}).get("output", [])
        only_tools = bool(outputs) and all(
            o.get("type") == "function_call" for o in outputs)

        if status == "failed":
            AI.error(f"response failed: {ev.get('response', {}).get('status_details')}")
            self._response.error = "failed"
            self._turn_done.set()
        # A response that was nothing but tool calls is the model waiting on
        # results, so it needs a follow-up to say anything. One that already
        # spoke does not — its tool results just sit in the conversation.
        elif only_tools and self._tool_rounds < self.max_tool_rounds:
            self._tool_rounds += 1
            await self._create_response()
        else:
            self._turn_done.set()
