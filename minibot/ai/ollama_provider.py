"""Local LLM provider, served by Ollama.

Everything that thinks runs on this machine: Whisper turns the utterance into
text (speech/stt.py), a local vision+tools model answers it, and the reply goes
to the SpeechProvider like any other provider's. No API key, no quota, and the
room's audio never leaves the laptop.

How it differs from the two cloud providers:

There is no live session. Ollama's /api/chat is stateless request/response, so
the conversation history lives here and is replayed on every call. It is kept
bounded — the local analogue of Gemini's sliding-window compression — and it
is trimmed only at user-turn boundaries, because cutting between an assistant
tool call and its tool result leaves the model answering a question it cannot
see.

Photos are dropped from history once their turn is over. A vision model
re-encodes every image in the context on every request; carrying an hour of
snapshots forward would make each reply slower than the last.

The input methods only queue. Nothing is sent until complete_turn(), which is
where the whole tool loop runs: call the model, run what it asked for, feed the
results back, repeat until it answers in words.

When on_text is set the reply is streamed, and each finished sentence is
handed over the moment it exists. The robot starts speaking the first sentence
while the model is still writing the second — on a two-sentence answer that is
most of the model's time taken off what the person waits through.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from typing import Any, Sequence

import requests

from ..obs.logger import AI
from ..speech.stt import SpeechToText
from .provider import (
    DIRECT_REPLY_KEY, IMAGE_RESULT_KEY, AIProvider, AIResponse, ProviderUnavailable, ToolCall,
    ToolHandler, ToolSpec,
)

# Some local models still emit their reasoning inline even with think=false.
# The robot must never read that out loud (see the same failure in the Gemini
# provider).
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)

# Where a streamed reply may be cut and handed to speech: the end of a
# sentence followed by more text, or a line break.
_BOUNDARY = re.compile(r"[.!?…][\"'”’)\]]*(?=\s)|\n")
# A first chunk shorter than this ("Hi.") is held for the next sentence: every
# chunk is its own TTS request, and two tiny ones sound choppier than one.
MIN_CHUNK_CHARS = 12


class OllamaProvider(AIProvider):
    name = "ollama"
    supports_direct_reply = True

    def __init__(self, host: str, model: str, instructions: str,
                 stt: SpeechToText | None = None, *, num_ctx: int = 8192,
                 keep_alive: str = "30m", max_history: int = 40,
                 max_tool_rounds: int = 6, request_timeout: float = 120.0):
        self.host = host.rstrip("/")
        self.model = model
        self.instructions = instructions
        self.stt = stt
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.max_history = max_history
        self.max_tool_rounds = max_tool_rounds
        self.request_timeout = request_timeout

        self._tools: list[ToolSpec] = []
        self._handler: ToolHandler | None = None
        self._http = requests.Session()
        self._thinks = False       # model has a thinking mode to switch off
        self._vision = True

        self._history: list[dict[str, Any]] = []
        self._pending_text: list[str] = []
        self._pending_images: list[str] = []
        # Context-only text (memory, the mute reminder) is not a turn by
        # itself; something the person actually said or typed is.
        self._pending_utterance = False

        self._response = AIResponse()
        self._inflight: asyncio.Task | None = None
        self._said = 0             # chars of this round's reply already streamed
        self._t_start = 0.0
        self._t_first = 0.0
        self._stt_ms = 0.0
        # What the person said in the last send_audio(), for callers that act
        # on exact words before the model does (the mute phrase).
        self.last_transcript = ""

    def register_tools(self, tools: Sequence[ToolSpec],
                       handler: ToolHandler) -> None:
        self._tools = list(tools)
        self._handler = handler

    # -- http ------------------------------------------------------
    async def _post(self, path: str, body: dict) -> dict:
        def call() -> dict:
            r = self._http.post(f"{self.host}{path}", json=body,
                                timeout=self.request_timeout)
            if r.status_code >= 400:
                try:
                    detail = r.json().get("error", r.text)
                except ValueError:
                    detail = r.text
                raise RuntimeError(f"ollama {path} {r.status_code}: {detail}")
            return r.json()
        return await asyncio.get_running_loop().run_in_executor(None, call)

    # -- lifecycle -------------------------------------------------
    async def connect(self) -> None:
        """Check the model is actually there and fit for the job, then load it.

        A fresh history on every connect, matching a fresh cloud session:
        autopilot reconnects per mention precisely so one stranger's text is
        never in context while the next person's reply is written.
        """
        self._history = []
        self._clear_pending()
        try:
            info = await self._post("/api/show", {"model": self.model})
        except requests.ConnectionError:
            raise ProviderUnavailable(
                f"ollama is not running at {self.host} — start the Ollama app "
                f"or run `ollama serve`") from None
        except RuntimeError as e:
            if "404" in str(e) or "not found" in str(e):
                raise ProviderUnavailable(f"model {self.model!r} is not pulled — run "
                                   f"`ollama pull {self.model}`") from None
            raise

        caps = set(info.get("capabilities") or [])
        self._thinks = "thinking" in caps
        self._vision = "vision" in caps
        if self._tools and "tools" not in caps:
            raise ProviderUnavailable(f"{self.model} does not support tool calling; "
                               "pick a model with the 'tools' capability")
        if self._tools and not self._vision:
            AI.warn(f"{self.model} has no vision — take_photo will return a "
                    "picture it cannot see")

        # Load the weights AND read the system prompt and tool list once now,
        # so the first real turn is not the one that pays for either. The
        # options must match the real requests exactly: a different num_ctx
        # makes Ollama reload the whole model on the next call (measured: a
        # 4.4 s first reply), and the same prefix is what lets it reuse the
        # prompt it has already processed.
        warm = self._body(offer_tools=True, messages=[])
        warm["options"] = {**warm["options"], "num_predict": 1}
        with AI.timed(f"llm loaded ({self.model})"):
            await self._post("/api/chat", warm)
        if self.stt is not None:
            await self.stt.load()
        AI.info(f"connected: ollama {self.model} at {self.host} "
                f"({len(self._tools)} tools, vision={'yes' if self._vision else 'no'})")

    async def disconnect(self) -> None:
        if self._inflight is not None:
            self._inflight.cancel()
        self._history = []
        self._clear_pending()

    # -- input -----------------------------------------------------
    async def send_audio(self, pcm: bytes, rate: int) -> None:
        if self.stt is None:
            raise RuntimeError("ollama provider has no speech-to-text configured")
        t0 = time.perf_counter()
        text = await self.stt.transcribe(pcm, rate)
        self._stt_ms = (time.perf_counter() - t0) * 1000
        self.last_transcript = text
        if not text:
            AI.info("heard nothing intelligible")
            return
        AI.info(f"you: {text}")
        self._pending_text.append(text)
        self._pending_utterance = True

    async def send_frame(self, jpeg: bytes) -> None:
        self._pending_images.append(base64.b64encode(jpeg).decode())

    async def send_text(self, text: str, *, turn_complete: bool = True) -> None:
        self._pending_text.append(text)
        if turn_complete:
            self._pending_utterance = True

    def _clear_pending(self) -> None:
        self._pending_text, self._pending_images = [], []
        self._pending_utterance = False

    # -- turn ------------------------------------------------------
    async def complete_turn(self, timeout: float = 90.0) -> AIResponse:
        self._response = AIResponse(streamed=self.on_text is not None)
        self._t_start, self._t_first = time.perf_counter(), 0.0
        if not self._pending_utterance:
            # Context with nothing said after it — a muted clip that turned out
            # to be a cough, say. Nothing to answer, and nothing worth keeping.
            self._clear_pending()
            return self._response

        msg: dict[str, Any] = {"role": "user",
                               "content": "\n\n".join(self._pending_text)}
        if self._pending_images:
            msg["images"] = self._pending_images
        self._clear_pending()
        turn_start = len(self._history)
        self._history.append(msg)

        self._inflight = asyncio.create_task(self._run_turn())
        try:
            await asyncio.wait_for(asyncio.shield(self._inflight), timeout)
        except asyncio.TimeoutError:
            AI.warn("turn timed out")
            self._inflight.cancel()
            self._response.error = "timeout"
        except asyncio.CancelledError:
            # interrupt() cancelled the work; the turn simply ends here.
            if not self._response.interrupted:
                raise
        except Exception as e:
            AI.error(f"ollama turn failed: {e}")
            self._response.error = str(e)
        finally:
            self._inflight = None

        if self._response.error or self._response.interrupted:
            # A half-finished tool exchange left in history would be replayed
            # on every later turn. Drop the whole turn instead.
            del self._history[turn_start:]
        self._settle_history()

        self._response.text = self._response.text.strip()
        self._log_timing()
        if not self._response.text and not self._response.error \
                and not self._response.interrupted:
            AI.info("no spoken reply this turn "
                    f"({len(self._response.tool_calls)} tool calls)")
        return self._response

    async def interrupt(self) -> None:
        self._response.interrupted = True
        if self._inflight is not None:
            self._inflight.cancel()

    async def _run_turn(self) -> None:
        for round_ in range(self.max_tool_rounds + 1):
            # Out of rounds: ask once more with no tools on offer, so the
            # person gets an answer rather than silence.
            offer_tools = round_ < self.max_tool_rounds
            if not offer_tools and self._tools:
                AI.warn(f"tool round limit ({self.max_tool_rounds}) reached")
            self._said = 0
            reply = (await self._chat(offer_tools)).get("message") or {}

            calls = reply.get("tool_calls") or []
            text = _THINK.sub("", reply.get("content") or "").strip()
            self._history.append({"role": "assistant", "content": text,
                                  **({"tool_calls": calls} if calls else {})})
            if text:
                self._response.text += (" " if self._response.text else "") + text
            await self._stream_out(reply.get("content") or "", final=True)
            if not calls or not offer_tools:
                return
            direct = await self._dispatch_all(calls)
            if direct:
                # The tools already said what to say. Asking the model to
                # rephrase it costs a full round — seconds, when another app
                # has just used the GPU — and only adds room to embellish.
                self._history.append({"role": "assistant", "content": direct})
                self._response.text += (" " if self._response.text else "") + direct
                self._said = 0
                await self._stream_out(direct, final=True)
                return

    async def _chat(self, offer_tools: bool) -> dict:
        body = self._body(offer_tools, self._history)
        if self.on_text is None:
            return await self._post("/api/chat", body)
        return await self._chat_streaming(body)

    def _body(self, offer_tools: bool, messages: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.instructions},
                         *messages],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx},
        }
        if offer_tools and self._tools:
            body["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in self._tools]
        if self._thinks:
            # Reasoning costs seconds a spoken reply cannot spare, and on a
            # desk robot the answer is usually one sentence anyway.
            body["think"] = False
        return body

    async def _chat_streaming(self, body: dict) -> dict:
        """The same /api/chat call with stream=true, reassembled into the
        non-streaming shape so the tool loop above does not care which ran.

        requests is blocking, so the HTTP read runs on a worker thread and
        hands chunks back to the event loop through a queue.
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        url = f"{self.host}/api/chat"

        def pump() -> None:
            def put(*item) -> None:
                loop.call_soon_threadsafe(q.put_nowait, item)
            try:
                with self._http.post(url, json={**body, "stream": True},
                                     stream=True,
                                     timeout=self.request_timeout) as r:
                    if r.status_code >= 400:
                        raise RuntimeError(
                            f"ollama /api/chat {r.status_code}: {r.text[:300]}")
                    for line in r.iter_lines():
                        if stop.is_set():
                            return
                        if line:
                            put("chunk", json.loads(line))
                put("end", None)
            except Exception as e:
                put("error", e)

        loop.run_in_executor(None, pump)
        content, calls = "", []
        try:
            while True:
                kind, val = await q.get()
                if kind == "error":
                    raise val
                if kind == "end":
                    break
                if val.get("error"):
                    raise RuntimeError(f"ollama: {val['error']}")
                m = val.get("message") or {}
                if m.get("content"):
                    content += m["content"]
                    await self._stream_out(content)
                calls += m.get("tool_calls") or []
                if val.get("done"):
                    break
        finally:
            stop.set()      # an abandoned turn stops reading the stream
        return {"message": {"content": content, "tool_calls": calls}}

    async def _stream_out(self, raw: str, final: bool = False) -> None:
        """Hand on_text every finished sentence of `raw` not yet handed over.
        `raw` is the whole reply of this round so far; final=True flushes the
        unfinished tail too."""
        if self.on_text is None:
            return
        clean = _THINK.sub("", raw)
        pending = clean[self._said:]
        if final:
            cut = len(pending)
        else:
            ends = [m.end() for m in _BOUNDARY.finditer(pending)]
            cut = next((e for e in reversed(ends)
                        if len(pending[:e].strip()) >= MIN_CHUNK_CHARS), 0)
        chunk = pending[:cut].strip()
        self._said += cut
        if chunk:
            if not self._t_first:
                self._t_first = time.perf_counter()
            await self.on_text(chunk)

    def _log_timing(self) -> None:
        """One line per turn saying where the wait went."""
        end = time.perf_counter()
        bits = []
        if self._stt_ms:
            bits.append(f"stt {self._stt_ms:.0f}ms")
        if self._t_first:
            bits.append(f"first words {(self._t_first - self._t_start) * 1000:.0f}ms")
        bits.append(f"llm {(end - self._t_start) * 1000:.0f}ms")
        if self._response.tool_calls:
            bits.append(f"{len(self._response.tool_calls)} tool calls")
        AI.info("timing: " + ", ".join(bits))
        self._stt_ms = 0.0

    async def _dispatch_all(self, calls: list[dict]) -> str:
        """Runs one round of tool calls. Returns the direct reply when every
        call in the round supplied one (DIRECT_REPLY_KEY), else ""."""
        images: list[str] = []
        replies: list[str] = []
        for c in calls:
            fn = c.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                # Some model templates hand arguments back as a JSON string.
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            call_id = c.get("id") or ""
            self._response.tool_calls.append(ToolCall(name, args, call_id))

            result: dict[str, Any] = {"ok": False, "error": "no handler"}
            if self._handler:
                try:
                    result = await self._handler(name, args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            # A tool message is text only, so a camera tool's picture follows
            # the results as a user message the model can look at.
            image = result.pop(IMAGE_RESULT_KEY, None) if result else None
            if image:
                images.append(base64.b64encode(image).decode())
                result = {**result, "note": "the photo follows in the next message"}
            reply = result.pop(DIRECT_REPLY_KEY, None) if result else None
            if reply:
                replies.append(str(reply).strip())

            self._history.append({"role": "tool", "tool_name": name,
                                  "content": json.dumps(result, default=str),
                                  **({"tool_call_id": call_id} if call_id else {})})
        if images:
            self._history.append({"role": "user", "images": images,
                                  "content": "This is the photo you just took."})
        if images or len(replies) != len(calls):
            return ""
        return " ".join(r for r in replies if r)

    # -- history ---------------------------------------------------
    def _settle_history(self) -> None:
        """Drop old photos and old turns once the current turn is over."""
        self._history = [_without_images(m) for m in self._history]
        if len(self._history) <= self.max_history:
            return
        # Cut at the oldest user message that leaves the history within the
        # limit — never mid tool-exchange.
        cut = len(self._history) - self.max_history
        while cut < len(self._history) and self._history[cut]["role"] != "user":
            cut += 1
        del self._history[:cut]


def _without_images(m: dict[str, Any]) -> dict[str, Any]:
    if "images" not in m:
        return m
    rest = {k: v for k, v in m.items() if k != "images"}
    rest["content"] = (rest.get("content", "") +
                       "\n[a photo was shown here; it is no longer in view]").strip()
    return rest
