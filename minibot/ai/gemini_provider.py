"""Gemini Robotics ER 2 streaming provider (§1).

Wraps `gemini-robotics-er-2-streaming-preview` on the Live API. Output is TEXT;
speech is synthesized separately by the SpeechProvider, so the robot keeps its
cloned voice.

Three design points worth knowing:

Automatic activity detection is DISABLED. The firmware already endpoints each
utterance — that is what /mic?silence= does — so what arrives here is a
complete, pre-trimmed turn. Letting the server's own VAD re-segment it would
fight work the robot already did. Instead each burst is bracketed explicitly
with activity_start / activity_end, which makes turn boundaries deterministic.

Tools are declared BLOCKING. The model waits for the robot to actually finish
the movement before continuing, which is what makes "never claim a physical
action succeeded until the result confirms it" enforceable rather than a
request in the prompt.

Frames are hard-throttled to the documented 1 FPS ceiling here, independently
of whatever the camera policy upstream decides. A policy bug must not be able
to violate the API contract.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Sequence

from google import genai
from google.genai import types

from ..audio.dsp import resample
from ..obs.logger import AI
from .provider import (
    IMAGE_RESULT_KEY, AIProvider, AIResponse, ToolCall, ToolHandler, ToolSpec,
)

# The Live API takes raw 16-bit PCM mono at 16 kHz — the same rate the firmware
# records and plays at, so audio crosses this boundary without resampling.
GEMINI_AUDIO_RATE = 16000
MAX_FRAME_FPS = 1.0


def _to_declaration(spec: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=spec.name,
        description=spec.description,
        parameters=spec.parameters,
        behavior=(types.Behavior.BLOCKING if spec.blocking
                  else types.Behavior.NON_BLOCKING),
    )


class GeminiRoboticsProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, instructions: str,
                 max_tool_rounds: int = 6, followup_timeout: float = 20.0):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.instructions = instructions
        self.max_tool_rounds = max_tool_rounds
        # How long to wait for the model's spoken reply after it has been given
        # tool results. Bounded so a purely physical command ("look left"),
        # which may never produce text at all, cannot hang the whole turn.
        self.followup_timeout = followup_timeout
        self._awaiting_followup = False
        self._followup_task: asyncio.Task | None = None

        self._cm = None            # the connect() context manager
        self._session = None
        self._pump: asyncio.Task | None = None
        self._tools: list[ToolSpec] = []
        self._handler: ToolHandler | None = None

        self._response = AIResponse()
        self._turn_done = asyncio.Event()
        self._tool_rounds = 0
        self._last_frame = 0.0
        self._connected = False

    # -- config ----------------------------------------------------
    def _config(self) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=types.Content(
                parts=[types.Part(text=self.instructions)]),
            tools=[types.Tool(function_declarations=[
                _to_declaration(t) for t in self._tools])] if self._tools else None,
            realtime_input_config=types.RealtimeInputConfig(
                # See module docstring: the robot endpoints its own turns.
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True),
            ),
            # A desk robot runs for hours. Without this the session eventually
            # walks into the context limit mid-conversation (§5: do not keep
            # replaying the whole transcript).
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            # This is a reasoning model and its thoughts land in
            # output_transcription alongside the answer. Without this the robot
            # reads its own chain of thought out loud — observed live: it spoke
            # a JSON fragment, three drafts of a greeting, and the text of a
            # safety policy before finally saying hello. Keep the thinking,
            # just keep it out of the output stream.
            thinking_config=types.ThinkingConfig(include_thoughts=False),
        )

    def register_tools(self, tools: Sequence[ToolSpec],
                       handler: ToolHandler) -> None:
        self._tools = list(tools)
        self._handler = handler

    # -- lifecycle -------------------------------------------------
    async def connect(self) -> None:
        # connect() is an async context manager; drive it by hand so the
        # session can outlive a single `async with` block.
        self._cm = self._client.aio.live.connect(model=self.model,
                                                 config=self._config())
        self._session = await self._cm.__aenter__()
        self._connected = True
        self._pump = asyncio.create_task(self._run_pump())
        AI.info(f"connected: gemini {self.model} (text out, "
                f"{len(self._tools)} tools, blocking)")

    async def disconnect(self) -> None:
        self._connected = False
        if self._pump:
            self._pump.cancel()
            try:
                await self._pump
            except (asyncio.CancelledError, Exception):
                pass
        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception as e:
                AI.debug(f"session close: {e!r}")
        self._session = None
        self._cm = None

    def _require(self):
        if self._session is None:
            raise RuntimeError("gemini provider is not connected")
        return self._session

    # -- input -----------------------------------------------------
    async def send_audio(self, pcm: bytes, rate: int) -> None:
        """One complete, already-endpointed utterance."""
        s = self._require()
        if rate != GEMINI_AUDIO_RATE:
            pcm = resample(pcm, rate, GEMINI_AUDIO_RATE)
        await s.send_realtime_input(activity_start=types.ActivityStart())
        await s.send_realtime_input(
            audio=types.Blob(data=pcm,
                             mime_type=f"audio/pcm;rate={GEMINI_AUDIO_RATE}"))
        await s.send_realtime_input(activity_end=types.ActivityEnd())
        AI.debug(f"sent {len(pcm) / 2 / GEMINI_AUDIO_RATE:.2f}s of audio")

    async def send_frame(self, jpeg: bytes) -> None:
        """Camera frame as visual context. Hard-capped at the documented 1 FPS
        regardless of what the caller asks for."""
        s = self._require()
        gap = time.monotonic() - self._last_frame
        floor = 1.0 / MAX_FRAME_FPS
        if gap < floor:
            await asyncio.sleep(floor - gap)
        self._last_frame = time.monotonic()
        await s.send_realtime_input(
            video=types.Blob(data=jpeg, mime_type="image/jpeg"))
        AI.debug(f"sent frame {len(jpeg)}B")

    async def send_text(self, text: str, *, turn_complete: bool = True) -> None:
        s = self._require()
        await s.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=turn_complete,
        )

    # -- turn ------------------------------------------------------
    async def complete_turn(self, timeout: float = 90.0) -> AIResponse:
        """Input for this turn is already sent; wait for the model to finish.

        Unlike the OpenAI provider there is no explicit "create a response"
        step — activity_end (or turn_complete on a text turn) is what starts
        generation.
        """
        self._response = AIResponse()
        self._tool_rounds = 0
        self._awaiting_followup = False
        self._cancel_followup()
        self._turn_done.clear()
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout)
        except asyncio.TimeoutError:
            AI.warn("turn timed out")
            self._response.error = "timeout"
        self._cancel_followup()
        self._response.text = self._response.text.strip()
        if not self._response.text and not self._response.error \
                and not self._response.interrupted:
            # Not an error: a purely physical instruction legitimately produces
            # no speech. But the server also sometimes drops a response with no
            # text and no turn_complete at all, and the two are indistinguishable
            # from here — so note it visibly instead of returning a silent blank.
            AI.info("no spoken reply this turn "
                    f"({len(self._response.tool_calls)} tool calls)")
        return self._response

    async def interrupt(self) -> None:
        """Barge-in. A client turn arriving mid-generation interrupts the model,
        and the server confirms it with server_content.interrupted."""
        if self._session is None:
            return
        try:
            await self._session.send_realtime_input(activity_start=types.ActivityStart())
            await self._session.send_realtime_input(activity_end=types.ActivityEnd())
        except Exception as e:
            AI.debug(f"interrupt failed: {e!r}")
        self._response.interrupted = True
        self._turn_done.set()

    # -- receive pump ----------------------------------------------
    async def _run_pump(self) -> None:
        """receive() yields for one turn and then returns, so it is re-entered
        for the life of the session."""
        try:
            while self._connected and self._session is not None:
                async for msg in self._session.receive():
                    await self._on_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            AI.error(f"gemini stream ended: {e!r}")
            self._response.error = str(e)
            self._turn_done.set()

    @staticmethod
    def _trace(msg: types.LiveServerMessage) -> str:
        """One-line shape of an incoming message, for diagnosing stalls.

        Worth keeping: the reply living in output_transcription rather than
        model_turn was only findable by looking at raw message shapes.
        """
        bits = [f for f in ("setup_complete", "tool_call", "tool_call_cancellation",
                            "go_away", "usage_metadata", "session_resumption_update")
                if getattr(msg, f, None) is not None]
        sc = msg.server_content
        if sc is not None:
            inner = [f for f in ("turn_complete", "interrupted",
                                 "generation_complete", "waiting_for_input")
                     if getattr(sc, f, None)]
            if sc.output_transcription:
                inner.append("output_transcription")
            if sc.input_transcription:
                inner.append("input_transcription")
            if sc.model_turn:
                inner.append("model_turn")
            bits.append(f"server_content[{','.join(inner) or 'empty'}]")
        return " ".join(bits) or "(empty)"

    async def _on_message(self, msg: types.LiveServerMessage) -> None:
        AI.debug(f"<- {self._trace(msg)}")

        if msg.setup_complete is not None:
            AI.debug("session setup complete")

        if msg.tool_call is not None:
            await self._dispatch_all(msg.tool_call.function_calls or [])

        if msg.tool_call_cancellation is not None:
            AI.info(f"tool calls cancelled: {msg.tool_call_cancellation.ids}")

        if msg.go_away is not None:
            AI.warn(f"server going away in {msg.go_away.time_left}")

        sc = msg.server_content

        # Where the reply actually lives.
        #
        # gemini-robotics-er-2-streaming-preview returns its answer in
        # server_content.output_transcription, NOT as text parts on
        # model_turn — so LiveServerMessage.text is always None and reading
        # only that silently discards every response. Verified against the
        # live API. msg.text is still consulted so a future model (or a
        # different one behind this provider) that uses model_turn works too.
        text = msg.text
        if not text and sc is not None and sc.output_transcription:
            text = sc.output_transcription.text

        if text:
            # Generation has started, so the model is not going to stay silent
            # after its tool calls; let it finish rather than timing it out.
            self._cancel_followup()
            self._response.text += text

        if sc is not None:
            # input_transcription is what the USER said; never let it reach
            # _response.text or the robot repeats the user back to them.
            if sc.input_transcription and sc.input_transcription.text:
                AI.info(f"you: {sc.input_transcription.text}")
            if sc.interrupted:
                AI.info("model interrupted")
                self._response.interrupted = True
                self._finish_turn()
            if sc.turn_complete:
                self._on_turn_complete()

    def _on_turn_complete(self) -> None:
        """A tool-call batch ends its own turn before the model has said
        anything. Ending here would return an empty reply and drop the answer
        the user is waiting for, so when tool results have just been submitted
        and no text has arrived yet, keep listening for the follow-up turn.

        Bounded by followup_timeout: a purely physical instruction may
        legitimately produce no speech at all.
        """
        if self._awaiting_followup and not self._response.text:
            self._awaiting_followup = False
            AI.debug("turn complete after tool calls — awaiting the reply")
            self._arm_followup()
            return
        self._finish_turn()

    def _arm_followup(self) -> None:
        self._cancel_followup()

        async def _expire() -> None:
            await asyncio.sleep(self.followup_timeout)
            if not self._response.text:
                AI.debug("no reply after tool results; ending the turn")
            self._turn_done.set()

        try:
            self._followup_task = asyncio.create_task(_expire())
        except RuntimeError:      # no running loop (tests calling synchronously)
            self._turn_done.set()

    def _cancel_followup(self) -> None:
        if self._followup_task is not None:
            self._followup_task.cancel()
            self._followup_task = None

    def _finish_turn(self) -> None:
        self._cancel_followup()
        self._turn_done.set()

    async def _dispatch_all(self, calls: list[types.FunctionCall]) -> None:
        """Every call in one batch is answered in a single tool response, which
        is what the BLOCKING contract expects."""
        if self._tool_rounds >= self.max_tool_rounds:
            AI.warn(f"tool round limit ({self.max_tool_rounds}) reached")
            return
        self._tool_rounds += 1

        responses: list[types.FunctionResponse] = []
        images: list[tuple[str, bytes]] = []
        for call in calls:
            args: dict[str, Any] = dict(call.args or {})
            self._response.tool_calls.append(
                ToolCall(call.name or "", args, call.id or ""))
            result: dict[str, Any] = {"ok": False, "error": "no handler"}
            if self._handler:
                try:
                    result = await self._handler(call.name or "", args)
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            # A camera tool answers with a picture, but it cannot travel on the
            # FunctionResponse: send_tool_response serializes with
            # convert_to_dict and never base64-encodes bytes, so any binary
            # part raises "Object of type bytes is not JSON serializable"
            # (google-genai 2.21). send_client_content does model_dump(
            # mode="json") first and handles it correctly, so the frame follows
            # the tool result as its own user turn instead. Verified against
            # the SDK; revisit if upstream fixes the tool-response path.
            image = result.pop(IMAGE_RESULT_KEY, None) if result else None
            if image:
                images.append((call.name or "camera", image))
                result = {**result, "note": "image follows in the next message"}

            responses.append(types.FunctionResponse(
                id=call.id, name=call.name, response=result))

        if responses and self._session is not None:
            # Images go in BEFORE the tool response, and without completing the
            # turn. A completed user turn counts as a new turn and interrupts
            # whatever the model is generating; sending it while the model is
            # still blocked on the tool result, with turn_complete=False, just
            # appends context. The tool response is then what resumes it, with
            # the picture already in view.
            for label, image in images:
                await self._send_image(label, image)
            await self._session.send_tool_response(function_responses=responses)
            # The model now owes us a reply; see _on_turn_complete.
            self._awaiting_followup = True
            # Arm the watchdog here, not only on turn_complete. When the server
            # drops a response on policy grounds it sends nothing further at
            # all — no text, no turn_complete — and waiting on the outer 90 s
            # timeout leaves the robot looking frozen mid-conversation.
            self._arm_followup()

    async def _send_image(self, label: str, jpeg: bytes) -> None:
        """Hand a captured frame to the model as realtime video.

        This is the only channel that works for a tool-returned image. All
        three were tried against the live API:
          - FunctionResponse.parts raises "Object of type bytes is not JSON
            serializable" — send_tool_response skips the mode="json" dump that
            base64-encodes bytes (google-genai 2.21).
          - send_client_content serializes correctly but counts as a user turn
            and interrupts the model, with turn_complete either True or False.
          - realtime video neither serializes wrong nor interrupts. Verified
            working: the robot describes what is actually in frame.

        Revisit if the SDK fixes the tool-response path, which is the
        semantically correct home for a camera tool's result.
        """
        if self._session is None:
            return
        await self.send_frame(jpeg)
        AI.debug(f"sent {len(jpeg)}B image from {label}")
