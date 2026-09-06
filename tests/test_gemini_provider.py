"""Gemini Robotics ER 2 provider tests (§26).

Drives the real provider logic against a fake Live session, so no API key or
network is involved. Messages are built from genuine SDK types, which means
these break if google-genai changes shape rather than silently passing.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.ai.gemini_provider import (  # noqa: E402
    GEMINI_AUDIO_RATE, MAX_FRAME_FPS, GeminiRoboticsProvider,
)
from minibot.ai.provider import ToolSpec  # noqa: E402

TOOLS = [
    ToolSpec("set_face", "Change the expression on the OLED face.",
             {"type": "object", "properties": {"emotion": {"type": "string"}},
              "required": ["emotion"]}),
    ToolSpec("look_at", "Turn the head.",
             {"type": "object", "properties": {"pan": {"type": "integer"}}}),
]


class FakeSession:
    def __init__(self):
        self.realtime: list[dict] = []
        self.client_content: list[dict] = []
        self.tool_responses: list[list[types.FunctionResponse]] = []
        self.closed = False

    async def send_realtime_input(self, **kw):
        self.realtime.append(kw)

    async def send_client_content(self, **kw):
        self.client_content.append(kw)

    async def send_tool_response(self, *, function_responses):
        self.tool_responses.append(list(function_responses))

    def receive(self):
        async def _gen():
            if False:
                yield
        return _gen()


def make_provider(handler=None, tools=TOOLS):
    p = GeminiRoboticsProvider("fake-key", "gemini-robotics-er-2-streaming-preview",
                               "you are a small desk robot")
    p.register_tools(tools, handler)
    p._session = FakeSession()
    p._connected = True
    return p


def text_msg(t: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(
        model_turn=types.Content(role="model", parts=[types.Part(text=t)])))


def tool_msg(*calls) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(function_calls=list(calls)))


def _tone(ms: int, rate: int) -> bytes:
    n = int(rate * ms / 1000)
    return (np.sin(2 * np.pi * 220 * np.arange(n) / rate) * 6000).astype("<i2").tobytes()


class TestConfig:
    def test_tools_are_blocking(self):
        """BLOCKING is what makes the model wait for the robot to finish, so
        it cannot claim an action succeeded before it did (§27)."""
        cfg = make_provider()._config()
        decls = cfg.tools[0].function_declarations
        assert {d.name for d in decls} == {"set_face", "look_at"}
        assert all(d.behavior == types.Behavior.BLOCKING for d in decls)

    def test_non_blocking_is_honoured(self):
        p = make_provider(tools=[ToolSpec("idle", "d", {"type": "object"},
                                          blocking=False)])
        decl = p._config().tools[0].function_declarations[0]
        assert decl.behavior == types.Behavior.NON_BLOCKING

    def test_text_output_only(self):
        """Speech is ElevenLabs' job; the model must not synthesize audio."""
        cfg = make_provider()._config()
        assert [str(m) for m in cfg.response_modalities] == ["Modality.TEXT"]

    def test_automatic_activity_detection_disabled(self):
        """The firmware already endpoints each utterance; server-side VAD would
        re-segment a turn the robot has already trimmed."""
        cfg = make_provider()._config()
        assert cfg.realtime_input_config.automatic_activity_detection.disabled is True

    def test_context_compression_enabled(self):
        assert make_provider()._config().context_window_compression is not None

    def test_no_tools_yields_no_tool_config(self):
        p = GeminiRoboticsProvider("k", "m", "i")
        assert p._config().tools is None


class TestAudioInput:
    def test_burst_is_bracketed_by_activity_markers(self):
        p = make_provider()
        asyncio.run(p.send_audio(_tone(200, GEMINI_AUDIO_RATE), GEMINI_AUDIO_RATE))
        kinds = [next(iter(k)) for k in p._session.realtime]
        assert kinds == ["activity_start", "audio", "activity_end"]

    def test_16k_audio_is_not_resampled(self):
        """The firmware records at exactly the rate the Live API wants."""
        pcm = _tone(200, GEMINI_AUDIO_RATE)
        p = make_provider()
        asyncio.run(p.send_audio(pcm, GEMINI_AUDIO_RATE))
        assert p._session.realtime[1]["audio"].data == pcm

    def test_other_rates_are_resampled(self):
        pcm = _tone(200, 24000)
        p = make_provider()
        asyncio.run(p.send_audio(pcm, 24000))
        sent = p._session.realtime[1]["audio"].data
        assert len(sent) == pytest.approx(len(pcm) * 16000 / 24000, rel=0.01)

    def test_mime_type_declares_the_rate(self):
        p = make_provider()
        asyncio.run(p.send_audio(_tone(50, GEMINI_AUDIO_RATE), GEMINI_AUDIO_RATE))
        assert p._session.realtime[1]["audio"].mime_type == "audio/pcm;rate=16000"

    def test_send_before_connect_raises(self):
        p = GeminiRoboticsProvider("k", "m", "i")
        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(p.send_audio(b"\x00\x00", 16000))


class TestFrames:
    def test_frame_is_jpeg(self):
        p = make_provider()
        asyncio.run(p.send_frame(b"\xff\xd8data\xff\xd9"))
        assert p._session.realtime[0]["video"].mime_type == "image/jpeg"

    def test_throttled_to_one_fps(self):
        """A camera-policy bug upstream must not be able to breach the
        documented 1 FPS ceiling."""
        p = make_provider()

        async def burst():
            for _ in range(3):
                await p.send_frame(b"\xff\xd8x\xff\xd9")

        t0 = time.monotonic()
        asyncio.run(burst())
        elapsed = time.monotonic() - t0
        assert elapsed >= 2 * (1.0 / MAX_FRAME_FPS) - 0.05
        assert len(p._session.realtime) == 3


class TestToolCalls:
    def test_handler_invoked_and_answered(self):
        seen = []

        async def handler(name, args):
            seen.append((name, args))
            return {"ok": True, "emotion": args["emotion"]}

        p = make_provider(handler)
        msg = tool_msg(types.FunctionCall(id="c1", name="set_face",
                                          args={"emotion": "happy"}))
        asyncio.run(p._on_message(msg))

        assert seen == [("set_face", {"emotion": "happy"})]
        resp = p._session.tool_responses[0][0]
        assert (resp.id, resp.name) == ("c1", "set_face")
        assert resp.response == {"ok": True, "emotion": "happy"}

    def test_batch_answered_in_one_response(self):
        """One tool response per batch is what the BLOCKING contract expects."""
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        asyncio.run(p._on_message(tool_msg(
            types.FunctionCall(id="a", name="set_face", args={"emotion": "happy"}),
            types.FunctionCall(id="b", name="look_at", args={"pan": 100}),
        )))
        assert len(p._session.tool_responses) == 1
        assert [r.id for r in p._session.tool_responses[0]] == ["a", "b"]

    def test_handler_exception_becomes_structured_error(self):
        """A raising tool must not kill the session."""
        async def handler(name, args):
            raise RuntimeError("servo jammed")

        p = make_provider(handler)
        asyncio.run(p._on_message(tool_msg(
            types.FunctionCall(id="c1", name="look_at", args={"pan": 90}))))
        resp = p._session.tool_responses[0][0]
        assert resp.response["ok"] is False
        assert "servo jammed" in resp.response["error"]

    def test_missing_handler_reports_error(self):
        p = make_provider(None)
        asyncio.run(p._on_message(tool_msg(
            types.FunctionCall(id="c1", name="look_at", args={}))))
        assert p._session.tool_responses[0][0].response["ok"] is False

    def test_calls_are_recorded_on_the_response(self):
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        asyncio.run(p._on_message(tool_msg(
            types.FunctionCall(id="c1", name="set_face", args={"emotion": "sad"}))))
        assert [c.name for c in p._response.tool_calls] == ["set_face"]

    def test_round_limit_stops_runaway_loops(self):
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        p.max_tool_rounds = 2

        async def go():
            for _ in range(5):
                await p._on_message(tool_msg(
                    types.FunctionCall(id="x", name="look_at", args={})))

        asyncio.run(go())
        assert len(p._session.tool_responses) == 2

    def test_null_args_do_not_crash(self):
        async def handler(name, args):
            return {"ok": True, "args": args}

        p = make_provider(handler)
        asyncio.run(p._on_message(tool_msg(
            types.FunctionCall(id="c1", name="look_at", args=None))))
        assert p._session.tool_responses[0][0].response["args"] == {}


class TestTurnLifecycle:
    def _run_turn(self, msgs, timeout=2.0):
        p = make_provider()

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=timeout))
            await asyncio.sleep(0)
            for m in msgs:
                await p._on_message(m)
            return await task

        return p, asyncio.run(go())

    def test_text_accumulates_until_turn_complete(self):
        _, resp = self._run_turn([
            text_msg("Hey, "), text_msg("that is a soldering iron."),
            types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)),
        ])
        assert resp.text == "Hey, that is a soldering iron."
        assert resp.error is None

    def test_interrupted_marks_the_response(self):
        _, resp = self._run_turn([
            text_msg("I was saying"),
            types.LiveServerMessage(
                server_content=types.LiveServerContent(interrupted=True)),
        ])
        assert resp.interrupted is True

    def test_timeout_is_reported_not_raised(self):
        p = make_provider()
        resp = asyncio.run(p.complete_turn(timeout=0.05))
        assert resp.error == "timeout"

    def test_turn_state_resets_between_turns(self):
        p = make_provider()

        async def go():
            async def one(word):
                task = asyncio.create_task(p.complete_turn(timeout=2))
                await asyncio.sleep(0)
                await p._on_message(text_msg(word))
                await p._on_message(types.LiveServerMessage(
                    server_content=types.LiveServerContent(turn_complete=True)))
                return await task
            return await one("first"), await one("second")

        a, b = asyncio.run(go())
        assert (a.text, b.text) == ("first", "second")

    def test_text_turn_uses_client_content(self):
        p = make_provider()
        asyncio.run(p.send_text("hello robot"))
        sent = p._session.client_content[0]
        assert sent["turn_complete"] is True
        assert sent["turns"].parts[0].text == "hello robot"


class TestRobustness:
    def test_go_away_is_logged_not_fatal(self):
        p = make_provider()
        msg = types.LiveServerMessage(go_away=types.LiveServerGoAway())
        asyncio.run(p._on_message(msg))     # must not raise

    def test_setup_complete_is_handled(self):
        p = make_provider()
        asyncio.run(p._on_message(
            types.LiveServerMessage(setup_complete=types.LiveServerSetupComplete())))

    def test_input_transcription_does_not_pollute_reply(self):
        """The user's own words must not end up in the robot's spoken answer."""
        p = make_provider()
        asyncio.run(p._on_message(types.LiveServerMessage(
            server_content=types.LiveServerContent(
                input_transcription=types.Transcription(text="where are my keys")))))
        assert p._response.text == ""

    def test_disconnect_without_connect_is_safe(self):
        asyncio.run(GeminiRoboticsProvider("k", "m", "i").disconnect())

    def test_interrupt_without_session_is_safe(self):
        asyncio.run(GeminiRoboticsProvider("k", "m", "i").interrupt())


class TestProviderSelection:
    """§24: Gemini and OpenAI must be switchable by configuration alone."""

    def _settings(self, monkeypatch, **env):
        from minibot.config import Settings
        for k in ("AI_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY",
                  "GOOGLE_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return Settings.load()

    def test_gemini_selected(self, monkeypatch):
        from minibot.cli import build_ai_provider
        s = self._settings(monkeypatch, AI_PROVIDER="gemini", GEMINI_API_KEY="k")
        p = build_ai_provider(s)
        assert p.name == "gemini"
        assert p.model == "gemini-robotics-er-2-streaming-preview"

    def test_openai_selected(self, monkeypatch):
        from minibot.cli import build_ai_provider
        s = self._settings(monkeypatch, AI_PROVIDER="openai", OPENAI_API_KEY="k")
        assert build_ai_provider(s).name == "openai"

    def test_missing_key_exits_with_a_clear_message(self, monkeypatch):
        from minibot.cli import build_ai_provider
        s = self._settings(monkeypatch, AI_PROVIDER="gemini")
        with pytest.raises(SystemExit) as e:
            build_ai_provider(s)
        assert "GEMINI_API_KEY" in str(e.value)

    def test_unknown_provider_exits(self, monkeypatch):
        from minibot.cli import build_ai_provider
        s = self._settings(monkeypatch, AI_PROVIDER="llama")
        with pytest.raises(SystemExit) as e:
            build_ai_provider(s)
        assert "llama" in str(e.value)

    def test_both_providers_share_one_persona(self, monkeypatch):
        """One robot, one personality, whichever brain is driving."""
        from minibot.agent.prompts import INSTRUCTIONS
        from minibot.cli import build_ai_provider
        g = build_ai_provider(self._settings(monkeypatch, AI_PROVIDER="gemini",
                                             GEMINI_API_KEY="k"))
        o = build_ai_provider(self._settings(monkeypatch, AI_PROVIDER="openai",
                                             OPENAI_API_KEY="k"))
        assert g.instructions == o.instructions == INSTRUCTIONS


class TestPersona:
    """§27 rules that are easy to lose in a later prompt edit.

    Whitespace is normalized first: the prompt is hard-wrapped for readability,
    so a rule can legitimately straddle a line break.
    """

    @staticmethod
    def _prompt() -> str:
        from minibot.agent.prompts import INSTRUCTIONS
        return " ".join(INSTRUCTIONS.lower().split())

    def test_forbids_narrating_tool_calls(self):
        p = self._prompt()
        assert "never narrate your own machinery" in p
        assert "i am moving my head now" in p

    def test_forbids_claiming_unconfirmed_actions(self):
        assert "never claim a physical action succeeded" in self._prompt()

    def test_forbids_inventing_vision(self):
        p = self._prompt()
        assert "blind until you take a photo" in p
        assert "never invent a sensor reading" in p

    def test_hides_memory_implementation(self):
        p = self._prompt()
        assert "never mention databases, embeddings" in p

    def test_requires_yielding_to_interruption(self):
        assert "if the user starts talking, stop" in self._prompt()

    def test_discourages_constant_motion(self):
        assert "do not move constantly" in self._prompt()


class TestToolThenTextTurn:
    """Regression for the live run of 2026-09-02.

    Gemini ends the tool-call turn with turn_complete BEFORE generating any
    text. Treating that as the end of the turn returned an empty reply and the
    robot said nothing after doing four tool calls.
    """

    def _drive(self, msgs, followup_timeout=20.0):
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        p.followup_timeout = followup_timeout

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=5))
            await asyncio.sleep(0)
            for m in msgs:
                await p._on_message(m)
                await asyncio.sleep(0)
            return await task

        return p, asyncio.run(go())

    def test_reply_after_tool_calls_is_not_lost(self):
        turn_end = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True))
        _, resp = self._drive([
            tool_msg(types.FunctionCall(id="a", name="set_face",
                                        args={"emotion": "happy"}),
                     types.FunctionCall(id="b", name="take_photo", args={})),
            turn_end,                       # tool turn ends, still no text
            text_msg("Hey! Good to see you."),
            turn_end,                       # the real end
        ])
        assert resp.text == "Hey! Good to see you."

    def test_several_tool_rounds_then_text(self):
        """The live run made two batches before replying."""
        turn_end = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True))
        _, resp = self._drive([
            tool_msg(types.FunctionCall(id="a", name="set_face",
                                        args={"emotion": "happy"})),
            turn_end,
            tool_msg(types.FunctionCall(id="b", name="look_at", args={"pan": 90})),
            turn_end,
            text_msg("That's a soldering iron on your desk."),
            turn_end,
        ])
        assert resp.text == "That's a soldering iron on your desk."

    def test_silent_action_turn_ends_promptly(self):
        """A purely physical command may never produce speech; the turn must
        still end quickly instead of hanging until the 90 s timeout."""
        turn_end = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True))
        t0 = time.monotonic()
        _, resp = self._drive([
            tool_msg(types.FunctionCall(id="a", name="look_at", args={"pan": 40})),
            turn_end,
        ], followup_timeout=0.2)
        assert resp.text == ""
        assert resp.error is None
        assert time.monotonic() - t0 < 2.0

    def test_plain_text_turn_still_ends_immediately(self):
        """No tool calls means no follow-up wait."""
        t0 = time.monotonic()
        _, resp = self._drive([
            text_msg("Morning."),
            types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)),
        ], followup_timeout=30.0)
        assert resp.text == "Morning."
        assert time.monotonic() - t0 < 1.0


def transcript_msg(t: str) -> types.LiveServerMessage:
    """How gemini-robotics-er-2-streaming-preview actually returns its reply."""
    return types.LiveServerMessage(server_content=types.LiveServerContent(
        output_transcription=types.Transcription(text=t)))


class TestOutputTranscription:
    """Regression for the live session of 2026-09-02.

    This model returns its answer in server_content.output_transcription, not
    as text parts on model_turn, so LiveServerMessage.text is always None.
    Reading only msg.text silently discarded every reply the robot ever made —
    tool calls fired, the head moved, and then it said nothing.

    Confirmed by dumping raw messages from a live session.
    """

    def test_reply_is_read_from_output_transcription(self):
        p = make_provider()

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=2))
            await asyncio.sleep(0)
            await p._on_message(transcript_msg("Hello! I am Mini Bot."))
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)))
            return await task

        assert asyncio.run(go()).text == "Hello! I am Mini Bot."

    def test_transcription_chunks_accumulate(self):
        p = make_provider()

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=2))
            await asyncio.sleep(0)
            for chunk in ("I see ", "a circuit board ", "and a screwdriver."):
                await p._on_message(transcript_msg(chunk))
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)))
            return await task

        assert asyncio.run(go()).text == "I see a circuit board and a screwdriver."

    def test_model_turn_text_still_works(self):
        """Kept as a fallback so a model that uses model_turn is not broken."""
        p = make_provider()

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=2))
            await asyncio.sleep(0)
            await p._on_message(text_msg("via model_turn"))
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)))
            return await task

        assert asyncio.run(go()).text == "via model_turn"

    def test_user_speech_never_becomes_the_reply(self):
        """input_transcription is the USER talking. Mixing it into the reply
        would make the robot read the user's own words back to them."""
        p = make_provider()

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=2))
            await asyncio.sleep(0)
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    input_transcription=types.Transcription(
                        text="what is on my desk"))))
            await p._on_message(transcript_msg("A screwdriver."))
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)))
            return await task

        assert asyncio.run(go()).text == "A screwdriver."

    def test_reply_after_tools_via_transcription(self):
        """The full live shape: tool calls, tool turn ends, then the answer."""
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        turn_end = types.LiveServerMessage(
            server_content=types.LiveServerContent(turn_complete=True))

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=3))
            await asyncio.sleep(0)
            for m in (tool_msg(types.FunctionCall(id="a", name="take_photo",
                                                  args={})),
                      turn_end,
                      transcript_msg("A circuit board and a screwdriver."),
                      turn_end):
                await p._on_message(m)
                await asyncio.sleep(0)
            return await task

        assert asyncio.run(go()).text == "A circuit board and a screwdriver."


class TestUnansweredTurn:
    """The server sometimes stops sending after a tool response — no text, no
    turn_complete, nothing. Observed live on 2026-09-02: the robot sat frozen
    for the full 90 s outer timeout. The watchdog is armed when tool results
    are submitted, not only on turn_complete, so this fails fast instead.
    """

    def test_silence_after_tool_response_ends_the_turn(self):
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        p.followup_timeout = 0.2

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=30))
            await asyncio.sleep(0)
            # tool call, tool response sent... and then nothing ever arrives.
            await p._on_message(tool_msg(
                types.FunctionCall(id="a", name="set_face",
                                   args={"emotion": "happy"})))
            return await task

        t0 = time.monotonic()
        resp = asyncio.run(go())
        assert time.monotonic() - t0 < 5.0, "should not wait for the outer timeout"
        assert resp.text == ""

    def test_late_reply_still_wins_over_the_watchdog(self):
        """A slow but real answer must not be cut off by the watchdog."""
        async def handler(name, args):
            return {"ok": True}

        p = make_provider(handler)
        p.followup_timeout = 1.0

        async def go():
            task = asyncio.create_task(p.complete_turn(timeout=30))
            await asyncio.sleep(0)
            await p._on_message(tool_msg(
                types.FunctionCall(id="a", name="take_photo", args={})))
            await asyncio.sleep(0.2)
            await p._on_message(transcript_msg("A screwdriver."))
            await p._on_message(types.LiveServerMessage(
                server_content=types.LiveServerContent(turn_complete=True)))
            return await task

        assert asyncio.run(go()).text == "A screwdriver."


class TestThinkingLeak:
    """Regression for the first live voice session (2026-09-03).

    This is a reasoning model and its thoughts arrive in output_transcription
    alongside the answer. The robot read its own chain of thought aloud: a JSON
    fragment, three drafts of a greeting, and the text of a safety policy,
    before finally saying hello.
    """

    def test_thoughts_excluded_from_output(self):
        cfg = make_provider()._config()
        assert cfg.thinking_config is not None
        assert cfg.thinking_config.include_thoughts is False


class TestSpeakableGuard:
    """Defence in depth at the speaker boundary — the leak was heard before it
    was seen, so make the next one visible in the log."""

    def test_code_fences_are_stripped(self):
        from minibot.speech.elevenlabs_provider import speakable
        out = speakable('```json\n{}\n```\n\nHey there!')
        assert "```" not in out and "json" not in out
        assert out == "Hey there!"

    def test_ordinary_reply_is_untouched(self):
        from minibot.speech.elevenlabs_provider import speakable
        line = "I see a desk with a keyboard. It's a bit dark in here!"
        assert speakable(line) == line

    def test_long_reply_is_flagged_but_not_truncated(self, caplog):
        from minibot.speech.elevenlabs_provider import (
            SPEAKABLE_WARN_CHARS, speakable,
        )
        long = "word " * (SPEAKABLE_WARN_CHARS // 2)
        with caplog.at_level("WARNING"):
            out = speakable(long)
        assert out == long.strip(), "a genuine long answer must still be spoken"
        assert any("unusually long" in r.message for r in caplog.records)

    def test_empty_is_safe(self):
        from minibot.speech.elevenlabs_provider import speakable
        assert speakable("") == ""


class TestAnswersWhenAddressed:
    """Regression for the first hands-free session (2026-09-03).

    The persona said a silent reaction is often right when the user is deep in
    work. The camera sees a person at a desk, the model concluded they were
    busy, and it answered spoken questions with an expression change and no
    speech — the robot appeared broken. Silence is now scoped to when the
    robot has NOT been addressed.
    """

    @staticmethod
    def _prompt() -> str:
        from minibot.agent.prompts import INSTRUCTIONS
        return " ".join(INSTRUCTIONS.lower().split())

    def test_answering_when_spoken_to_is_mandatory(self):
        p = self._prompt()
        assert "when someone speaks to you, answer them out loud. always." in p

    def test_expression_alone_is_not_an_answer(self):
        assert "changing your face is not an answer on its own" in self._prompt()

    def test_silence_is_scoped_to_not_being_addressed(self):
        p = self._prompt()
        assert "only for when you have not been addressed" in p

    def test_seeing_someone_working_is_not_a_reason_to_go_silent(self):
        assert "seeing someone at a desk is not a reason to go silent" in self._prompt()
