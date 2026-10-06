"""Local Ollama provider tests.

Drives the real provider logic against a scripted /api/chat, so neither Ollama
nor Whisper needs to be running.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.ai.ollama_provider import OllamaProvider  # noqa: E402
from minibot.ai.provider import IMAGE_RESULT_KEY, ToolSpec  # noqa: E402
from minibot.speech.stt import SpeechToText  # noqa: E402

TOOLS = [
    ToolSpec("look_at", "Turn the head.",
             {"type": "object", "properties": {"pan": {"type": "integer"}}}),
    ToolSpec("take_photo", "Look.", {"type": "object", "properties": {}}),
]


class FakeSTT(SpeechToText):
    def __init__(self, text: str):
        self.text = text

    async def load(self) -> None:
        pass

    async def transcribe(self, pcm: bytes, rate: int) -> str:
        return self.text


def make(replies, handler=None, stt=None, **kw):
    """A provider whose /api/chat answers with `replies` in order."""
    p = OllamaProvider("http://x", "m", "be a robot", stt, **kw)
    calls = []

    async def fake_post(path, body):
        calls.append(body)
        return {"message": replies.pop(0)}

    p._post = fake_post
    p.register_tools(TOOLS, handler or _ok)
    return p, calls


async def _ok(name, args):
    return {"ok": True}


def tool(name, **args):
    return {"function": {"name": name, "arguments": args}}


def test_plain_text_turn():
    p, calls = make([{"content": "hello there"}])

    async def go():
        await p.send_text("hi")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert r.text == "hello there"
    assert calls[0]["messages"][0] == {"role": "system", "content": "be a robot"}
    assert calls[0]["messages"][-1] == {"role": "user", "content": "hi"}
    assert [t["function"]["name"] for t in calls[0]["tools"]] == ["look_at",
                                                                 "take_photo"]


def test_tool_loop_runs_to_a_spoken_answer():
    seen = []

    async def handler(name, args):
        seen.append((name, args))
        return {"ok": True, "pan": args.get("pan")}

    p, calls = make([{"content": "", "tool_calls": [tool("look_at", pan=30)]},
                     {"content": "Looking left."}], handler)

    async def go():
        await p.send_text("look left")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert seen == [("look_at", {"pan": 30})]
    assert r.text == "Looking left."
    assert [c.name for c in r.tool_calls] == ["look_at"]
    roles = [m["role"] for m in calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert calls[1]["messages"][-1]["tool_name"] == "look_at"


def test_photo_follows_as_an_image_and_is_dropped_after_the_turn():
    async def handler(name, args):
        return {"ok": True, IMAGE_RESULT_KEY: b"JPEG"}

    p, calls = make([{"content": "", "tool_calls": [tool("take_photo")]},
                     {"content": "A mug."}], handler)

    async def go():
        await p.send_text("what do you see")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert r.text == "A mug."
    last = calls[1]["messages"][-1]
    assert last["role"] == "user"
    assert last["images"] == [base64.b64encode(b"JPEG").decode()]
    # The raw bytes never reach the JSON tool message.
    assert "JPEG" not in calls[1]["messages"][-2]["content"]
    assert not any("images" in m for m in p._history)


def test_tool_rounds_are_capped_and_still_end_in_words():
    loop = {"content": "", "tool_calls": [tool("look_at", pan=90)]}
    p, calls = make([dict(loop), dict(loop), {"content": "Done."}],
                    max_tool_rounds=2)

    async def go():
        await p.send_text("spin")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert r.text == "Done."
    assert "tools" not in calls[-1]      # the last call offers none


def test_audio_is_transcribed_locally():
    p, calls = make([{"content": "Paris."}], stt=FakeSTT("capital of France?"))

    async def go():
        await p.send_audio(b"\0\0" * 1600, 16000)
        return await p.complete_turn()

    r = asyncio.run(go())
    assert r.text == "Paris."
    assert calls[0]["messages"][-1]["content"] == "capital of France?"


def test_context_without_an_utterance_calls_nothing():
    """A muted clip that transcribes to nothing leaves only the mute reminder
    queued; that is not a turn and must not reach the model."""
    p, calls = make([], stt=FakeSTT(""))

    async def go():
        await p.send_text("You are MUTED...", turn_complete=False)
        await p.send_audio(b"\0\0" * 1600, 16000)
        return await p.complete_turn()

    r = asyncio.run(go())
    assert r.text == "" and calls == [] and p._history == []


def test_memory_context_is_sent_with_the_turn():
    p, calls = make([{"content": "Espresso, right?"}])

    async def go():
        await p.send_text("RELEVANT MEMORY: likes espresso", turn_complete=False)
        await p.send_text("what coffee do I like")
        return await p.complete_turn()

    asyncio.run(go())
    content = calls[0]["messages"][-1]["content"]
    assert "likes espresso" in content and "what coffee" in content


def test_thinking_is_never_spoken():
    p, _ = make([{"content": "<think>plan the greeting</think>Hi!"}])

    async def go():
        await p.send_text("hello")
        return await p.complete_turn()

    assert asyncio.run(go()).text == "Hi!"


def test_failed_turn_leaves_no_half_exchange_in_history():
    p, _ = make([])

    async def boom(path, body):
        raise RuntimeError("ollama /api/chat 500: out of memory")

    p._post = boom

    async def go():
        await p.send_text("hi")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert "out of memory" in r.error
    assert p._history == []


def test_history_is_trimmed_at_a_user_boundary():
    replies = []
    for _ in range(10):
        replies += [{"content": "", "tool_calls": [tool("look_at", pan=90)]},
                    {"content": "ok"}]
    p, _ = make(replies, max_history=6)

    async def go():
        for i in range(10):
            await p.send_text(f"turn {i}")
            await p.complete_turn()

    asyncio.run(go())
    assert len(p._history) <= 6
    assert p._history[0]["role"] == "user"
    assert p._history[0]["content"] == "turn 9"


@pytest.mark.parametrize("raw,expected", [
    ("localhost:11434", "http://localhost:11434"),
    ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
])
def test_ollama_host_accepts_ollamas_own_format(raw, expected):
    from minibot.config import _url
    assert _url(raw) == expected


class FakeStream:
    """Stands in for requests' streaming response: one JSON object per line."""

    def __init__(self, chunks):
        import json
        self.status_code = 200
        self._lines = [json.dumps(c).encode() for c in chunks]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        yield from self._lines


def streaming(rounds, handler=None):
    """A provider streaming each of `rounds` (a list of chunk lists) in turn,
    with on_text recording what would have been spoken."""
    p = OllamaProvider("http://x", "m", "be a robot")
    p.register_tools(TOOLS, handler or _ok)
    spoken = []

    async def on_text(chunk):
        spoken.append(chunk)

    p.on_text = on_text
    p._http.post = lambda *a, **kw: FakeStream(rounds.pop(0))
    return p, spoken


def piece(text="", calls=None, done=False):
    m = {"role": "assistant", "content": text}
    if calls:
        m["tool_calls"] = calls
    return {"message": m, "done": done}


def test_streamed_reply_is_spoken_sentence_by_sentence():
    p, spoken = streaming([[piece("Hi! I am Mini"), piece(" Bot. I live on"),
                            piece(" a desk."), piece(done=True)]])

    async def go():
        await p.send_text("who are you")
        return await p.complete_turn()

    r = asyncio.run(go())
    # "Hi!" alone is too short to send by itself, so it waits for company.
    assert spoken == ["Hi! I am Mini Bot.", "I live on a desk."]
    assert r.streamed and r.text == "Hi! I am Mini Bot. I live on a desk."


def test_streamed_tool_round_then_answer():
    seen = []

    async def handler(name, args):
        seen.append(name)
        return {"ok": True}

    p, spoken = streaming([
        [piece(calls=[tool("look_at", pan=20)]), piece(done=True)],
        [piece("Now looking left."), piece(done=True)],
    ], handler)

    async def go():
        await p.send_text("look left")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert seen == ["look_at"]
    assert spoken == ["Now looking left."]
    assert p._history[1]["tool_calls"][0]["function"]["name"] == "look_at"


def test_stream_error_is_reported_not_spoken():
    p, spoken = streaming([[{"error": "model crashed"}]])

    async def go():
        await p.send_text("hi")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert "model crashed" in r.error and spoken == [] and p._history == []


def test_a_direct_reply_ends_the_turn_without_another_model_round():
    """A tool that already knows what to say (an external tool's report) skips the round
    that would only rephrase it."""
    from minibot.ai.provider import DIRECT_REPLY_KEY

    async def handler(name, args):
        return {"ok": True, DIRECT_REPLY_KEY: "YouTube is open."}

    p, spoken = streaming([
        [piece("On it.", calls=[tool("look_at", pan=20)]), piece(done=True)],
        # Never reached: there is no second model call.
    ], handler)

    async def go():
        await p.send_text("open youtube")
        return await p.complete_turn()

    r = asyncio.run(go())
    assert spoken == ["On it.", "YouTube is open."]
    assert r.text == "On it. YouTube is open."
    assert [m["role"] for m in p._history] == ["user", "assistant", "tool",
                                               "assistant"]
    # The marker itself never reaches the model's context.
    assert DIRECT_REPLY_KEY not in p._history[2]["content"]


def test_without_a_direct_reply_the_model_still_answers():
    p, calls = make([{"content": "", "tool_calls": [tool("look_at", pan=30)]},
                     {"content": "It failed, sorry."}])

    async def go():
        await p.send_text("look left")
        return await p.complete_turn()

    assert asyncio.run(go()).text == "It failed, sorry."
    assert len(calls) == 2
