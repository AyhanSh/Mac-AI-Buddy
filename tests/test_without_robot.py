"""Running with the voice and mic on the Mac and no robot body attached."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.agent.robot_agent import RobotAgent  # noqa: E402
from minibot.config import Settings  # noqa: E402
from minibot.events.bus import EventBus  # noqa: E402
from minibot.robot.hardware import DetachedHardware  # noqa: E402


def agent():
    return RobotAgent(Settings(), DetachedHardware(), provider=None,
                      speech=None, bus=EventBus())


def test_physical_tools_say_plainly_that_there_is_no_body():
    a = agent()

    async def go():
        return [await a._dispatch(n, args) for n, args in
                (("look_at", {"pan": 30}), ("set_face", {"emotion": "happy"}),
                 ("take_photo", {}))]

    for r in asyncio.run(go()):
        assert r["ok"] is False
        assert "not connected" in r["error"]


def test_faces_and_mute_do_not_break_the_conversation():
    a = agent()

    async def go():
        await a._face("thinking")             # swallowed, not raised
        return await a._dispatch("mute", {})

    r = asyncio.run(go())
    assert r["ok"] and a.muted
