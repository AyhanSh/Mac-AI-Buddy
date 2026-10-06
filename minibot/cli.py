"""Entrypoint. `python -m minibot [options]`

Composition root: the only place that constructs concrete implementations and
wires them into the interfaces. Nothing below this file knows which provider
is in use.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

from .agent.prompts import AUTOPILOT_PERSONA, INSTRUCTIONS
from .agent.robot_agent import RobotAgent
from .ai.provider import AIProvider, ProviderUnavailable
from .audio.dsp import BOT_RATE, wav_to_pcm
from .audio.mac_mic import MacMicUnavailable
from .audio.vad import calibrate
from .config import PROJECT_ROOT, Settings
from .events.bus import Event, EventBus
from .memory.manager import MemoryManager
from .obs import logger
from .obs.logger import AI, APP, MEMORY, ROBOT, SOCIAL, STATE
from .robot.esp32_client import Esp32Client
from .robot.hardware import Esp32RobotHardware
from .social.account import XAccount, describe_failure
from .social.autopilot import Autopilot
from .social.seen import SeenStore
from .social.x_client import XClient, XCredentials
from .speech.elevenlabs_provider import ElevenLabsProvider, list_voices


def build_ai_provider(s: Settings, persona: str = INSTRUCTIONS,
                      with_stt: bool = True) -> AIProvider:
    """Provider selection lives here alone (§24).
    AI_PROVIDER=ollama|gemini|openai — ollama is fully local.

    `persona` is a parameter because autopilot is a different job from being a
    robot on a desk — it has no body, no room, and nobody to answer to — and
    handing it the desk-robot instructions would have it describing movements
    it cannot make to people who cannot see it.
    """
    if s.ai_provider == "ollama":
        from .ai.ollama_provider import OllamaProvider
        from .speech.stt import MlxWhisperSTT
        # Autopilot only ever sends text; loading Whisper there would cost a
        # gigabyte of memory for ears it never uses.
        stt = (MlxWhisperSTT(s.stt_model, s.stt_language,
                             languages=set(s.stt_languages.split(",")))
               if with_stt else None)
        return OllamaProvider(s.ollama_host, s.ollama_model, persona, stt,
                              num_ctx=s.ollama_num_ctx,
                              keep_alive=s.ollama_keep_alive)
    if s.ai_provider == "gemini":
        from .ai.gemini_provider import GeminiRoboticsProvider
        if not s.gemini_api_key:
            sys.exit("AI_PROVIDER=gemini but GEMINI_API_KEY is not set")
        return GeminiRoboticsProvider(s.gemini_api_key, s.gemini_model, persona)
    if s.ai_provider == "openai":
        from .ai.openai_provider import OpenAIRealtimeProvider
        if not s.openai_api_key:
            sys.exit("AI_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAIRealtimeProvider(s.openai_api_key, s.openai_model, persona)
    sys.exit(f"unknown AI_PROVIDER={s.ai_provider!r} "
             "(expected ollama, gemini or openai)")


# How long startup waits for a robot that is required but not answering.
ROBOT_WAIT_S = 30.0


def build_stack(s: Settings, speaker=None):
    """`speaker` is where the voice plays: the robot when None, otherwise
    anything with play_audio()/stop_audio() — the Mac's own speakers."""
    bus = EventBus()
    client = Esp32Client(s.esp32_base_url, s.esp32_timeout, s.esp32_retries,
                         s.esp32_min_interval)
    hardware = Esp32RobotHardware(client)
    speech = ElevenLabsProvider(speaker or hardware, s.elevenlabs_api_key,
                                s.elevenlabs_voice_id, s.elevenlabs_model)
    return bus, client, hardware, speech


def build_memory(s: Settings) -> MemoryManager | None:
    """§21: memory is a nice-to-have, not a dependency the robot's mouth and
    body wait on. Any failure here — no network for the model download, a bad
    EMBEDDING_MODEL name, a locked db file — degrades to no long-term memory
    rather than stopping the robot from starting."""
    try:
        from .memory.embeddings import Model2VecEmbeddingProvider
        from .memory.store import MemoryStore

        db_path = Path(s.memory.db_path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path

        with MEMORY.timed("embedding model loaded"):
            embeddings = Model2VecEmbeddingProvider(s.memory.embedding_model)
        store = MemoryStore(db_path)
        return MemoryManager(store, embeddings, decay_rate=s.memory.decay_rate,
                             min_similarity=s.memory.min_similarity,
                             default_limit=s.memory.retrieval_limit)
    except Exception as e:
        MEMORY.warn(f"memory unavailable, continuing without it: {e!r}")
        return None


def build_x(s: Settings, announce: bool = True) -> XAccount | None:
    """The robot's X account, or None if it doesn't have one.

    Same posture as memory (§21): a social account is a nice-to-have, and no
    failure here may stop the robot from starting. Unconfigured is silent and
    normal. Bad credentials warn and continue *without* the tools registered —
    a robot that thinks it can post but can't would tell the person it posted.
    """
    if not s.x.configured:
        return None
    client = XClient(XCredentials(s.x.api_key, s.x.api_secret,
                                  s.x.access_token, s.x.access_secret),
                     timeout=s.x.timeout, retries=s.x.retries)
    account = XAccount(client, require_confirm=s.x.require_confirm,
                       dry_run=s.x.dry_run, max_per_hour=s.x.max_per_hour,
                       max_per_day=s.x.max_per_day)
    try:
        who = account.verify()
    except Exception as e:
        SOCIAL.warn(f"X credentials rejected, posting disabled: "
                    f"{describe_failure(e)}")
        return None
    # Skipped when the caller is about to change the posture: announcing
    # "confirmation required" and then turning confirmation off one line later
    # is the log telling you a guard is in place that is not.
    if announce:
        mode = ("dry run" if s.x.dry_run else
                "confirmation required" if s.x.require_confirm else
                "POSTS WITHOUT CONFIRMATION")
        SOCIAL.info(f"x: @{who['username']} ({mode}, "
                    f"{s.x.max_per_hour}/hour, {s.x.max_per_day}/day)")
    return account


async def autopilot(s: Settings, once: bool = False,
                    dry_run: bool = False) -> None:
    """Answer mentions with nobody in the room.

    Runs headless on purpose. There is no microphone, no speaker and no robot
    in this path — it is a laptop, an API key and an X account, so it keeps
    working when the ESP32 is unplugged, asleep, or on a different network.
    A reachable robot is used for nothing here and is never waited on.
    """
    x = build_x(s, announce=False)
    if x is None:
        sys.exit("autopilot needs a working X account — run --x-check to see "
                 "why the credentials were refused.")
    if dry_run:
        x.dry_run = True
    # Confirmation is meaningless here: there is nobody to confirm to. The
    # flag on the command line IS the consent, and it is per-run. X_REQUIRE_
    # CONFIRM in .env is deliberately overridden rather than honoured, so the
    # log has to say that plainly instead of repeating what the file said.
    x.require_confirm = False
    posture = ("dry run — nothing is published" if x.dry_run else
               "REPLIES GO OUT UNCONFIRMED")
    SOCIAL.info(f"x: @{x.username} (autopilot: {posture}, "
                f"{s.x.max_per_hour}/hour, {s.x.max_per_day}/day)")

    seen_path = Path(s.x.seen_path)
    if not seen_path.is_absolute():
        seen_path = PROJECT_ROOT / seen_path
    seen = SeenStore(seen_path)

    provider = build_ai_provider(s, AUTOPILOT_PERSONA, with_stt=False)
    provider.register_tools([], _no_tools)   # no tools at all: see autopilot.py
    # Connect once to prove the key works — a bad key should fail now, not in
    # four hours when the first mention arrives — then let the socket go.
    # Autopilot opens a fresh session per mention; an idle Live connection is
    # closed from the far end within minutes anyway.
    try:
        await provider.connect()
    except ProviderUnavailable as e:
        sys.exit(str(e))
    await provider.disconnect()

    pilot = Autopilot(provider, x, seen,
                      max_replies_per_pass=s.x.autopilot_max_replies,
                      mention_limit=s.x.autopilot_mention_limit,
                      poll_seconds=s.x.autopilot_poll_seconds)

    live = "DRY RUN" if x.dry_run else "LIVE — replies are public"
    APP.info(f"autopilot: {live}. up to {s.x.autopilot_max_replies} replies per "
             f"pass, every {_every(s.x.autopilot_poll_seconds)}. ctrl-c to stop.")
    if not x.dry_run:
        APP.warn("nobody is checking these replies before they go out")
    # Reads are the metered half of the X API and this loop does nothing else.
    # Said once at startup rather than as a warning per poll, which would bury
    # the replies you actually want to see in the log.
    per_day = 86400 / max(1.0, s.x.autopilot_poll_seconds)
    if per_day > 500:
        APP.warn(f"polling this often is about {per_day:,.0f} reads a day — "
                 "well past what X's cheaper tiers allow, so expect 429s and "
                 "long stretches where nothing is checked at all")
    try:
        if once:
            await pilot.poll_once()
        else:
            await pilot.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await provider.disconnect()
        APP.info("autopilot stopped")


def _every(seconds: float) -> str:
    """"every 0 min" is what `seconds / 60` prints for a 30 second interval."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


async def _no_tools(name: str, args: dict) -> dict:
    """Autopilot registers no tools; this exists so a model that invents one
    gets a refusal instead of reaching a dispatcher that does something."""
    AI.warn(f"autopilot has no tools; refused {name!r}")
    return {"ok": False, "error": "you have no tools here. Reply with text."}


def x_check(s: Settings) -> None:
    """Credential smoke test, the X analogue of --check. Verifies the four
    OAuth values and reports the account they belong to. Posts nothing."""
    if not s.x.configured:
        missing = [n for n, v in (("X_API_KEY", s.x.api_key),
                                  ("X_API_SECRET", s.x.api_secret),
                                  ("X_ACCESS_TOKEN", s.x.access_token),
                                  ("X_ACCESS_TOKEN_SECRET", s.x.access_secret))
                   if not v]
        sys.exit("X is not configured — missing " + ", ".join(missing))
    client = XClient(XCredentials(s.x.api_key, s.x.api_secret,
                                  s.x.access_token, s.x.access_secret),
                     timeout=s.x.timeout, retries=s.x.retries)
    try:
        who = client.verify()
    except Exception as e:
        sys.exit(f"X check failed: {describe_failure(e)}\n"
                 "If this is 401/403: the app needs 'Read and write' "
                 "permission, and the access token must be regenerated after "
                 "that permission was granted.")
    SOCIAL.info(f"authenticated as @{who['username']} ({who['name']}), "
                f"user id {who['id']}")
    SOCIAL.info(f"require_confirm={s.x.require_confirm} dry_run={s.x.dry_run} "
                f"limits={s.x.max_per_hour}/hour {s.x.max_per_day}/day")


async def run(args, s: Settings) -> None:
    # Build the provider first: a missing key or a typo'd AI_PROVIDER should
    # fail instantly, not hide behind a network timeout to the robot.
    provider = build_ai_provider(s)

    speaker = None
    if s.audio_output == "mac":
        from .audio.mac_speaker import MacSpeaker
        try:
            speaker = MacSpeaker(BOT_RATE, s.speaker_device or None)
        except MacMicUnavailable as e:
            sys.exit(str(e))
    elif s.audio_output != "bot":
        APP.warn(f"unknown AUDIO_OUTPUT={s.audio_output!r}, using the robot's "
                 "speaker")

    bus, client, hardware, speech = build_stack(s, speaker)

    # Probe with short timeouts: the full request policy (8 s x 3 tries) made
    # an unanswering robot cost ~24 s of silence before anything was logged.
    saved = client.timeout, client.retries
    client.timeout, client.retries = 3.0, 0
    try:
        st = hardware.get_status()
        if not (speaker and s.audio_input != "bot"):
            # The robot is required here, but a failed first check is often the
            # robot rejoining WiFi, not a robot that is gone: observed
            # 2026-10-06, "Host is down" at startup and answering again ~20 s
            # later. Give it that long before giving up.
            deadline = time.monotonic() + ROBOT_WAIT_S
            if not st.online:
                APP.warn(f"robot not answering at {s.esp32_base_url} — waiting "
                         f"up to {ROBOT_WAIT_S:.0f} s in case it is rejoining WiFi")
            while not st.online and time.monotonic() < deadline:
                time.sleep(2)
                st = hardware.get_status()
    finally:
        client.timeout, client.retries = saved
    if not st.online:
        # With the voice and the ears both on this computer, a conversation
        # does not need the body. Anything else does: without the robot there
        # would be nothing to hear you or nothing to speak.
        if not (speaker and s.audio_input != "bot"):
            sys.exit(f"robot unreachable at {s.esp32_base_url} — power-cycle "
                     "it, or set AUDIO_OUTPUT=mac to run without it")
        APP.warn(f"robot unreachable at {s.esp32_base_url} — carrying on "
                 "without it: talking through this Mac, but no face, head or "
                 "camera this run")
        from .robot.hardware import DetachedHardware
        hardware = DetachedHardware()
    else:
        APP.info(f"robot: camera={st.camera} oled={st.oled} servos={st.servos} "
                 f"audio={st.audio} mic={st.mic} rssi={st.rssi}dBm")
        if not st.audio and not speaker:
            APP.warn("audio not ready on the robot — you will hear nothing")
        if not st.mic and s.audio_input == "bot":
            APP.warn("mic not ready — use --text")
        if not st.camera:
            APP.warn("camera not ready — /capture will 503 and take_photo "
                     "will fail. Check the serial log for 'Camera init "
                     "failed'.")
    if not speech.configured:
        APP.warn("ElevenLabs not configured — the robot will be silent")

    memory = build_memory(s)
    if memory:
        APP.info(f"memory: {memory.store.count()} memories stored")
    else:
        APP.warn("memory unavailable — conversation continues without it")

    x = build_x(s)
    if not x and s.x.configured:
        APP.warn("X configured but unusable — the robot cannot post this run")

    bus.subscribe(Event.STATE_CHANGED,
                  lambda m: STATE.info(str(m.payload.get("state"))))

    agent = RobotAgent(s, hardware, provider, speech, bus, memory, x,
                       speaker=speaker)

    try:
        try:
            await agent.start()
        except ProviderUnavailable as e:
            sys.exit(str(e))
        except MacMicUnavailable as e:
            # This is the default input now, so its failure is a first-class
            # startup error rather than a traceback out of an opt-in path.
            sys.exit(f"this computer's microphone is unavailable: {e}\n"
                     "Grant your terminal microphone access in System Settings "
                     "> Privacy & Security > Microphone (it has to be approved "
                     "interactively), or set AUDIO_INPUT=bot to listen on the "
                     "robot's own mic instead.")
        if args.text:
            await agent.respond_to_text(args.text)
            return
        APP.info(f"listening — just talk, {s.silence_ms} ms of quiet ends your "
                 f"turn. ctrl-c to stop.")
        await agent.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()
        APP.info("bye")


def check(s: Settings) -> None:
    """Hardware smoke test, ported from mac_realtime.py --check."""
    _, client, hardware, _ = build_stack(s)
    st = hardware.get_status()
    ROBOT.info(f"status: {st.raw}")
    ROBOT.info(f"beep: {client.beep(880, 200)}")
    tone = np.sin(2 * np.pi * 440 * np.arange(BOT_RATE) / BOT_RATE) * 8000
    client.play(tone.astype("<i2").tobytes())
    ROBOT.info("sent 1 s of 440 Hz to /say — you should hear it")
    ROBOT.info("recording 2 s...")
    wav = client.record(2000)
    with open("mic.wav", "wb") as f:
        f.write(wav)
    pcm, rate = wav_to_pcm(wav)
    a = np.frombuffer(pcm, dtype="<i2")
    ROBOT.info(f"mic.wav saved: {len(a)} samples @ {rate} Hz, "
               f"peak {int(np.abs(a).max())}")
    if np.abs(a).max() < 500:
        ROBOT.warn("peak is very low — check the MAX4466 gain pot and wiring")


def open_mic(s: Settings):
    """The same AUDIO_INPUT selection RobotAgent._build_mic() does, minus the
    agent-lifecycle bits (probe_vad, boot chirp) that don't apply standalone."""
    if s.audio_input == "bot":
        _, client, _, _ = build_stack(s)
        return client
    from .audio.mac_mic import MacMicSource
    return MacMicSource(BOT_RATE, s.mic_device or None)


def levels(s: Settings, seconds: float = 15.0) -> None:
    """Live mic meter against the calibrated thresholds.

    Added after a session where the robot never triggered: the onset threshold
    was simply higher than anything the microphone produced, which is invisible
    from the normal logs. Respects AUDIO_INPUT — this is the actual tool for
    diagnosing that class of problem, so it needs to look at whichever mic is
    actually configured, not always the ESP32's.
    """
    on_bot = s.audio_input == "bot"
    mic = open_mic(s)
    APP.info("mic: " + ("robot (AUDIO_INPUT=bot)" if on_bot else "this computer"))
    room = calibrate(mic, onset_mult=s.vad_onset_mult,
                     onset_margin=s.vad_onset_margin,
                     endpoint_mult=s.vad_endpoint_mult,
                     endpoint_margin=s.vad_endpoint_margin)
    APP.info(f"speak now — {seconds:.0f}s. onset={room.onset:.0f} "
             f"endpoint={room.endpoint:.0f}")
    over, total, peak = 0, 0, 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        try:
            rms = float(mic.level()["rms"])
        except Exception as e:
            APP.warn(f"level read failed: {e!r}")
            continue
        total += 1
        peak = max(peak, rms)
        hit = rms > room.onset
        over += hit
        bar = "#" * min(50, int(rms / 10))
        print(f"  {rms:6.0f} |{bar}{'  <-- TRIGGER' if hit else ''}")
        time.sleep(0.12)
    print()
    APP.info(f"peak={peak:.0f} onset={room.onset:.0f} "
             f"crossed {over}/{total} samples")
    if not over:
        # The gain advice has to match the mic actually in use, or it sends
        # you to a potentiometer on a board that isn't listening.
        louder = ("turn up the MAX4466 gain pot" if on_bot else
                  "raise the input volume in System Settings > Sound > Input")
        APP.warn(f"nothing crossed the threshold. Either speak louder/closer, "
                 f"{louder}, or lower VAD_ONSET_MULT in .env")


def main() -> None:
    ap = argparse.ArgumentParser(prog="minibot")
    ap.add_argument("--text", help="send one typed turn and exit")
    ap.add_argument("--check", action="store_true", help="hardware check only")
    ap.add_argument("--levels", action="store_true",
                    help="live mic meter vs the voice-onset threshold")
    ap.add_argument("--list-voices", action="store_true",
                    help="print your ElevenLabs voices and exit")
    ap.add_argument("--x-check", action="store_true",
                    help="verify the X credentials and exit (posts nothing)")
    ap.add_argument("--autopilot", action="store_true",
                    help="answer X mentions unattended. No robot, no "
                         "microphone, and no confirmation before replies go "
                         "out — rehearse it with --dry-run first")
    ap.add_argument("--once", action="store_true",
                    help="with --autopilot: check mentions a single time and "
                         "exit, instead of polling forever")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --autopilot: compose and log replies without "
                         "publishing any of them")
    ap.add_argument("--provider", choices=["ollama", "gemini", "openai"],
                    help="override AI_PROVIDER for this run")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args()

    s = Settings.load()
    if args.provider:
        s = type(s)(**{**s.__dict__, "ai_provider": args.provider})
    logger.setup(args.log_level or s.log_level)

    if args.list_voices:
        if not s.elevenlabs_api_key:
            sys.exit("set ELEVENLABS_API_KEY")
        for v in list_voices(s.elevenlabs_api_key):
            cat = v.get("category", "")
            mark = "  <-- cloned" if cat in ("cloned", "professional") else ""
            print(f"{v['voice_id']}  {v.get('name','?'):<24} {cat}{mark}")
        print("\nexport ELEVENLABS_VOICE_ID=<the id you want>")
        return
    if args.x_check:
        x_check(s)
        return
    if args.check:
        check(s)
        return
    if args.levels:
        levels(s)
        return
    if args.autopilot:
        asyncio.run(autopilot(s, once=args.once, dry_run=args.dry_run))
        return
    try:
        asyncio.run(run(args, s))
    except KeyboardInterrupt:
        pass    # ctrl-c is how you stop it; "bye" is already logged


if __name__ == "__main__":
    main()
