#!/usr/bin/env python3
"""
mac_realtime.py — AI Mini Bot brain on the OpenAI Realtime API (gpt-realtime-2)

Architecture
    bot mic  --(WAV)-->  Mac  --(pcm16 24k)-->  Realtime API
    bot spkr <--(pcm16 16k)-- Mac <--(audio deltas)-- Realtime API
    head/face/camera <--(HTTP)-- Mac <--(tool calls)-- Realtime API

Turn-based, not full duplex. You talk, it waits a second for you to finish, it
answers. Nothing to press. What you keep: the model's own speech understanding,
its voice, and tool calling. What you lose: barge-in mid-sentence.

Turn-taking happens in two stages, because the bot cannot listen and be told to
stop at the same time — /mic owns the core for the whole recording. So the Mac
polls /level to notice you have started, then hands the endpointing job to the
firmware: /mic?silence=1000 keeps recording until you have been quiet that
long. Firmware without that support falls back to chunked recording, which is
deaf for a WiFi round trip between chunks and loses audio at every seam.

That is also the right trade for this hardware anyway — the speaker and mic sit
on the same board, so without echo cancellation a full-duplex session would
hear itself and interrupt constantly.

Setup
    pip install requests websockets numpy
    export OPENAI_API_KEY=...
    export BOT_URL=http://192.168.1.79

    Optional — speak in your own cloned voice instead of the model's:
    export ELEVENLABS_API_KEY=...
    export ELEVEN_VOICE_ID=...        # python mac_realtime.py --list-voices
    export VOICE_PROVIDER=eleven

Run
    python mac_realtime.py --check                 # bot reachable? speaker works?
    python mac_realtime.py --list-voices           # your ElevenLabs voices
    python mac_realtime.py --voice eleven --say "hello"   # TTS only, no session
    python mac_realtime.py --text "hello there"    # type instead of speak
    python mac_realtime.py                         # voice loop, hands free
    python mac_realtime.py --silence-ms 700        # end turns sooner
    python mac_realtime.py --push                  # press Enter to talk instead

API note
    Realtime event names and the session schema have moved between releases.
    If the socket errors on session.update, diff build_session() below against
    the current docs at developers.openai.com. MODEL is also worth checking —
    gpt-realtime-2.1 and 2.1-mini exist alongside gpt-realtime-2.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import wave

from collections import deque

import numpy as np
import requests
import websockets

BOT_URL = os.environ.get("BOT_URL", "http://192.168.1.79")
MODEL   = os.environ.get("REALTIME_MODEL", "gpt-realtime-2")
WS_URL  = f"wss://api.openai.com/v1/realtime?model={MODEL}"

API_RATE = 24000    # Realtime API pcm16 is 24 kHz mono
BOT_RATE = 16000    # AUDIO_RATE in the firmware

# ---------- turn taking ----------
SILENCE_MS    = 1000   # quiet needed to end your turn
MAX_TURN_MS   = 12000  # hard cap on one utterance (firmware allows 15000)
LEAD_MS       = 1500   # if the trigger was a false alarm, bail out this fast
MIC_FRAME_MS  = 16     # matches MIC_FRAME / MIC_RATE in the firmware
# recordMic() strips DC and multiplies 12-bit samples by 16 on the way out, so
# /level readings (raw ADC counts) need this factor to compare against the WAV.
WAV_GAIN      = 16

VOICE = "cedar"     # OpenAI voice: alloy, echo, shimmer, cedar, marin

# ---------- ElevenLabs ----------
# Your cloned voice. Find the id with:  python mac_realtime.py --list-voices
ELEVEN_KEY      = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "")
ELEVEN_MODEL    = os.environ.get("ELEVEN_MODEL", "eleven_flash_v2_5")
ELEVEN_URL      = "https://api.elevenlabs.io/v1"
# pcm_16000 lands exactly on BOT_RATE, so nothing gets resampled on this path.
# Only 44.1 kHz PCM needs a Pro subscription.
ELEVEN_FORMAT   = "pcm_16000"
ELEVEN_SETTINGS = {
    "stability": 0.45,          # lower = more expressive, more variable
    "similarity_boost": 0.80,   # how tightly it hugs your clone
    "style": 0.30,              # 0 for a neutral read, higher for character
    "use_speaker_boost": True,
}

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprised", "thinking",
            "listening", "talking", "sleep", "searching", "loading",
            "scanning", "wifi", "memory", "saving"]

INSTRUCTIONS = """You are Mini Bot, a small desk robot built by Aykhan, an electronics
hobbyist in Warsaw. You have a camera, an animated OLED face, a pan/tilt head
and a speaker.

Be warm, curious, concise and a little playful. Keep spoken replies to one or
two sentences — you are a small robot on a desk, not a podcast.

Use your tools naturally as part of talking:
  set_face   whenever your mood shifts. Do this often, it is your main
             expression. Set it BEFORE you say the matching thing.
  look_at    to turn your head toward what you are discussing, or up when
             you are thinking.
  take_photo when you need to see. You cannot see anything until you call it.

Your microphone is cheap and shares power with a camera and a WiFi radio, so
transcription may be rough. If something makes no sense, say so rather than
guessing wildly."""

TOOLS = [
    {
        "type": "function",
        "name": "set_face",
        "description": "Change the expression on the OLED face.",
        "parameters": {
            "type": "object",
            "properties": {"emotion": {"type": "string", "enum": EMOTIONS}},
            "required": ["emotion"],
        },
    },
    {
        "type": "function",
        "name": "look_at",
        "description": ("Turn the head. pan 15-165 where 90 is centre and lower "
                        "looks left. tilt 30-150 where 90 is level and lower looks up."),
        "parameters": {
            "type": "object",
            "properties": {
                "pan":  {"type": "integer", "minimum": 15, "maximum": 165},
                "tilt": {"type": "integer", "minimum": 30, "maximum": 150},
            },
        },
    },
    {
        "type": "function",
        "name": "take_photo",
        "description": ("Capture a frame from the camera and look at it. Call this "
                        "whenever you need to see what is in front of you."),
        "parameters": {"type": "object", "properties": {}},
    },
]

def build_session(provider="openai"):
    """OpenAI voice: the model speaks directly, lowest latency.
    ElevenLabs: the model returns text, we synthesise it in your cloned voice.
    Costs an extra round trip but the voice is yours."""
    s = {
        "type": "realtime",
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    if provider == "eleven":
        s["output_modalities"] = ["text"]
        s["audio"] = {"input": {"format": {"type": "audio/pcm", "rate": API_RATE},
                                "turn_detection": None}}
    else:
        s["output_modalities"] = ["audio"]
        s["audio"] = {
            "input":  {"format": {"type": "audio/pcm", "rate": API_RATE},
                       "turn_detection": None},
            "output": {"format": {"type": "audio/pcm", "rate": API_RATE},
                       "voice": VOICE},
        }
    return s


# ------------------------------------------------------------------
# ElevenLabs TTS
# ------------------------------------------------------------------
def eleven_voices():
    r = requests.get(f"{ELEVEN_URL}/voices",
                     headers={"xi-api-key": ELEVEN_KEY}, timeout=15)
    r.raise_for_status()
    return r.json().get("voices", [])


def eleven_stream(text: str, on_chunk, chunk_bytes=BOT_RATE * 2 // 2):
    """Streams pcm_16000 and calls on_chunk(pcm) as it arrives, so the bot
    starts speaking before synthesis finishes. Default chunk is ~500 ms."""
    if not ELEVEN_KEY:
        raise RuntimeError("set ELEVENLABS_API_KEY")
    if not ELEVEN_VOICE_ID:
        raise RuntimeError("set ELEVEN_VOICE_ID (run --list-voices)")

    url = f"{ELEVEN_URL}/text-to-speech/{ELEVEN_VOICE_ID}/stream"
    body = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": ELEVEN_SETTINGS,
    }
    with requests.post(url, json=body, stream=True, timeout=60,
                       headers={"xi-api-key": ELEVEN_KEY,
                                "Content-Type": "application/json"},
                       params={"output_format": ELEVEN_FORMAT}) as r:
        if r.status_code != 200:
            raise RuntimeError(f"elevenlabs {r.status_code}: {r.text[:200]}")
        buf = bytearray()
        for part in r.iter_content(chunk_size=4096):
            if not part:
                continue
            buf += part
            while len(buf) >= chunk_bytes:
                on_chunk(bytes(buf[:chunk_bytes]))
                del buf[:chunk_bytes]
        if buf:
            on_chunk(bytes(buf))


# ------------------------------------------------------------------
# Resampling — linear interpolation, good enough for speech
# ------------------------------------------------------------------
def resample(pcm: bytes, src: int, dst: int) -> bytes:
    if src == dst or not pcm:
        return pcm
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = int(len(a) * dst / src)
    if n_out < 1:
        return b""
    xi = np.linspace(0, len(a) - 1, n_out)
    out = np.interp(xi, np.arange(len(a)), a)
    return np.clip(out, -32768, 32767).astype("<i2").tobytes()


def wav_to_pcm(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return b.getvalue()


def frame_rms(pcm: bytes, rate: int) -> np.ndarray:
    """RMS per MIC_FRAME_MS window, each measured about its own mean so a
    drifting bias cannot read as loudness. Same shape as the firmware's VAD."""
    a = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = max(1, int(rate * MIC_FRAME_MS / 1000))
    if len(a) < n:
        return np.zeros(0, np.float32)
    f = a[: len(a) - len(a) % n].reshape(-1, n)
    return np.sqrt(((f - f.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))


# ------------------------------------------------------------------
# Bot HTTP client
# ------------------------------------------------------------------
class Bot:
    def __init__(self, base=BOT_URL, timeout=8):
        self.base = base.rstrip("/")
        self.t = timeout
        self.s = requests.Session()

    def _get(self, path, **params):
        r = self.s.get(self.base + path, params=params or None, timeout=self.t)
        r.raise_for_status()
        return r

    def status(self):   return self._get("/status").json()
    def capture(self):  return self._get("/capture").content
    def level(self):    return self._get("/level").json()
    def center(self):   return self._get("/look/center").text
    def stop(self):     return self._get("/stop").text
    def beep(self, f=880, ms=150): return self._get("/beep", f=f, ms=ms).text
    def volume(self, v):           return self._get("/volume", v=int(v)).text

    def face(self, emotion):
        if emotion not in EMOTIONS:
            emotion = "neutral"
        return self._get("/set", e=emotion).text

    def look(self, pan=None, tilt=None):
        p = {}
        if pan  is not None: p["pan"]  = int(pan)
        if tilt is not None: p["tilt"] = int(tilt)
        return self._get("/look", **p).text if p else None

    def play(self, pcm16k: bytes):
        """Raw int16 LE mono at BOT_RATE. Returns when the upload finishes,
        not when playback does."""
        if not pcm16k:
            return
        r = self.s.post(self.base + "/say", data=pcm16k,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=30)
        r.raise_for_status()
        return r.json()

    def record(self, max_ms=3500, silence_ms=0, thresh=0, lead_ms=LEAD_MS) -> bytes:
        """Fixed-length by default. With silence_ms the firmware endpoints the
        recording itself and returns as soon as you stop talking."""
        p = {"ms": int(max_ms)}
        if silence_ms:
            p.update(silence=int(silence_ms), thresh=int(thresh), lead=int(lead_ms))
        r = self.s.get(self.base + "/mic", params=p, timeout=max_ms / 1000 + 10)
        r.raise_for_status()
        return r.content

    def probe_vad(self) -> bool:
        """True if the firmware endpoints recordings itself. Older builds just
        ignore the extra args and record the full span, so tell them apart by
        how much audio comes back: an unreachable threshold means nothing ever
        counts as speech, so a VAD build gives up after `lead` instead of `ms`."""
        try:
            wav = self.record(max_ms=3000, silence_ms=100, thresh=32000, lead_ms=200)
            pcm, rate = wav_to_pcm(wav)
        except (requests.RequestException, wave.Error):
            return False
        return len(pcm) / 2 / rate < 1.0

    def wait_quiet(self, timeout=30):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if not self.status().get("speaking"):
                    return
            except requests.RequestException:
                pass
            time.sleep(0.15)


class Room:
    """Running estimate of the noise floor, in raw ADC counts, and the two
    thresholds taken from it. Onset sits above endpoint deliberately: it should
    cost more energy to start a turn than to keep one going, so a sentence does
    not end on its own quiet consonants.

    The floor tracks the room rather than being measured once, because the fans
    and the servos and the camera all change what quiet sounds like."""

    def __init__(self, floor=60.0):
        self.hist = deque([floor] * 8, maxlen=40)

    @property
    def floor(self):
        return float(np.median(self.hist))

    @property
    def onset(self):
        f = self.floor
        return max(f * 3.0, f + 45)

    @property
    def endpoint(self):
        f = self.floor
        return max(f * 1.8, f + 20)

    def observe(self, rms):
        # Only quiet readings update the floor, or the first loud sentence
        # would teach it that shouting is normal.
        if rms < self.onset:
            self.hist.append(float(rms))


def calibrate(bot: Bot, n=10) -> Room:
    seen = []
    for _ in range(n):
        try:
            seen.append(bot.level()["rms"])
        except requests.RequestException:
            pass
    room = Room()
    room.hist.clear()
    room.hist.extend(float(s) for s in (seen or [60.0]))
    return room


def wait_for_voice(bot: Bot, room: Room, consecutive=2, poll=0.02):
    """Block until sustained sound. Each /level call already takes ~20 ms on
    the device plus a round trip, so two hits is roughly 150 ms of evidence —
    enough to ignore a single click, short enough that little is clipped off
    the front of your first word before /mic takes over."""
    hits = 0
    while True:
        try:
            rms = bot.level()["rms"]
        except requests.RequestException:
            time.sleep(0.4)
            continue
        room.observe(rms)
        hits = hits + 1 if rms > room.onset else 0
        if hits >= consecutive:
            return
        time.sleep(poll)


def has_speech(wav: bytes, room: Room, min_ms=120) -> bool:
    """Did anything actually get said? Guards against a door slam opening a
    turn and burning an API round trip on a second of room tone."""
    pcm, rate = wav_to_pcm(wav)
    r = frame_rms(pcm, rate)
    return float((r > room.endpoint * WAV_GAIN).sum()) * MIC_FRAME_MS >= min_ms


def trim_tail(wav: bytes, room: Room, keep_ms=250) -> bytes:
    """Drop the dead air the endpointer leaves behind. The model does not need
    to sit through the silence that ended your sentence, and it is charged for
    the audio either way."""
    pcm, rate = wav_to_pcm(wav)
    r = frame_rms(pcm, rate)
    loud = np.nonzero(r > room.endpoint * WAV_GAIN)[0]
    if not len(loud):
        return wav
    keep = int((loud[-1] + 1) * MIC_FRAME_MS + keep_ms) * rate // 1000
    return pcm_to_wav(pcm[: keep * 2], rate)


def record_chunked(bot: Bot, room: Room, max_ms, silence_ms, chunk_ms=900) -> bytes:
    """Endpointing for firmware that cannot do it itself. /mic only records a
    fixed span, so take it in chunks and stop once the tail goes quiet.

    The mic is deaf for a WiFi round trip between chunks, so syllables land in
    the seams. Reflash for the endpointing build if transcripts read badly."""
    gate = room.endpoint * WAV_GAIN
    need = max(1, int(round(silence_ms / MIC_FRAME_MS)))
    pcm, rate, took = bytearray(), BOT_RATE, 0
    while took < max_ms:
        chunk, rate = wav_to_pcm(bot.record(max_ms=min(chunk_ms, max_ms - took)))
        pcm += chunk
        took += chunk_ms
        r = frame_rms(bytes(pcm), rate)
        if len(r) >= need and (r > gate).any() and (r[-need:] <= gate).all():
            break
    return pcm_to_wav(bytes(pcm), rate)


def capture_turn(bot: Bot, room: Room, vad: bool, max_ms, silence_ms,
                 lead_ms=LEAD_MS) -> bytes:
    """Everything you just said, ending silence_ms after you stop."""
    if vad:
        return bot.record(max_ms=max_ms, silence_ms=silence_ms,
                          thresh=int(room.endpoint), lead_ms=lead_ms)
    return record_chunked(bot, room, max_ms, silence_ms)


# ------------------------------------------------------------------
# Realtime session
# ------------------------------------------------------------------
class Session:
    """One long-lived WebSocket. Audio out is forwarded to the bot in chunks
    as it arrives, so speech starts before the model finishes generating."""

    CHUNK_MS = 400   # how much API audio to buffer before pushing to the bot

    def __init__(self, bot: Bot, key: str, provider="openai"):
        self.bot = bot
        self.key = key
        self.provider = provider
        self.ws = None
        self.buf = bytearray()
        self.text_buf = ""
        self.pending_photo = None
        self.turn_done = asyncio.Event()
        # The API allows exactly one response in flight. `active` tracks that,
        # optimistically — set when we ask for a response, not when the server
        # confirms — because two creates sent back to back would both look
        # inactive otherwise.
        self.active = False
        self.want_followup = False
        self.tool_rounds = 0

    async def connect(self):
        self.ws = await websockets.connect(
            WS_URL,
            additional_headers={"Authorization": f"Bearer {self.key}"},
            max_size=None,
        )
        await self.send({"type": "session.update",
                         "session": build_session(self.provider)})
        voice = f"ElevenLabs {ELEVEN_VOICE_ID[:8]}…" if self.provider == "eleven" \
                else f"OpenAI {VOICE}"
        print(f"connected: {MODEL}, voice {voice}")

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    # -- audio out -------------------------------------------------
    async def push_audio(self, force=False):
        want = API_RATE * self.CHUNK_MS // 1000 * 2
        while len(self.buf) >= want or (force and self.buf):
            take = bytes(self.buf[:want]) if len(self.buf) >= want else bytes(self.buf)
            del self.buf[:len(take)]
            pcm = resample(take, API_RATE, BOT_RATE)
            await asyncio.get_running_loop().run_in_executor(None, self.bot.play, pcm)
            if not force:
                break

    async def speak_eleven(self, text: str):
        """Synthesise in the cloned voice and stream it to the bot."""
        text = text.strip()
        if not text:
            return
        loop = asyncio.get_running_loop()

        def work():
            eleven_stream(text, self.bot.play)

        try:
            await loop.run_in_executor(None, work)
        except Exception as e:
            print("  elevenlabs failed:", e)
            # Don't leave the bot mute — fall back to a chirp so you notice.
            await loop.run_in_executor(None, lambda: self.bot.beep(400, 200))

    # -- tools -----------------------------------------------------
    async def run_tool(self, name, args, call_id):
        loop = asyncio.get_running_loop()
        result = {"ok": True}
        try:
            if name == "set_face":
                emo = args.get("emotion", "neutral")
                await loop.run_in_executor(None, self.bot.face, emo)
                print(f"  [face] {emo}")
            elif name == "look_at":
                pan, tilt = args.get("pan"), args.get("tilt")
                await loop.run_in_executor(None, lambda: self.bot.look(pan, tilt))
                print(f"  [look] pan={pan} tilt={tilt}")
            elif name == "take_photo":
                jpeg = await loop.run_in_executor(None, self.bot.capture)
                self.pending_photo = jpeg
                print(f"  [photo] {len(jpeg)} bytes")
                result = {"ok": True, "note": "photo attached to the conversation"}
            else:
                result = {"ok": False, "error": f"unknown tool {name}"}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
            print(f"  [tool error] {name}: {e}")

        await self.send({
            "type": "conversation.item.create",
            "item": {"type": "function_call_output",
                     "call_id": call_id,
                     "output": json.dumps(result)},
        })

        # The camera frame goes in as a separate user message; a function
        # result is text only and cannot carry an image.
        if self.pending_photo:
            b64 = base64.b64encode(self.pending_photo).decode()
            self.pending_photo = None
            await self.send({
                "type": "conversation.item.create",
                "item": {"role": "user", "type": "message",
                         "content": [{"type": "input_image",
                                      "image_url": f"data:image/jpeg;base64,{b64}"}]},
            })
            # A photo is only useful if the model gets to look at it, so this
            # one does need a follow-up even if the model already spoke.
            self.want_followup = True

        # No response.create here. One response may carry several tool calls,
        # and asking for a new response after each would collide with the one
        # the earlier call already started — that is the
        # conversation_already_has_active_response error. The single follow-up
        # is issued from response.done, once every call in the batch is in.

    async def create_response(self) -> bool:
        """One response at a time, or the API rejects the second."""
        if self.active:
            return False
        self.active = True
        await self.send({"type": "response.create"})
        return True

    # -- event pump ------------------------------------------------
    async def pump(self):
        async for raw in self.ws:
            ev = json.loads(raw)
            t = ev.get("type", "")

            if t.endswith("audio.delta") and "delta" in ev:
                self.buf += base64.b64decode(ev["delta"])
                await self.push_audio()

            elif t.endswith("output_text.delta") and "delta" in ev:
                self.text_buf += ev["delta"]

            elif t.endswith("output_text.done"):
                said = ev.get("text") or self.text_buf
                self.text_buf = ""
                if said:
                    print(f"  bot: {said}")
                    if self.provider == "eleven":
                        await self.speak_eleven(said)

            elif t.endswith("audio_transcript.done"):
                print(f"  bot: {ev.get('transcript','')}")

            elif t.endswith("input_audio_transcription.completed"):
                print(f"  you: {ev.get('transcript','')}")

            elif t == "response.function_call_arguments.done":
                try:
                    args = json.loads(ev.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                await self.run_tool(ev.get("name"), args, ev.get("call_id"))

            elif t == "response.done":
                await self.push_audio(force=True)
                self.active = False
                status = ev.get("response", {}).get("status")
                outputs = ev.get("response", {}).get("output", [])
                only_tools = bool(outputs) and all(
                    o.get("type") == "function_call" for o in outputs)

                if status == "failed":
                    print("  response failed:",
                          ev.get("response", {}).get("status_details"))
                    self.want_followup = False
                    self.turn_done.set()

                # A response that was nothing but tool calls is the model
                # waiting on results, so it needs a follow-up to say anything.
                # One that already spoke does not — its tool results just sit
                # in the conversation for next time. Asking anyway is what made
                # it tack a second sentence onto every reply.
                elif (only_tools or self.want_followup) and self.tool_rounds < 4:
                    self.want_followup = False
                    self.tool_rounds += 1
                    await self.create_response()
                else:
                    self.turn_done.set()

            elif t == "response.created":
                self.active = True

            elif t == "error":
                code = (ev.get("error") or {}).get("code")
                print("  API error:", json.dumps(ev.get("error", ev))[:400])
                # A rejected create means nothing new is in flight, and the
                # response already running will end the turn on its own.
                if code != "conversation_already_has_active_response":
                    self.turn_done.set()

    # -- one turn --------------------------------------------------
    async def turn_from_audio(self, wav: bytes):
        pcm, rate = wav_to_pcm(wav)
        pcm24 = resample(pcm, rate, API_RATE)
        await self.send({"type": "input_audio_buffer.append",
                         "audio": base64.b64encode(pcm24).decode()})
        await self.send({"type": "input_audio_buffer.commit"})
        self.tool_rounds = 0
        await self.create_response()
        await self._wait_turn()

    async def turn_from_text(self, text: str):
        await self.send({
            "type": "conversation.item.create",
            "item": {"role": "user", "type": "message",
                     "content": [{"type": "input_text", "text": text}]},
        })
        self.tool_rounds = 0
        await self.create_response()
        await self._wait_turn()

    async def _wait_turn(self, timeout=90):
        self.turn_done.clear()
        try:
            await asyncio.wait_for(self.turn_done.wait(), timeout)
        except asyncio.TimeoutError:
            print("  turn timed out")
        await self.push_audio(force=True)
        await asyncio.get_running_loop().run_in_executor(None, self.bot.wait_quiet)


# ------------------------------------------------------------------
async def run(args):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY")

    bot = Bot(args.bot)
    try:
        st = bot.status()
    except requests.RequestException as e:
        sys.exit(f"bot unreachable at {args.bot}: {e}")
    print("bot:", json.dumps(st))
    if not st.get("audio"):
        print("  warning: audio not ready on the bot, you will hear nothing")
    if not st.get("mic"):
        print("  warning: mic not ready, use --text or --push with typed input")
    if not st.get("camera"):
        print("  warning: camera not ready — /capture will 503 and take_photo "
              "will fail. Check the serial log for the 'Camera init failed' code.")

    if args.volume is not None:
        bot.volume(args.volume)

    sess = Session(bot, key, provider=args.voice)
    await sess.connect()
    pump = asyncio.create_task(sess.pump())

    try:
        if args.text:
            await sess.turn_from_text(args.text)
            return

        loop = asyncio.get_running_loop()

        vad = False if args.no_vad else await loop.run_in_executor(None, bot.probe_vad)
        if not vad:
            print("  warning: firmware has no /mic endpointing — falling back to "
                  "chunked recording, which goes deaf between chunks and chops "
                  "up what you say. Reflash ai_mini_bot.ino to fix this.")

        bot.beep(1047, 80)
        bot.face("neutral")
        room = await loop.run_in_executor(None, calibrate, bot)
        print(f"room: floor {room.floor:.0f}, trigger {room.onset:.0f}, "
              f"endpoint {room.endpoint:.0f} (raw ADC counts)")
        if args.push:
            print("ready — press Enter to talk.")
        else:
            print(f"listening — just talk, {args.silence_ms} ms of quiet ends "
                  f"your turn. ctrl-c to stop.")

        while True:
            if args.push:
                await loop.run_in_executor(None, input)
            else:
                # Never start listening while the speaker is still going, or
                # the bot triggers its own turn on the tail of its own voice.
                await loop.run_in_executor(None, bot.wait_quiet)
                await asyncio.sleep(0.25)
                await loop.run_in_executor(None, wait_for_voice, bot, room)

            bot.face("listening")
            # In voice mode /mic starts with your first word already underway,
            # so a short lead is enough to reject a false trigger. After Enter
            # you may still be drawing breath, so wait longer for you to begin.
            wav = await loop.run_in_executor(
                None, capture_turn, bot, room, vad, args.max_ms, args.silence_ms,
                5000 if args.push else LEAD_MS)

            if not args.push and not has_speech(wav, room):
                bot.face("neutral")
                continue        # false trigger — back to listening, no API call

            bot.face("thinking")
            await sess.turn_from_audio(trim_tail(wav, room))
            bot.face("neutral")

    except KeyboardInterrupt:
        pass
    finally:
        pump.cancel()
        try:
            bot.face("sleep")
            await sess.ws.close()
        except Exception:
            pass
        print("\nbye")


def check(args):
    bot = Bot(args.bot)
    print("status:", json.dumps(bot.status(), indent=2))
    print("beep:", bot.beep(880, 200))
    time.sleep(0.5)
    tone = (np.sin(2 * np.pi * 440 * np.arange(BOT_RATE) / BOT_RATE) * 8000)
    bot.play(tone.astype("<i2").tobytes())
    print("sent 1 s of 440 Hz to /say — you should hear it")
    print("recording 2 s...")
    wav = bot.record(2000)
    open("mic.wav", "wb").write(wav)
    pcm, rate = wav_to_pcm(wav)
    a = np.frombuffer(pcm, dtype="<i2")
    print(f"mic.wav saved: {len(a)} samples @ {rate} Hz, peak {int(np.abs(a).max())}")
    if np.abs(a).max() < 500:
        print("  peak is very low — check the MAX4466 gain pot and wiring")


def list_voices():
    if not ELEVEN_KEY:
        sys.exit("set ELEVENLABS_API_KEY")
    for v in eleven_voices():
        cat = v.get("category", "")
        mark = "  <-- cloned" if cat in ("cloned", "professional") else ""
        print(f"{v['voice_id']}  {v.get('name','?'):<24} {cat}{mark}")
    print("\nexport ELEVEN_VOICE_ID=<the id you want>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default=BOT_URL)
    ap.add_argument("--text", help="send one typed turn and exit")
    ap.add_argument("--push", action="store_true",
                    help="press Enter to talk instead of just talking")
    ap.add_argument("--max-ms", type=int, default=MAX_TURN_MS,
                    help="hard cap on one spoken turn")
    ap.add_argument("--silence-ms", type=int, default=SILENCE_MS,
                    help="quiet needed to end your turn")
    ap.add_argument("--no-vad", action="store_true",
                    help="force chunked recording instead of firmware endpointing")
    ap.add_argument("--volume", type=int)
    ap.add_argument("--check", action="store_true", help="hardware check only")
    ap.add_argument("--voice", choices=["openai", "eleven"],
                    default=os.environ.get("VOICE_PROVIDER", "openai"),
                    help="who speaks: the realtime model, or your ElevenLabs clone")
    ap.add_argument("--list-voices", action="store_true",
                    help="print your ElevenLabs voices and exit")
    ap.add_argument("--say", help="speak this through the chosen voice and exit")
    args = ap.parse_args()

    if args.list_voices:
        list_voices()
        return
    if args.say:
        bot = Bot(args.bot)
        if args.voice == "eleven":
            eleven_stream(args.say, bot.play)
        else:
            sys.exit("--say only works with --voice eleven "
                     "(the OpenAI voice needs a live session)")
        print("sent")
        return
    if args.check:
        check(args)
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()