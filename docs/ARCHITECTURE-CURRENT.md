# Current architecture (pre-migration baseline)

Written during Phase 1 by inspection of the working tree. This is the contract
the refactor must not break.

## Files

| File | Lines | Role |
|---|---|---|
| `ai_mini_bot.ino` | ~1150 | ESP32-S3 firmware. The whole robot body. |
| `control_page.h` | 335 | `PROGMEM` HTML control page served at `/`. |
| `mac_realtime.py` | ~830 | The entire Mac brain, single file. |
| `mic.wav` | — | Throwaway artifact written by `--check`. |

No git repository. No tests. No dependency manifest. `.venv` is Python 3.11
with `numpy`, `requests`, `websockets`.

## Layer 1 — ESP32 firmware (`ai_mini_bot.ino`)

Board: XIAO ESP32S3 Sense. Arduino core 3.x, OPI PSRAM required.

### HTTP API (the contract)

Registered in `setupRoutes()`. **These endpoints must keep working.**

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/` | HTML | `control_page.h` via `send_P` |
| GET | `/capture` | `image/jpeg` | Sets `capturing` flag for the duration |
| GET | `/look?pan=&tilt=` | `text/plain` | 503 if servos not ready |
| GET | `/look/center` | `text/plain` | |
| GET | `/look/status` | JSON | `{pan,tilt,ready}` |
| GET | `/status` | JSON | Full health, see below |
| GET | `/set?e=<emotion>` | `text/plain` | |
| GET | `/<emotion>` | `text/plain` | 15 path aliases |
| POST | `/say` | JSON | Raw body: int16 LE mono @ 16 kHz |
| GET | `/mic?ms=&silence=&thresh=&lead=` | `audio/wav` | Blocking; endpointing added this session |
| GET | `/level` | JSON | `{rms,peak}` — 20 ms window |
| GET | `/beep?f=&ms=` | `text/plain` | |
| GET | `/volume?v=` | `text/plain` | 0–15 |
| GET | `/stop` | `text/plain` | Flush speech queue |

`/status` fields: `camera, oled, servos, audio, mic, speaking, listening,
volume, emotion, pan, tilt, rssi, heap, psram`.

### CRITICAL: the web server is single-threaded

`server.handleClient()` runs inside `loop()`. **One request at a time, and any
long handler stalls every other endpoint.** Consequences the Mac must respect:

- `/mic` blocks the device for up to **15 s**. During that window `/status`,
  `/set`, `/look` and `/capture` are all unreachable.
- Face animation freezes for the whole of any blocking handler — `renderFace()`
  also lives in `loop()`.
- Therefore the Mac side must **serialize** ESP32 requests. A parallel HTTP
  client would queue up on the socket and time out, not gain throughput.

This single fact is the strongest justification for a central action scheduler.

### Electrical / stability workarounds — DO NOT REMOVE

The board has **no decoupling capacitors**. Every mitigation is in software:

| Mitigation | Location | Why |
|---|---|---|
| `Wire.setClock(100000)` | `setup()` | 100 kHz I2C, not 400 kHz |
| `WiFi.setTxPower(WIFI_POWER_11dBm)` | `setup()` | Lower TX current draw |
| `display.setContrast(80)` | `setup()` | Reduced OLED current |
| Camera init **before** `Wire.begin()` | `setup()` | Ordering matters |
| Servo motion deferred during capture | `setHead()` | Waits up to 500 ms on `capturing` |
| `vTaskDelay(1)` every 16 mic frames | `recordMic()` | Feeds the 5 s task watchdog |
| Audio clamp at ±full scale | `playClip()` | Prevents wrap-around on gain > unity |
| PSRAM-first allocation | `psAlloc`, `handleMic` | Audio and mic buffers off internal heap |

`setHead()` busy-waits while `capturing` is true — this is the camera/servo
mutual exclusion the new scheduler must preserve and mirror on the Mac side.

### Servo constraints

PCA9685 @ `0x40`. **`SERVO_CH_PAN 3`, `SERVO_CH_TILT 4`.**

> Note: the file header comment says "CH0 = pan, CH2 = tilt". That comment is
> stale and contradicts the `#define`s. The defines are authoritative.

| Limit | Value |
|---|---|
| `PAN_MIN` / `PAN_MAX` | 15 / 165 |
| `TILT_MIN` / `TILT_MAX` | 30 / 150 |
| `PAN_CENTER` / `TILT_CENTER` | 90 / 90 |
| `SERVO_US_MIN` / `MAX` | 500 / 2500 µs |

Pan/tilt are clamped twice: `setHead()` clamps to the min/max pair, then
`writeServoAngle()` clamps again to 0–180 before the µs conversion. Lower pan
looks left; lower tilt looks up.

### Expressions — the authoritative 15

Enum order in `ai_mini_bot.ino`, names from `emotionName()` / `parseEmotion()`:

```
neutral  happy   sad     angry   surprised
thinking listening talking sleep searching
loading  scanning  wifi   memory  saving
```

`renderFace()` overrides the resting faces only: if the current emotion is
`neutral`, `listening` or `talking`, it is replaced by `talking` when
`speaking` is set and `listening` when `listening` is set. An explicit emotion
is left alone. Blinking only runs for those same three resting faces.

**There is no `curious`, `confused`, `content` or `sleepy` face.** The
`ExpressionManager` must map semantic names onto this fixed set.

### Audio out

MAX98357 on I2S1. 16 kHz mono in, duplicated to stereo. FreeRTOS queue depth
24 (`clipQueue`), drained by `audioTask` pinned to core 1. Fades are applied
only at the true start and end of an utterance so chunked TTS does not warble
at every seam — `playClip(c, fadeIn, fadeOut)`. Volume 0–15, 10 = unity.

## Layer 2 — Mac brain (`mac_realtime.py`)

One 830-line module. Everything is in it.

| Concern | Symbols |
|---|---|
| Config | Module-level constants + `os.environ` reads |
| ESP32 transport | `class Bot` — 13 methods, raw `requests.Session` |
| Turn taking / VAD | `Room`, `calibrate`, `wait_for_voice`, `has_speech`, `trim_tail`, `record_chunked`, `capture_turn` |
| Audio DSP | `resample`, `wav_to_pcm`, `pcm_to_wav`, `frame_rms` |
| AI provider | `class Session` — OpenAI Realtime WebSocket, `build_session`, `INSTRUCTIONS`, `TOOLS` |
| Tool dispatch | `Session.run_tool` — 3 tools: `set_face`, `look_at`, `take_photo` |
| Speech | `eleven_stream`, `eleven_voices`, `Session.speak_eleven` |
| Entrypoint | `run`, `check`, `list_voices`, `main` |

### What works and must be preserved

- **ElevenLabs streaming.** `pcm_16000` output lands exactly on the firmware's
  16 kHz, so nothing is resampled on that path. Chunks are pushed to `/say` as
  they arrive, so the robot starts speaking before synthesis finishes.
- **Response lifecycle.** One response in flight at a time (`create_response`
  guard, `active` flag). Tool outputs are submitted as they complete and a
  single follow-up response is issued from `response.done`.
- **Firmware VAD with fallback.** `probe_vad()` detects endpointing support by
  timing a deliberately-unreachable-threshold recording; falls back to chunked
  capture on older firmware.
- **Echo avoidance.** `wait_quiet()` polls `/status.speaking` before listening,
  plus a 250 ms settle. The mic and speaker share a board with no AEC.
- **Adaptive noise floor.** `Room` keeps a rolling median and derives a higher
  onset threshold than endpoint threshold, so a sentence does not end on its
  own quiet consonants.

### Known limitations at baseline

- Turn-based, not duplex. The bot cannot listen while speaking.
- `take_photo` currently fails: `/status` reports `camera: false` on the
  device, so `/capture` returns 503.
- No memory of any kind between runs.
- No tracking; the model must issue every `look_at` itself.
- Conversation history grows unbounded inside the provider session.
- The user has been running system Python 3.9, not `.venv` (3.11).
