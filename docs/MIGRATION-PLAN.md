# Migration plan

Companion to `ARCHITECTURE-CURRENT.md`. Symbol-level mapping from the current
single-file brain to the target architecture.

## Verified external facts

Checked against Google's docs rather than assumed, because the whole AI layer
depends on them:

| Fact | Value | Consequence |
|---|---|---|
| Model ID | `gemini-robotics-er-2-streaming-preview` | Confirmed live in preview |
| SDK | `google-genai`, `client.aio.live.connect(model=..., config=...)` | Async context manager |
| Audio in | raw PCM **16 kHz** 16-bit mono, `audio/pcm;rate=16000` | **Same rate as the firmware.** The 24 kHz resample on the input path disappears entirely. |
| Video in | JPEG, **≤ 1 FPS** | Hard ceiling. The camera policy is a rate limiter, not a streamer. |
| Output | `response_modalities=["TEXT"]` | Matches the ElevenLabs plan exactly |
| Tools | function declarations support `"behavior": "BLOCKING"` | The model *waits* for the robot to finish the action |
| Interruption | heartbeats act as user turns and interrupt generation | Maps onto barge-in |

`behavior: "BLOCKING"` is the important one. It lets a physical action's real
completion — as reported by the scheduler — gate the model's next token, which
is exactly the spec's "never claim a physical action succeeded until the action
result confirms success."

## KEEP — untouched

- **All of `ai_mini_bot.ino`.** No firmware changes are required by this
  refactor. Every endpoint, servo limit, expression, and stability workaround
  stays exactly as it is.
- **`control_page.h`.** Unaffected.
- The ESP32 HTTP contract, byte for byte.

## KEEP — moved verbatim, not rewritten

These are working, tuned, and hardware-specific. They get relocated behind an
interface with their logic intact:

| Current symbol | Destination |
|---|---|
| `Bot` (all 13 methods) | `robot/esp32_client.py` → `Esp32Client` |
| `resample`, `wav_to_pcm`, `pcm_to_wav`, `frame_rms` | `audio/dsp.py` |
| `Room`, `calibrate`, `wait_for_voice` | `audio/vad.py` |
| `has_speech`, `trim_tail`, `record_chunked`, `capture_turn` | `audio/capture.py` |
| `eleven_stream`, `eleven_voices`, `ELEVEN_*` | `speech/elevenlabs_provider.py` |
| `INSTRUCTIONS` | `agent/prompts.py` (rewritten for the robot persona, §27) |
| `EMOTIONS` | `behavior/expressions.py` as the authoritative ID set |
| `check`, `list_voices` | `cli.py` |

## REFACTOR — same behavior, new seams

| Current | Becomes | Why |
|---|---|---|
| `Session` (OpenAI WS) | `ai/openai_provider.py` → `OpenAIRealtimeProvider(AIProvider)` | Kept switchable via `AI_PROVIDER=openai` during migration, per §24 |
| `Session.run_tool` | `agent/tool_registry.py` + `robot/scheduler.py` | Dispatch split from execution |
| `Bot.look` callers | `RobotActionScheduler.look_at()` | Nothing calls the transport directly any more |
| Module-level constants | `config/settings.py` + `.env` | §19 |
| `print()` diagnostics | `logging/logger.py` structured events | §20 |
| `run()` loop | `agent/robot_agent.py` | Orchestrator |

## ADD — new subsystems

`ai/provider.py` (`AIProvider` ABC), `ai/gemini_provider.py`,
`agent/{robot_agent,tool_registry,conversation}.py`,
`behavior/{controller,state_machine,expressions}.py`,
`robot/{hardware,scheduler}.py`,
`vision/{local_vision,tracking,camera_policy}.py`,
`memory/{manager,store,embeddings,extractor,retriever,consolidator}.py`,
`speech/{provider,elevenlabs_provider}.py`, `events/bus.py`,
`config/settings.py`, `logging/logger.py`, `tests/`.

Added since: `social/{x_client,account}.py` — the robot's own X account. Not in
the original §2 tool set; see `X-ACCOUNT.md`. It is the first capability that
reaches outside the room, so it carries a confirmation step, rate limits and a
dry-run mode on top of the usual validate-before-the-wire posture.

## Tool surface: 3 → 20

Current tools are `set_face`, `look_at`, `take_photo`. The target set from §2
adds motion primitives (`nod`, `shake_head`, `tilt_head`, `center_head`),
perception (`track_person`, `stop_tracking`, `inspect_scene`, `capture_scene`),
lifecycle (`idle`, `wake`, `sleep`, `blink`, `wink`), speech (`speak`), and
memory (`remember`, `recall`, `forget`).

Every one is semantic. None expose PCA9685, PWM, I2C or GPIO.

## Safety mapping — firmware rule to Mac-side enforcement

The scheduler mirrors what the firmware already enforces, so a bad tool call is
rejected *before* it reaches the wire:

| Firmware rule | Scheduler enforcement |
|---|---|
| `PAN 15–165`, `TILT 30–150` | Clamp/reject in `validate()`; `pan=10000` never leaves the Mac |
| `setHead()` waits on `capturing` | Camera lock; servo commands queue behind an in-flight capture |
| One HTTP request at a time | Single serialized worker, never parallel |
| `/mic` blocks the device ≤ 15 s | Scheduler treats recording as an exclusive whole-device lease |
| 500 ms capture wait | Settle delay after movement before capture (§13 sequence) |
| Speech queue depth 24 | Rate limit on `/say` chunk submission |

The LLM is treated as an untrusted source of *requested* actions throughout.

## Open architectural tension: continuous audio

Gemini Live wants a continuous 16 kHz stream and interrupts on user speech.
The firmware physically cannot provide that — `/mic` blocks the core, there is
no ADC-DMA path, and the mic and speaker share a board with no echo
cancellation. Barge-in while the robot is speaking is impossible through the
bot's own microphone.

This is resolved by configuration rather than a firmware rewrite; see the
`AUDIO_INPUT` setting. The bot mic path is preserved either way.

## Decisions taken

| Question | Decision | Consequence |
|---|---|---|
| Version control | **Skipped** | No git. Mitigated by building `minibot/` alongside `mac_realtime.py` and deleting nothing — the old entrypoint is the rollback path and is covered by equivalence tests. |
| Audio input | **Bot mic, turn-based bursts** | No firmware change. Each finished utterance is sent to the Live session as one burst. **No barge-in** — the robot cannot hear while speaking. `AUDIO_INPUT=bot`. |
| Local stack | **model2vec + MediaPipe** | ~100 MB, no PyTorch. `EmbeddingProvider` keeps the model swappable via `EMBEDDING_MODEL`. |

## Status

- **Phase 1 — done.** These documents.
- **Phase 2 — done.** `AIProvider`, `RobotHardware`, `SpeechProvider`, `EventBus`.
- **Phase 3 — done.** Existing implementation moved behind them. Byte-equivalence
  tests against the original functions.
- **Phase 4 — done.** `GeminiRoboticsProvider` on the Live API, switchable by
  `AI_PROVIDER`. §27 persona applied to both providers. **103 tests green.**
- Phases 5–12 pending.
- **X account — done.** `social/`, four tools, confirmation gate. **205 tests
  green.** The OAuth 1.0a signer is checked against `oauthlib` for identical
  signatures across all five endpoints rather than only against itself.

### Phase 4 notes

Built against the installed SDK (`google-genai` 2.21) by introspection, not
from memory — `Behavior.BLOCKING`, `RealtimeInputConfig`, `ActivityStart/End`
and `LiveServerMessage` shapes were all confirmed before use.

Three decisions worth recording:

- **Automatic activity detection is disabled.** The firmware already endpoints
  each utterance via `/mic?silence=`, so bursts arrive pre-trimmed. Server-side
  VAD would re-segment work the robot already did. Turns are bracketed
  explicitly with `activity_start` / `activity_end`.
- **All tools are `BLOCKING`.** The model waits for the action to actually
  finish, which is what makes §27's "never claim success until confirmed"
  enforceable rather than merely requested.
- **Frames are throttled inside the provider** to the documented 1 FPS, so a
  camera-policy bug in Phase 10 cannot breach the API contract.

Context-window compression (sliding window) is enabled: a desk robot runs for
hours and the session would otherwise walk into the context limit mid-chat.

### Live-session findings (2026-09-02)

Three defects that only a real session exposed. All are covered by regression
tests now.

**The reply is not where the SDK says it is.** This model returns its answer in
`server_content.output_transcription.text`, not as text parts on `model_turn`,
so `LiveServerMessage.text` is *always* `None`. Reading only `msg.text`
discarded every response: tool calls fired, the head moved, the robot went
silent. Both fields are read now, transcription first.

**`turn_complete` fires at the end of the tool-call turn**, before the model
has generated anything. Ending the turn there returned an empty reply. The
provider now waits for the follow-up turn after submitting tool results,
bounded by `followup_timeout` so a purely physical command that never produces
speech cannot hang.

**A tool cannot return an image.** `FunctionResponse.parts` with inline binary
raises `Object of type bytes is not JSON serializable` —
`send_tool_response` skips the `model_dump(mode="json")` step that
base64-encodes bytes, unlike `send_client_content`. And `send_client_content`
*interrupts* the model, at `turn_complete` either True or False. Camera frames
therefore go out as realtime video, which neither mis-serializes nor
interrupts. Revisit if the SDK fixes the tool-response path.

Verified working end to end: asked what was on the desk, the robot set
`searching`, tilted down, captured a frame, and described the actual contents.

## Phase order

Per §25, with a build/test gate after each. Phases 2–3 must be behavior-neutral.

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | These documents | — |
| 2 | Interfaces only: `AIProvider`, `RobotHardware`, `SpeechProvider`, `EventBus` | imports clean |
| 3 | Existing code moved behind them; OpenAI path still runs | `--check` + a live turn behave as before |
| 4 | `GeminiRoboticsProvider` | Provider switchable by env |
| 5 | Semantic memory + local embeddings | Retrieval tests |
| 6 | Episodic memory, extraction, consolidation | Dedup/decay tests |
| 7 | `RobotActionScheduler` + central validation | Bounds/exclusion tests |
| 8 | `BehaviorController` + state machine | Transition tests |
| 9 | Local CV + tracking loop | Tracking runs without model calls |
| 10 | Adaptive camera policy | ≤ 1 FPS enforced |
| 11 | Active perception loop | Move→settle→lock→capture ordering |
| 12 | Tests, telemetry, polish | Full suite green |
