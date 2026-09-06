"""Configuration layer (§19).

Everything tunable lives here, read once from the environment. Secrets are
never defaulted to a real value and never logged.

Legacy names from mac_realtime.py (BOT_URL, ELEVEN_VOICE_ID, ELEVEN_MODEL) are
still accepted so an existing shell keeps working after the refactor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader. A dependency for this would not earn its keep.

    Real environment variables always win, so `FOO=1 python -m minibot` still
    overrides the file.
    """
    path = path or PROJECT_ROOT / ".env"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ.setdefault(key, val)


def _env(*names: str, default: str = "") -> str:
    """First name that is set wins, so legacy variables keep working."""
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


def _int(*names: str, default: int) -> int:
    try:
        return int(float(_env(*names, default=str(default))))
    except ValueError:
        return default


def _float(*names: str, default: float) -> float:
    try:
        return float(_env(*names, default=str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ServoLimits:
    """Mirrors the #defines in ai_mini_bot.ino. The firmware clamps too — this
    is the Mac-side copy so a bad tool call is rejected before it reaches the
    wire, not silently corrected on the far end (§22)."""

    pan_min: int = 15
    pan_max: int = 165
    tilt_min: int = 30
    tilt_max: int = 150
    pan_center: int = 90
    tilt_center: int = 90


@dataclass(frozen=True)
class CameraPolicyConfig:
    """Gemini Live accepts at most 1 FPS of JPEG, so these are all ceilings
    below a hard cap rather than target frame rates (§12)."""

    idle_fps: float = 0.0
    conversation_fps: float = 0.2
    visual_query_fps: float = 1.0
    tracking_fps: float = 0.1
    max_fps: float = 1.0


@dataclass(frozen=True)
class MemoryConfig:
    db_path: str = "minibot_memory.db"
    embedding_model: str = "minishlab/potion-base-8M"
    retrieval_limit: int = 6
    # Measured against the real model, 2026-09-03: keyword/topical queries
    # score ~0.56-0.64 against a matching memory ("espresso" -> "Aykhan likes
    # espresso." = 0.64); unrelated pairs stay below ~0.12. 0.25 sat above what
    # real topical queries actually produce and returned nothing. 0.18 clears
    # the noise floor while still passing genuine topical matches — see
    # memory/manager.py's module docstring for the full numbers.
    min_similarity: float = 0.18
    decay_rate: float = 0.02
    working_ttl_seconds: int = 900


def _bool(*names: str, default: bool) -> bool:
    v = _env(*names, default="").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


@dataclass(frozen=True)
class XConfig:
    """The robot's own X account (§ social layer).

    Four OAuth 1.0a credentials from developer.x.com. Absent any of them the
    whole feature is simply not registered — no tools, no startup check — so an
    unconfigured robot behaves exactly as it did before.

    The limits are local guards, not X's quotas. They bound the damage a
    runaway model can do; X's own free-tier cap is far higher and is enforced
    on their side regardless.
    """

    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    require_confirm: bool = True
    dry_run: bool = False
    max_per_hour: int = 5
    max_per_day: int = 20
    timeout: float = 15.0
    retries: int = 2

    @property
    def configured(self) -> bool:
        return all((self.api_key, self.api_secret,
                    self.access_token, self.access_secret))


@dataclass(frozen=True)
class Settings:
    # --- providers ---
    ai_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-realtime-2"
    openai_voice: str = "cedar"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-robotics-er-2-streaming-preview"

    # --- speech ---
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = "eleven_flash_v2_5"
    speech_provider: str = "elevenlabs"

    # --- robot transport ---
    esp32_base_url: str = "http://192.168.1.79"
    esp32_timeout: float = 8.0
    esp32_retries: int = 2
    esp32_min_interval: float = 0.02

    # --- audio ---
    # The computer's own microphone is the input. It is the one that is always
    # there, needs no firmware support for endpointing, and does not go deaf
    # for a WiFi round trip mid-sentence. AUDIO_INPUT=bot still switches to the
    # robot's MAX4466 for anyone who wants the robot to hear the room itself.
    audio_input: str = "mac"
    bot_rate: int = 16000
    silence_ms: int = 1000
    max_turn_ms: int = 12000
    lead_ms: int = 1500
    volume: int = 0
    # Voice-onset tuning. Raise the multipliers if the robot triggers on room
    # noise; lower them if it never hears you. Check with `--levels`.
    vad_onset_mult: float = 2.0
    vad_onset_margin: float = 35.0
    vad_endpoint_mult: float = 1.5
    vad_endpoint_margin: float = 18.0

    servo: ServoLimits = field(default_factory=ServoLimits)
    camera: CameraPolicyConfig = field(default_factory=CameraPolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    x: XConfig = field(default_factory=XConfig)

    log_level: str = "INFO"

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            ai_provider=_env("AI_PROVIDER", default="openai").lower(),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_model=_env("OPENAI_REALTIME_MODEL", "REALTIME_MODEL",
                              default="gpt-realtime-2"),
            openai_voice=_env("OPENAI_VOICE", default="cedar"),
            gemini_api_key=_env("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            gemini_model=_env("GEMINI_MODEL",
                              default="gemini-robotics-er-2-streaming-preview"),
            elevenlabs_api_key=_env("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", "ELEVEN_VOICE_ID"),
            elevenlabs_model=_env("ELEVENLABS_MODEL", "ELEVEN_MODEL",
                                  default="eleven_flash_v2_5"),
            speech_provider=_env("SPEECH_PROVIDER", default="elevenlabs").lower(),
            esp32_base_url=_env("ESP32_BASE_URL", "BOT_URL",
                                default="http://192.168.1.79").rstrip("/"),
            esp32_timeout=_float("ESP32_TIMEOUT", default=8.0),
            esp32_retries=_int("ESP32_RETRIES", default=2),
            esp32_min_interval=_float("ESP32_MIN_INTERVAL", default=0.02),
            audio_input=_env("AUDIO_INPUT", default="mac").lower(),
            silence_ms=_int("SILENCE_MS", default=1000),
            max_turn_ms=_int("MAX_TURN_MS", default=12000),
            lead_ms=_int("LEAD_MS", default=1500),
            volume=_int("ROBOT_VOLUME", default=0),
            vad_onset_mult=_float("VAD_ONSET_MULT", default=2.0),
            vad_onset_margin=_float("VAD_ONSET_MARGIN", default=35.0),
            vad_endpoint_mult=_float("VAD_ENDPOINT_MULT", default=1.5),
            vad_endpoint_margin=_float("VAD_ENDPOINT_MARGIN", default=18.0),
            servo=ServoLimits(
                pan_min=_int("SERVO_MIN_PAN", default=15),
                pan_max=_int("SERVO_MAX_PAN", default=165),
                tilt_min=_int("SERVO_MIN_TILT", default=30),
                tilt_max=_int("SERVO_MAX_TILT", default=150),
            ),
            camera=CameraPolicyConfig(
                idle_fps=_float("CAMERA_IDLE_FPS", default=0.0),
                conversation_fps=_float("CAMERA_CONVERSATION_FPS", default=0.2),
                visual_query_fps=_float("CAMERA_VISUAL_QUERY_FPS", default=1.0),
                tracking_fps=_float("CAMERA_TRACKING_FPS", default=0.1),
            ),
            memory=MemoryConfig(
                db_path=_env("MEMORY_DB_PATH", default="minibot_memory.db"),
                embedding_model=_env("EMBEDDING_MODEL",
                                     default="minishlab/potion-base-8M"),
                retrieval_limit=_int("MEMORY_RETRIEVAL_LIMIT", default=6),
                min_similarity=_float("MEMORY_MIN_SIMILARITY", default=0.18),
                decay_rate=_float("MEMORY_DECAY_RATE", default=0.02),
            ),
            x=XConfig(
                api_key=_env("X_API_KEY", "X_CONSUMER_KEY"),
                api_secret=_env("X_API_SECRET", "X_CONSUMER_SECRET"),
                access_token=_env("X_ACCESS_TOKEN"),
                access_secret=_env("X_ACCESS_TOKEN_SECRET", "X_ACCESS_SECRET"),
                require_confirm=_bool("X_REQUIRE_CONFIRM", default=True),
                dry_run=_bool("X_DRY_RUN", default=False),
                max_per_hour=_int("X_MAX_POSTS_PER_HOUR", default=5),
                max_per_day=_int("X_MAX_POSTS_PER_DAY", default=20),
                timeout=_float("X_TIMEOUT", default=15.0),
                retries=_int("X_RETRIES", default=2),
            ),
            log_level=_env("LOG_LEVEL", default="INFO").upper(),
        )
