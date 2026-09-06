"""Shared test setup.

Isolates the suite from the developer's environment. Without this, creating a
real .env made three tests fail: Settings.load() reads it, load_dotenv()
mutates os.environ via setdefault, and a monkeypatched variable then loses to
whatever happens to be on this particular machine. Test results must not depend
on local config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot import config  # noqa: E402

# Every variable Settings.load() consults, legacy aliases included.
CONFIG_ENV_VARS = [
    "AI_PROVIDER",
    "OPENAI_API_KEY", "OPENAI_REALTIME_MODEL", "REALTIME_MODEL", "OPENAI_VOICE",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
    "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "ELEVEN_VOICE_ID",
    "ELEVENLABS_MODEL", "ELEVEN_MODEL", "SPEECH_PROVIDER",
    "ESP32_BASE_URL", "BOT_URL", "ESP32_TIMEOUT", "ESP32_RETRIES",
    "ESP32_MIN_INTERVAL",
    "AUDIO_INPUT", "SILENCE_MS", "MAX_TURN_MS", "LEAD_MS", "ROBOT_VOLUME",
    "SERVO_MIN_PAN", "SERVO_MAX_PAN", "SERVO_MIN_TILT", "SERVO_MAX_TILT",
    "CAMERA_IDLE_FPS", "CAMERA_CONVERSATION_FPS", "CAMERA_VISUAL_QUERY_FPS",
    "CAMERA_TRACKING_FPS",
    "MEMORY_DB_PATH", "EMBEDDING_MODEL", "MEMORY_RETRIEVAL_LIMIT",
    "MEMORY_MIN_SIMILARITY", "MEMORY_DECAY_RATE",
    "X_API_KEY", "X_CONSUMER_KEY", "X_API_SECRET", "X_CONSUMER_SECRET",
    "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET", "X_ACCESS_SECRET",
    "X_REQUIRE_CONFIRM", "X_DRY_RUN", "X_MAX_POSTS_PER_HOUR",
    "X_MAX_POSTS_PER_DAY", "X_TIMEOUT", "X_RETRIES",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def isolate_config_env(monkeypatch):
    """Neutralize .env and any inherited shell config for every test."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
