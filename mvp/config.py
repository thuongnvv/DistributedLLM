from __future__ import annotations

import os
from pathlib import Path

# Auto-load .env from same directory
_env = Path(__file__).parent / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)

DEFAULT_MODE = "mock"
SUPPORTED_MODES = ("mock", "openai", "mega", "gemini", "groq", "openrouter")

DEFAULT_K = 5
DEFAULT_TAU_FAIL = 2
DEFAULT_SEED = int(os.getenv("MVP_SEED", "42"))
DEFAULT_LOG_ROOT = "logs"


def _optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None

    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


MAX_POINTS_PER_ANSWER = _optional_positive_int("MAX_POINTS_PER_ANSWER")
MAX_USED_POINTS = _optional_positive_int("MAX_USED_POINTS")
if MAX_USED_POINTS is None:
    MAX_USED_POINTS = 10

WIN_BONUS = 5.0
FAIL_PENALTY = 2.0
POINT_SCORE_WEIGHT = 1.0

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "45"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
OPENAI_RETRY_BASE_SECONDS = float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1.0"))

MEGA_BASE_URL = os.getenv("MEGA_BASE_URL", "https://ai.megallm.io/v1").rstrip("/")
MEGA_MODEL = os.getenv("MEGA_MODEL", "openai-gpt-oss-120b")

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
