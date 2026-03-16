from __future__ import annotations

import os

DEFAULT_MODE = "mock"
SUPPORTED_MODES = ("mock", "openai", "mega")

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
