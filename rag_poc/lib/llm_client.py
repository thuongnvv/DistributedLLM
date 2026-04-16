"""LLM client for rag_poc — lightweight version."""
from __future__ import annotations

import json
import os
import random
import time as _time
from datetime import datetime, timezone
from typing import Any

import requests

from lib.config import (
    GEMINI_BASE_URL, GEMINI_MODEL,
    GROQ_BASE_URL, GROQ_MODEL,
    MEGA_BASE_URL, MEGA_MODEL,
    OPENAI_BASE_URL, OPENAI_MAX_RETRIES,
    OPENAI_MODEL, OPENAI_RETRY_BASE_SECONDS,
    OPENAI_TEMPERATURE, OPENAI_TIMEOUT,
    OPENROUTER_BASE_URL, OPENROUTER_MODEL,
    SUPPORTED_MODES,
    MEGALLM_API_KEY,
)


class LLMClient:
    def __init__(self, mode: str = "mock", model: str | None = None, seed: int = 42):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of {SUPPORTED_MODES}")
        self.mode = mode
        if mode == "mega":
            self.model = model or MEGA_MODEL
            self.base_url = MEGA_BASE_URL
            self.api_key = os.getenv("MEGALLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        elif mode == "gemini":
            self.model = model or GEMINI_MODEL
            self.base_url = GEMINI_BASE_URL
            self.api_key = os.getenv("GOOGLE_API_KEY") or ""
        elif mode == "groq":
            self.model = model or GROQ_MODEL
            self.base_url = GROQ_BASE_URL
            self.api_key = os.getenv("GROQ_API_KEY") or ""
        elif mode == "openrouter":
            self.model = model or OPENROUTER_MODEL
            self.base_url = OPENROUTER_BASE_URL
            self.api_key = os.getenv("OPENROUTER_API_KEY") or ""
        else:
            self.model = model or OPENAI_MODEL
            self.base_url = OPENAI_BASE_URL
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MEGALLM_API_KEY") or ""
        self.rng = random.Random(seed)
        self._raw_events: list[dict[str, Any]] = []

    def reset_raw_events(self) -> None:
        self._raw_events = []

    def get_raw_events(self) -> list[dict[str, Any]]:
        return list(self._raw_events)

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        trace: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(f"{self.mode.upper()}_API_KEY not set. Use --mode mock or provide API key.")

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": OPENAI_TEMPERATURE,
        }
        if self.mode == "openai":
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retry_errors: list[dict[str, Any]] = []
        for attempt in range(OPENAI_MAX_RETRIES + 1):
            resp = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
            if resp.status_code < 400:
                break

            error_text = resp.text[:400].strip()
            retry_errors.append({"attempt": attempt + 1, "status_code": resp.status_code, "error": error_text})
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "").strip()
                sleep_sec = float(retry_after) if retry_after.isdigit() else OPENAI_RETRY_BASE_SECONDS * (2 ** attempt)
                sleep_sec += random.uniform(0.0, 0.4)
                _time.sleep(min(sleep_sec, 20.0))
                continue

            self._record_raw_event(trace, payload, "", None, resp.status_code, retry_errors, f"HTTP {resp.status_code}: {error_text}")
            resp.raise_for_status()

        if resp.status_code >= 400:
            error_text = resp.text[:400].strip()
            self._record_raw_event(trace, payload, "", None, resp.status_code, retry_errors, f"HTTP {resp.status_code}: {error_text}")
            resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI response has no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            text_parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenAI response has empty content: {data}")

        self._record_raw_event(trace, payload, content, data, resp.status_code, retry_errors, None)
        return content

    def _record_raw_event(
        self,
        trace: dict[str, Any] | None,
        request_payload: dict[str, Any],
        raw_content: str,
        response_json: dict[str, Any] | None,
        status_code: int | None,
        retry_errors: list[dict[str, Any]],
        error: str | None,
    ) -> None:
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "model": self.model,
            "base_url": self.base_url,
            "trace": trace or {},
            "status_code": status_code,
            "request": request_payload,
            "raw_content": raw_content,
            "response_json": response_json,
            "retry_errors": retry_errors,
            "error": error,
        }
        self._raw_events.append(event)

    def _mock_answer(self, persona, query, input_text, max_points):
        raise NotImplementedError("mock mode not supported — use openai or another LLM provider")

    def _is_abstain_point(self, text: str) -> bool:
        lowered = text.strip().lower()
        markers = {
            "out_of_scope",
            "unknown",
            "i don't know",
            "cannot determine",
            "not enough information",
        }
        if lowered in markers:
            return True
        if lowered.startswith("out_of_scope"):
            return True
        return False

    def _contains_clear_error(self, text: str) -> bool:
        lowered = text.lower()
        patterns = [
            "2+2=5",
            "always use string concatenation for sql",
            "disable authentication",
            "tcp is always faster than udp",
            "all a are b and some b are c implies some a are c",
            "set verify=false in production",
        ]
        return any(p in lowered for p in patterns)
