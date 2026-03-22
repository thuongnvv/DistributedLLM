from __future__ import annotations

from datetime import datetime, timezone
import os
import random
import time
from typing import Any

try:
    import requests
except Exception as exc:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _REQUESTS_IMPORT_ERROR = exc
else:
    _REQUESTS_IMPORT_ERROR = None

from config import (
    MAX_POINTS_PER_ANSWER,
    MAX_USED_POINTS,
    MEGA_BASE_URL,
    MEGA_MODEL,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_RETRY_BASE_SECONDS,
    OPENAI_TEMPERATURE,
    OPENAI_TIMEOUT,
    SUPPORTED_MODES,
)
from prompts import (
    Persona,
    build_answer_prompt,
    build_grade_batch_prompt,
    build_grade_prompt,
    build_synthesize_prompt,
)
from protocol import GradeVote, Point, SynthesisDraft, WorkerAnswer, normalize_grade_vote
from utils import dedupe_keep_order, extract_first_json_object, generate_point_id, sha256_short


class LLMClient:
    def __init__(self, mode: str = "mock", model: str | None = None, seed: int = 42):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}. Expected one of {SUPPORTED_MODES}")

        self.mode = mode
        if mode == "mega":
            self.model = model or os.getenv("MEGA_MODEL", MEGA_MODEL)
            self.base_url = os.getenv("MEGA_BASE_URL", MEGA_BASE_URL).rstrip("/")
            self.api_key = os.getenv("MEGALLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.model = model or os.getenv("OPENAI_MODEL", OPENAI_MODEL)
            self.base_url = os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL).rstrip("/")
            self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MEGALLM_API_KEY")
        self.rng = random.Random(seed)
        self._raw_events: list[dict[str, Any]] = []

    def reset_raw_events(self) -> None:
        self._raw_events = []

    def get_raw_events(self) -> list[dict[str, Any]]:
        return list(self._raw_events)

    # ---------------- Public phase APIs ----------------
    def answer(
        self,
        persona: Persona,
        query: str,
        input_text: str | None = None,
        max_points: int | None = MAX_POINTS_PER_ANSWER,
    ) -> WorkerAnswer:
        if self.mode == "mock":
            point_texts = self._mock_answer(persona, query, input_text, max_points=max_points)
        else:
            point_texts = self._openai_answer(persona, query, input_text, max_points)

        point_texts = [pt[:160].strip() for pt in point_texts if pt and pt.strip()]
        point_texts = [pt for pt in point_texts if not self._is_abstain_point(pt)]
        point_texts = dedupe_keep_order(point_texts)

        if max_points is not None and max_points > 0:
            point_texts = point_texts[:max_points]
        points: list[Point] = []
        for idx, text in enumerate(point_texts):
            pid = generate_point_id(persona.node_id, idx, text)
            points.append(Point(point_id=pid, text=text))

        return WorkerAnswer(node_id=persona.node_id, points=points)

    def synthesize(
        self,
        persona: Persona,
        bundle: list[WorkerAnswer],
        query: str,
        input_text: str | None = None,
        max_used_points: int | None = MAX_USED_POINTS,
    ) -> SynthesisDraft:
        points_map = self._bundle_points_map(
            bundle,
            shuffle_seed=f"{persona.node_id}|{query}|{input_text or ''}",
        )
        if self.mode == "mock":
            synthesis_text, used_points = self._mock_synthesis(
                persona,
                query,
                points_map,
                max_used_points=max_used_points,
            )
        else:
            synthesis_text, used_points = self._openai_synthesis(
                persona=persona,
                query=query,
                input_text=input_text,
                points_map=points_map,
                max_used_points=max_used_points,
            )

        used_points = [pid for pid in dedupe_keep_order(used_points) if pid in points_map]
        if max_used_points is not None and max_used_points > 0:
            used_points = used_points[:max_used_points]

        if not synthesis_text.strip():
            synthesis_text = "UNKNOWN"

        if synthesis_text.strip().upper() != "UNKNOWN" and not used_points:
            synthesis_text = "UNKNOWN"

        return SynthesisDraft(
            node_id=persona.node_id,
            synthesis_text=synthesis_text.strip(),
            used_points=used_points,
        )

    def grade(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        target_id: str,
        target_draft_text: str,
        target_used_points: list[str],
        points_map: dict[str, str],
    ) -> GradeVote:
        if self.mode == "mock":
            raw_vote = self._mock_grade(
                persona=persona,
                query=query,
                input_text=input_text,
                target_id=target_id,
                target_draft_text=target_draft_text,
                target_used_points=target_used_points,
                points_map=points_map,
            )
        else:
            raw_vote = self._openai_grade(
                persona=persona,
                query=query,
                input_text=input_text,
                target_id=target_id,
                target_draft_text=target_draft_text,
                target_used_points=target_used_points,
                points_map=points_map,
            )

        return normalize_grade_vote(raw_vote, target_used_points)

    def grade_batch(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
    ) -> list[GradeVote]:
        if not targets:
            return []

        if self.mode == "mock":
            raw_votes = self._mock_grade_batch(
                persona=persona,
                query=query,
                input_text=input_text,
                targets=targets,
                points_map=points_map,
            )
        else:
            raw_votes = self._openai_grade_batch(
                persona=persona,
                query=query,
                input_text=input_text,
                targets=targets,
                points_map=points_map,
            )

        used_by_target = {target.node_id: list(target.used_points) for target in targets}
        allowed_targets = set(used_by_target.keys())
        normalized_votes: list[GradeVote] = []
        seen_targets: set[str] = set()

        for vote in raw_votes:
            target_id = vote.target_id
            if target_id not in allowed_targets or target_id in seen_targets:
                continue
            vote.grader_id = persona.node_id
            normalized = normalize_grade_vote(vote, used_by_target[target_id])
            normalized_votes.append(normalized)
            seen_targets.add(target_id)

        for target in targets:
            if target.node_id in seen_targets:
                continue
            normalized_votes.append(
                GradeVote(
                    grader_id=persona.node_id,
                    target_id=target.node_id,
                    valid="UNKNOWN",
                    agree_points=[],
                    reject_points=[],
                    unknown_points=list(target.used_points),
                    note="Missing target in batch response",
                )
            )

        return normalized_votes

    # ---------------- Mock mode ----------------
    def _mock_answer(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        max_points: int | None,
    ) -> list[str]:
        query_short = query.strip().replace("\n", " ")[:220]

        domain_templates: dict[str, list[str]] = {
            "coding_debugging": [
                "Reproduce issue with minimal failing test",
                "Inspect stack trace and failing branch",
                "Validate input assumptions and null handling",
                "Apply targeted patch with smallest surface area",
                "Add regression test for failing scenario",
                "Check edge cases and error handling paths",
            ],
            "databases_sql": [
                "Inspect query plan before and after changes",
                "Add selective index for WHERE/JOIN predicates",
                "Avoid full table scans on hot paths",
                "Review join cardinality and filter pushdown",
                "Use proper isolation level for data consistency",
                "Measure latency impact with representative data",
            ],
            "networking_systems": [
                "Map requirement to transport reliability needs",
                "Differentiate latency versus throughput bottlenecks",
                "Use retries with timeout and backoff strategy",
                "Validate cache semantics and freshness policy",
                "Monitor packet loss and tail latency metrics",
                "Prefer observable and reversible config changes",
            ],
            "math_logic": [
                "State assumptions before derivation",
                "Transform expression using valid algebraic rules",
                "Check logical implication direction carefully",
                "Verify result by substitution or contradiction",
                "Separate proven facts from unknown claims",
                "Return concise final result after validation",
            ],
            "security_defensive": [
                "Prefer parameterized queries over string concatenation",
                "Validate and sanitize untrusted input paths",
                "Apply least privilege for services and accounts",
                "Protect session/token lifecycle and rotation",
                "Log security events for anomaly detection",
                "Treat unknown risk as requiring hardening",
            ],
        }

        points = list(
            domain_templates.get(
                persona.domain,
                [
                    "State known constraints",
                    "Avoid unsupported assumptions",
                    "Mark uncertainty explicitly",
                    "Provide next verification step",
                ],
            )
        )
        if query_short:
            points.insert(0, f"Address query objective: {query_short[:80]}")
        if input_text:
            points.append(f"Ground in provided input_text (len={len(input_text)})")
        if max_points is not None and max_points > 0:
            return points[: min(max_points, len(points))]
        return points

    def _mock_synthesis(
        self,
        persona: Persona,
        query: str,
        points_map: dict[str, str],
        max_used_points: int | None,
    ) -> tuple[str, list[str]]:
        if not points_map:
            return "UNKNOWN", []

        ordered_ids = list(points_map.keys())
        limit = max_used_points if max_used_points is not None and max_used_points > 0 else len(ordered_ids)
        top = ordered_ids[:limit]
        if not top:
            top = list(points_map.keys())[:5]

        snippets = [points_map[pid] for pid in top[:4]]
        synthesis = (
            f"[{persona.domain}] Consolidated answer for '{query[:90]}': "
            + "; ".join(snippets)
        )
        return synthesis, top

    def _mock_grade(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        target_id: str,
        target_draft_text: str,
        target_used_points: list[str],
        points_map: dict[str, str],
    ) -> GradeVote:
        seed_material = f"{persona.node_id}|{target_id}|{query}|{sha256_short(target_draft_text)}"
        rng = random.Random(int(sha256_short(seed_material), 16))

        clear_error = self._contains_clear_error(target_draft_text)

        if clear_error:
            valid = "FAIL"
            note = "Clear contradiction/error detected"
        else:
            roll = rng.random()
            if roll < 0.10:
                valid = "FAIL"
                note = "Detected likely factual inconsistency"
            elif roll < 0.25:
                valid = "UNKNOWN"
                note = "Insufficient confidence for judgment"
            else:
                valid = "PASS"
                note = "No clear contradiction found"

        agree: list[str] = []
        reject: list[str] = []
        unknown: list[str] = []

        for pid in target_used_points:
            text = points_map.get(pid, "")
            if self._contains_clear_error(text):
                reject.append(pid)
                continue

            r = rng.random()
            if valid == "PASS":
                if r < 0.78:
                    agree.append(pid)
                elif r < 0.95:
                    unknown.append(pid)
                else:
                    reject.append(pid)
            elif valid == "FAIL":
                if r < 0.60:
                    reject.append(pid)
                elif r < 0.85:
                    unknown.append(pid)
                else:
                    agree.append(pid)
            else:
                if r < 0.15:
                    agree.append(pid)
                elif r < 0.85:
                    unknown.append(pid)
                else:
                    reject.append(pid)

        return GradeVote(
            grader_id=persona.node_id,
            target_id=target_id,
            valid=valid,  # type: ignore[arg-type]
            agree_points=agree,
            reject_points=reject,
            unknown_points=unknown,
            note=note,
        )

    def _mock_grade_batch(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
    ) -> list[GradeVote]:
        votes: list[GradeVote] = []
        for target in targets:
            votes.append(
                self._mock_grade(
                    persona=persona,
                    query=query,
                    input_text=input_text,
                    target_id=target.node_id,
                    target_draft_text=target.synthesis_text,
                    target_used_points=target.used_points,
                    points_map=points_map,
                )
            )
        return votes

    # ---------------- OpenAI mode ----------------
    def _openai_answer(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        max_points: int | None,
    ) -> list[str]:
        response_text = self._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=build_answer_prompt(query=query, input_text=input_text, max_points=max_points),
            trace={
                "phase": "stage1_answer",
                "node_id": persona.node_id,
            },
        )
        obj = extract_first_json_object(response_text)
        abstain_raw = obj.get("abstain", False)
        abstain = False
        if isinstance(abstain_raw, bool):
            abstain = abstain_raw
        elif isinstance(abstain_raw, str):
            abstain = abstain_raw.strip().lower() in {"true", "1", "yes"}
        if abstain:
            return []

        points_field = obj.get("points", [])
        point_texts = self._normalize_point_texts(points_field, max_points=max_points)
        return point_texts

    def _openai_synthesis(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        points_map: dict[str, str],
        max_used_points: int | None,
    ) -> tuple[str, list[str]]:
        synthesis_text, used_points = self._run_openai_synthesis_attempt(
            persona=persona,
            query=query,
            input_text=input_text,
            points_map=points_map,
            max_used_points=max_used_points,
            repair_feedback=None,
            trace_phase="stage2_synthesize",
        )
        if synthesis_text.upper() != "UNKNOWN" and not used_points:
            synthesis_text, used_points = self._run_openai_synthesis_attempt(
                persona=persona,
                query=query,
                input_text=input_text,
                points_map=points_map,
                max_used_points=max_used_points,
                repair_feedback=(
                    "Your previous output violated the contract: synthesis_text was not UNKNOWN "
                    "but used_points was empty or invalid. Return corrected JSON. If you cannot "
                    "support the answer with valid point_ids, return synthesis_text='UNKNOWN' and used_points=[]."
                ),
                trace_phase="stage2_synthesize_retry",
            )

        if synthesis_text.upper() != "UNKNOWN" and not used_points:
            return "UNKNOWN", []
        return synthesis_text, used_points

    def _openai_grade(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        target_id: str,
        target_draft_text: str,
        target_used_points: list[str],
        points_map: dict[str, str],
    ) -> GradeVote:
        response_text = self._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=build_grade_prompt(
                query=query,
                input_text=input_text,
                target_draft_text=target_draft_text,
                target_used_points=target_used_points,
                points_map=points_map,
            ),
            trace={
                "phase": "stage3_grade_single",
                "node_id": persona.node_id,
                "target_id": target_id,
            },
        )
        obj = extract_first_json_object(response_text)

        valid = str(obj.get("valid", "UNKNOWN")).upper()
        agree = obj.get("agree_points", [])
        reject = obj.get("reject_points", [])
        unknown = obj.get("unknown_points", [])
        note = str(obj.get("note", "")).strip()

        vote = GradeVote(
            grader_id=persona.node_id,
            target_id=target_id,
            valid=(valid if valid in {"PASS", "FAIL", "UNKNOWN"} else "UNKNOWN"),  # type: ignore[arg-type]
            agree_points=[str(x) for x in agree] if isinstance(agree, list) else [],
            reject_points=[str(x) for x in reject] if isinstance(reject, list) else [],
            unknown_points=[str(x) for x in unknown] if isinstance(unknown, list) else [],
            note=note,
        )
        return vote

    def _openai_grade_batch(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
    ) -> list[GradeVote]:
        target_payload: list[dict[str, Any]] = []
        for target in targets:
            target_payload.append(
                {
                    "target_id": target.node_id,
                    "target_draft": target.synthesis_text,
                    "target_used_points": list(target.used_points),
                }
            )

        response_text = self._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=build_grade_batch_prompt(
                query=query,
                input_text=input_text,
                targets=target_payload,
                points_map=points_map,
            ),
            trace={
                "phase": "stage3_grade_batch",
                "node_id": persona.node_id,
                "target_count": len(target_payload),
            },
        )
        obj = extract_first_json_object(response_text)

        votes_field = obj.get("votes", [])
        normalized_items: list[dict[str, Any]] = []
        if isinstance(votes_field, list):
            normalized_items = [entry for entry in votes_field if isinstance(entry, dict)]
        elif isinstance(votes_field, dict):
            for target_id, detail in votes_field.items():
                if not isinstance(detail, dict):
                    continue
                item = dict(detail)
                item["target_id"] = str(target_id)
                normalized_items.append(item)

        votes: list[GradeVote] = []
        for entry in normalized_items:
            target_id = str(entry.get("target_id", "")).strip()
            if not target_id:
                continue
            valid = str(entry.get("valid", "UNKNOWN")).upper()
            agree = entry.get("agree_points", [])
            reject = entry.get("reject_points", [])
            unknown = entry.get("unknown_points", [])
            note = str(entry.get("note", "")).strip()

            votes.append(
                GradeVote(
                    grader_id=persona.node_id,
                    target_id=target_id,
                    valid=(valid if valid in {"PASS", "FAIL", "UNKNOWN"} else "UNKNOWN"),  # type: ignore[arg-type]
                    agree_points=[str(x) for x in agree] if isinstance(agree, list) else [],
                    reject_points=[str(x) for x in reject] if isinstance(reject, list) else [],
                    unknown_points=[str(x) for x in unknown] if isinstance(unknown, list) else [],
                    note=note,
                )
            )

        return votes

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        trace: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY (or MEGALLM_API_KEY) not set. Use --mode mock or provide key.")
        if requests is None:
            raise RuntimeError(f"requests is required for openai mode: {_REQUESTS_IMPORT_ERROR}")

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": OPENAI_TEMPERATURE,
        }
        # Not all OpenAI-compatible providers implement response_format.
        if self.mode == "openai":
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = None
        last_error_text = ""
        retry_errors: list[dict[str, Any]] = []
        for attempt in range(OPENAI_MAX_RETRIES + 1):
            response = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
            status = response.status_code

            if status < 400:
                break

            error_preview = response.text[:400].strip()
            last_error_text = error_preview
            retry_errors.append(
                {
                    "attempt": attempt + 1,
                    "status_code": status,
                    "error_preview": error_preview,
                }
            )
            retryable = status == 429 or status >= 500
            if not retryable or attempt >= OPENAI_MAX_RETRIES:
                self._record_raw_event(
                    trace=trace,
                    request_payload=payload,
                    raw_content="",
                    response_json=None,
                    status_code=status,
                    retry_errors=retry_errors,
                    error=f"HTTP {status}: {error_preview}",
                )
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after.isdigit():
                sleep_sec = float(retry_after)
            else:
                sleep_sec = OPENAI_RETRY_BASE_SECONDS * (2**attempt)
            sleep_sec += random.uniform(0.0, 0.4)
            time.sleep(min(sleep_sec, 20.0))

        if response is None:
            raise RuntimeError("No response returned from provider")

        if response.status_code >= 400:
            self._record_raw_event(
                trace=trace,
                request_payload=payload,
                raw_content="",
                response_json=None,
                status_code=response.status_code,
                retry_errors=retry_errors,
                error=f"HTTP {response.status_code}: {response.text[:400].strip()}",
            )
            response.raise_for_status()

        data = response.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI response has no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            hint = f" last_error={last_error_text}" if last_error_text else ""
            raise RuntimeError(f"OpenAI response has empty content: {data}{hint}")

        self._record_raw_event(
            trace=trace,
            request_payload=payload,
            raw_content=content,
            response_json=data,
            status_code=response.status_code,
            retry_errors=retry_errors,
            error=None,
        )
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

    # ---------------- Helpers ----------------
    def _bundle_points_map(
        self,
        bundle: list[WorkerAnswer],
        shuffle_seed: str | None = None,
    ) -> dict[str, str]:
        items: list[tuple[str, str]] = []
        for answer in bundle:
            for p in answer.points:
                items.append((p.point_id, p.text))

        if shuffle_seed:
            rng = random.Random(self._stable_seed(shuffle_seed))
            rng.shuffle(items)

        points_map: dict[str, str] = {}
        for point_id, text in items:
            points_map[point_id] = text
        return points_map

    def _run_openai_synthesis_attempt(
        self,
        persona: Persona,
        query: str,
        input_text: str | None,
        points_map: dict[str, str],
        max_used_points: int | None,
        repair_feedback: str | None,
        trace_phase: str,
    ) -> tuple[str, list[str]]:
        response_text = self._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=build_synthesize_prompt(
                query=query,
                input_text=input_text,
                points_map=points_map,
                max_used_points=max_used_points,
                repair_feedback=repair_feedback,
            ),
            trace={
                "phase": trace_phase,
                "node_id": persona.node_id,
            },
        )
        obj = extract_first_json_object(response_text)

        synthesis_text = str(obj.get("synthesis_text", "UNKNOWN")).strip() or "UNKNOWN"
        used_points_raw = obj.get("used_points", [])
        if not isinstance(used_points_raw, list):
            used_points_raw = []
        used_points = [pid for pid in dedupe_keep_order(str(x) for x in used_points_raw) if pid in points_map]
        if max_used_points is not None and max_used_points > 0:
            used_points = used_points[:max_used_points]
        return synthesis_text, used_points

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

    def _stable_seed(self, seed_material: str) -> int:
        return int(sha256_short(seed_material), 16)

    def _normalize_point_texts(
        self,
        points_field: Any,
        max_points: int | None,
    ) -> list[str]:
        texts: list[str] = []
        if isinstance(points_field, list):
            for entry in points_field:
                if isinstance(entry, dict):
                    text = entry.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                elif isinstance(entry, str) and entry.strip():
                    texts.append(entry.strip())

        texts = dedupe_keep_order(t[:160] for t in texts if t and t.strip())
        if max_points is not None and max_points > 0:
            return texts[:max_points]
        return texts
