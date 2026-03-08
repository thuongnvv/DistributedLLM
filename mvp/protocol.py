from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from utils import dedupe_keep_order

ValidLabel = Literal["PASS", "FAIL", "UNKNOWN"]


@dataclass
class Point:
    point_id: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"point_id": self.point_id, "text": self.text}


@dataclass
class WorkerAnswer:
    node_id: str
    points: list[Point] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class SynthesisDraft:
    node_id: str
    synthesis_text: str
    used_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "synthesis_text": self.synthesis_text,
            "used_points": list(self.used_points),
        }


@dataclass
class GradeVote:
    grader_id: str
    target_id: str
    valid: ValidLabel
    agree_points: list[str] = field(default_factory=list)
    reject_points: list[str] = field(default_factory=list)
    unknown_points: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "grader_id": self.grader_id,
            "target_id": self.target_id,
            "valid": self.valid,
            "agree_points": list(self.agree_points),
            "reject_points": list(self.reject_points),
            "unknown_points": list(self.unknown_points),
        }
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class FinalizationOutput:
    winner: str
    final_answer: str
    metrics: dict[str, dict[str, int]]
    reputation_updates: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "final_answer": self.final_answer,
            "metrics": self.metrics,
            "reputation_updates": self.reputation_updates,
        }


@dataclass
class PersonaMeta:
    node_id: str
    domain: str
    style: str
    scope: str
    system_prompt: str
    rep_snapshot: float = 0.0


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def validate_worker_answer(answer: WorkerAnswer, max_points: int = 8) -> None:
    answer.node_id = _require_nonempty_str(answer.node_id, "WorkerAnswer.node_id")

    if not isinstance(answer.points, list):
        raise ValueError("WorkerAnswer.points must be a list")
    if len(answer.points) > max_points:
        raise ValueError(f"WorkerAnswer.points exceeds max_points={max_points}")

    ids = set()
    for p in answer.points:
        p.point_id = _require_nonempty_str(p.point_id, "Point.point_id")
        p.text = _require_nonempty_str(p.text, "Point.text")
        if p.point_id in ids:
            raise ValueError(f"Duplicated point_id in WorkerAnswer: {p.point_id}")
        ids.add(p.point_id)


def normalize_used_points(used_points: Sequence[str], known_point_ids: set[str]) -> list[str]:
    filtered = [pid for pid in dedupe_keep_order(used_points) if isinstance(pid, str) and pid in known_point_ids]
    return filtered


def validate_synthesis_draft(draft: SynthesisDraft, known_point_ids: set[str]) -> None:
    draft.node_id = _require_nonempty_str(draft.node_id, "SynthesisDraft.node_id")
    draft.synthesis_text = _require_nonempty_str(draft.synthesis_text, "SynthesisDraft.synthesis_text")
    if not isinstance(draft.used_points, list):
        raise ValueError("SynthesisDraft.used_points must be a list")
    if len(draft.used_points) != len(set(draft.used_points)):
        raise ValueError("SynthesisDraft.used_points has duplicated ids")
    for pid in draft.used_points:
        if pid not in known_point_ids:
            raise ValueError(f"SynthesisDraft.used_points contains unknown point_id: {pid}")


def normalize_grade_vote(vote: GradeVote, target_used_points: Sequence[str]) -> GradeVote:
    allowed = list(dedupe_keep_order(target_used_points))
    allowed_set = set(allowed)

    valid = vote.valid.upper() if isinstance(vote.valid, str) else "UNKNOWN"
    if valid not in {"PASS", "FAIL", "UNKNOWN"}:
        valid = "UNKNOWN"

    agree = [pid for pid in dedupe_keep_order(vote.agree_points) if pid in allowed_set]
    reject = [pid for pid in dedupe_keep_order(vote.reject_points) if pid in allowed_set and pid not in set(agree)]
    unknown = [
        pid
        for pid in dedupe_keep_order(vote.unknown_points)
        if pid in allowed_set and pid not in set(agree) and pid not in set(reject)
    ]

    covered = set(agree) | set(reject) | set(unknown)
    for pid in allowed:
        if pid not in covered:
            unknown.append(pid)

    note = vote.note if isinstance(vote.note, str) else ""
    return GradeVote(
        grader_id=vote.grader_id,
        target_id=vote.target_id,
        valid=valid,  # type: ignore[arg-type]
        agree_points=agree,
        reject_points=reject,
        unknown_points=unknown,
        note=note,
    )


def validate_grade_vote(vote: GradeVote, target_used_points: Sequence[str]) -> None:
    vote.grader_id = _require_nonempty_str(vote.grader_id, "GradeVote.grader_id")
    vote.target_id = _require_nonempty_str(vote.target_id, "GradeVote.target_id")

    if vote.valid not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ValueError(f"Invalid GradeVote.valid: {vote.valid}")

    allowed_set = set(target_used_points)
    agree = set(vote.agree_points)
    reject = set(vote.reject_points)
    unknown = set(vote.unknown_points)

    if agree & reject or agree & unknown or reject & unknown:
        raise ValueError("GradeVote point lists must be mutually exclusive")

    for pid in list(agree | reject | unknown):
        if pid not in allowed_set:
            raise ValueError(f"GradeVote contains point_id outside target used_points: {pid}")

    if (agree | reject | unknown) != allowed_set:
        raise ValueError("GradeVote must classify all target used_points exactly once")
