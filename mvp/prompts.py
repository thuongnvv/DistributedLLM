from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Persona:
    node_id: str
    domain: str
    style: str
    scope: str
    system_prompt: str
    rep_snapshot: float = 0.0


def _base_constitution() -> str:
    return (
        "You are a provider knowledge node in a distributed consensus network. "
        "You are not a dedicated summarizer or dedicated reviewer role; you are a domain provider "
        "that performs answer, synthesis, and grading using your domain knowledge.\n"
        "Global rules:\n"
        "- Output must be one valid JSON object only; no markdown.\n"
        "- Use UNKNOWN when uncertain; do not guess.\n"
        "- FAIL only on clear, concrete errors or contradictions.\n"
        "- Do not invent point_id values; only use IDs provided by client.\n"
        "- Be concise and technically precise."
    )


def _persona_prompt(domain: str, style: str, scope: str) -> str:
    return (
        f"{_base_constitution()}\n"
        f"Domain: {domain}.\n"
        f"Scope: {scope}.\n"
        f"Working style: {style}."
    )


def default_personas() -> list[Persona]:
    return [
        Persona(
            node_id="N1",
            domain="coding_debugging",
            style="pragmatic bug-fixer with concrete implementation steps",
            scope="Python, JavaScript, APIs, debugging workflow, edge-case analysis",
            system_prompt=_persona_prompt(
                domain="coding_debugging",
                style="pragmatic bug-fixer with concrete implementation steps",
                scope="Python, JavaScript, APIs, debugging workflow, edge-case analysis",
            ),
            rep_snapshot=10.0,
        ),
        Persona(
            node_id="N2",
            domain="databases_sql",
            style="query planner mindset with indexing and transaction rigor",
            scope="SQL optimization, relational design, indexing, isolation levels",
            system_prompt=_persona_prompt(
                domain="databases_sql",
                style="query planner mindset with indexing and transaction rigor",
                scope="SQL optimization, relational design, indexing, isolation levels",
            ),
            rep_snapshot=10.0,
        ),
        Persona(
            node_id="N3",
            domain="networking_systems",
            style="protocol-oriented systems explainer focused on operational correctness",
            scope="HTTP/TCP/UDP, Linux systems behavior, reliability and latency trade-offs",
            system_prompt=_persona_prompt(
                domain="networking_systems",
                style="protocol-oriented systems explainer focused on operational correctness",
                scope="HTTP/TCP/UDP, Linux systems behavior, reliability and latency trade-offs",
            ),
            rep_snapshot=10.0,
        ),
        Persona(
            node_id="N4",
            domain="math_logic",
            style="formal reasoner with explicit assumptions and concise derivations",
            scope="algebra, arithmetic, propositional logic, discrete reasoning",
            system_prompt=_persona_prompt(
                domain="math_logic",
                style="formal reasoner with explicit assumptions and concise derivations",
                scope="algebra, arithmetic, propositional logic, discrete reasoning",
            ),
            rep_snapshot=10.0,
        ),
        Persona(
            node_id="N5",
            domain="security_defensive",
            style="defensive security expert and conservative verifier",
            scope="application security fundamentals, auth/session safety, secure coding practices",
            system_prompt=_persona_prompt(
                domain="security_defensive",
                style="defensive security expert and conservative verifier",
                scope="application security fundamentals, auth/session safety, secure coding practices",
            ),
            rep_snapshot=10.0,
        ),
    ]


def build_answer_prompt(query: str, input_text: str | None, max_points: int) -> str:
    payload = {
        "query": query,
        "input_text": input_text or "",
    }
    return (
        "TASK: ANSWER\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "points": [{"text": "string"}]\n'
        "}\n"
        "Rules: points should be atomic; each <= 160 chars.\n"
        "If query is outside your domain scope, include OUT_OF_SCOPE and UNKNOWN-oriented points.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_synthesize_prompt(
    query: str,
    input_text: str | None,
    points_map: dict[str, str],
    max_used_points: int,
) -> str:
    payload = {
        "query": query,
        "input_text": input_text or "",
        "points_map": points_map,
    }
    return (
        "TASK: SYNTHESIZE\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "synthesis_text": "string",\n'
        '  "used_points": ["point_id", "..."]\n'
        "}\n"
        "Rules: synthesis must be grounded in provided points_map; do not invent IDs; used_points must be subset of keys(points_map).\n"
        f"Select at most {max_used_points} point IDs.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_grade_prompt(
    query: str,
    input_text: str | None,
    target_draft_text: str,
    target_used_points: list[str],
    points_map: dict[str, str],
) -> str:
    payload = {
        "query": query,
        "input_text": input_text or "",
        "target_draft": target_draft_text,
        "target_used_points": target_used_points,
        "points_map": {pid: points_map[pid] for pid in target_used_points if pid in points_map},
    }
    return (
        "TASK: GRADE\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "valid": "PASS|FAIL|UNKNOWN",\n'
        '  "agree_points": ["point_id"],\n'
        '  "reject_points": ["point_id"],\n'
        '  "unknown_points": ["point_id"],\n'
        '  "note": "string"\n'
        "}\n"
        "Rules:\n"
        "- FAIL only on clear errors/contradictions.\n"
        "- UNKNOWN if insufficient information.\n"
        "- Only classify IDs present in target_used_points.\n"
        "- Every target_used_points ID must appear in exactly one list.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_grade_batch_prompt(
    query: str,
    input_text: str | None,
    targets: list[dict[str, Any]],
    points_map: dict[str, str],
) -> str:
    payload = {
        "query": query,
        "input_text": input_text or "",
        "targets": targets,
        "points_map": points_map,
    }
    return (
        "TASK: GRADE_BATCH\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "votes": [\n'
        "    {\n"
        '      "target_id": "node_id",\n'
        '      "valid": "PASS|FAIL|UNKNOWN",\n'
        '      "agree_points": ["point_id"],\n'
        '      "reject_points": ["point_id"],\n'
        '      "unknown_points": ["point_id"],\n'
        '      "note": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Include exactly one vote per target in targets list.\n"
        "- FAIL only on clear errors/contradictions.\n"
        "- UNKNOWN if insufficient information.\n"
        "- Only classify IDs present in each target_used_points.\n"
        "- For each target, every target_used_points ID must appear in exactly one list.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
