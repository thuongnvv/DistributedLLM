"""Persona and prompt builders."""
from __future__ import annotations

from dataclasses import dataclass


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
        "- Be concise and technically precise.\n"
        "- When grading, do not penalize missing info; only flag clear errors.\n"
    )


def _persona_prompt(domain: str, style: str, scope: str) -> str:
    return (
        f"{_base_constitution()}\n"
        f"Domain: {domain}.\n"
        f"Scope: {scope}.\n"
        f"Working style: {style}."
    )
