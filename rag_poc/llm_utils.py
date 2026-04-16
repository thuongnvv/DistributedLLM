"""
LLM utilities for provider onboarding.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pdfplumber


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from a PDF or TXT file."""
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8")

    text_parts: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_upload(content: bytes, filename: str) -> str:
    """Extract text from uploaded file content (bytes)."""
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as f:
        f.write(content)
        f.flush()
        path = Path(f.name)

    try:
        return extract_text_from_file(path)
    finally:
        path.unlink(missing_ok=True)


def estimate_chunk_count(text: str, chunk_size: int = 600, chunk_overlap: int = 80) -> int:
    """Roughly estimate how many chunks a document will produce."""
    import math
    # Average effective chunk size accounting for overlap
    effective_size = chunk_size - chunk_overlap
    words = len(text.split())
    # Rough estimate: ~4 chars per word, chunk_size chars per chunk
    chars_per_word = 5
    total_chars = words * chars_per_word
    n_chunks = math.ceil(total_chars / effective_size) if effective_size > 0 else 0
    return n_chunks


def detect_node_metadata(
    text: str,
    llm_client: Any,
    node_id: str,
) -> dict[str, str]:
    """
    Use LLM to auto-detect domain, style, and scope from document text.
    Returns dict with 'domain', 'style', 'scope'.
    """
    # Use first 3000 chars for fast analysis
    snippet = text[:3000]

    prompt = (
        "Analyze the following document and extract metadata for a knowledge provider node.\n"
        "Output ONLY a valid JSON object with exactly these fields:\n"
        "{\n"
        '  "domain": "one-word or hyphenated lowercase domain label (e.g. healthcare, legal, finance, agriculture, technology, education)",\n'
        '  "style": "working style description (e.g. clinical researcher; evidence-based; formal)",\n'
        '  "scope": "comma-separated list of 5-10 specific topics this document covers"\n'
        "}\n\n"
        "Rules:\n"
        "- domain: be specific but concise (1-3 words max)\n"
        "- style: describe the author's perspective/approach (2-4 short phrases)\n"
        "- scope: list the main topics covered in the document\n"
        "- If you cannot determine a field, use a reasonable default\n\n"
        f"DOCUMENT:\n{snippet}"
    )

    system_prompt = (
        "You are a metadata extraction assistant. You analyze documents and extract structured metadata. "
        "Output ONLY JSON — no explanation, no markdown, no text outside the JSON object."
    )

    try:
        response = llm_client._chat_completion(
            system_prompt=system_prompt,
            user_prompt=prompt,
            trace={"phase": "node_metadata_detection", "node_id": node_id},
        )
        obj = json.loads(response.strip())
        return {
            "domain": str(obj.get("domain", "general")).strip(),
            "style": str(obj.get("style", "knowledge provider")).strip(),
            "scope": str(obj.get("scope", "general knowledge")).strip(),
        }
    except (json.JSONDecodeError, Exception):
        # Fallback
        return {
            "domain": "general",
            "style": "knowledge provider; informational",
            "scope": "general knowledge",
        }
