from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from config import DEFAULT_K, DEFAULT_LOG_ROOT, DEFAULT_MODE, DEFAULT_SEED, DEFAULT_TAU_FAIL, SUPPORTED_MODES
from llm_client import LLMClient
from orchestrator import MVPOrchestrator
from prompts import default_personas
from utils import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Community-driven distributed LLM MVP simulator")
    parser.add_argument("--query", type=str, default=None, help="Single query string")
    parser.add_argument("--input-text", type=str, default=None, help="Optional grounded input text")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=SUPPORTED_MODES)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of worker nodes")
    parser.add_argument("--tau-fail", type=int, default=DEFAULT_TAU_FAIL, help="FAIL threshold for discard")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--logs-root", type=str, default=DEFAULT_LOG_ROOT)

    parser.add_argument("--testcases", type=str, default=None, help="Path to testcases JSON")
    parser.add_argument("--case", type=str, default=None, help="Case id or 1-based index")
    parser.add_argument("--list-cases", action="store_true", help="List testcase IDs and exit")

    return parser.parse_args()


def load_testcases(path: str) -> list[dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        cases = data.get("cases")
    else:
        cases = data

    if not isinstance(cases, list):
        raise ValueError("testcases file must be a JSON list or object with 'cases' list")

    normalized: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"testcase index {idx} is not an object")
        if "query" not in case or not str(case["query"]).strip():
            raise ValueError(f"testcase index {idx} missing non-empty query")
        c = dict(case)
        c.setdefault("id", f"case_{idx}")
        normalized.append(c)

    if not normalized:
        raise ValueError("No testcases found")
    return normalized


def resolve_testcases_path(path_arg: str | None) -> Path | None:
    script_dir = Path(__file__).resolve().parent
    if path_arg:
        path = Path(path_arg)
        if path.exists():
            return path
        alt = script_dir / path_arg
        if alt.exists():
            return alt
        raise ValueError(f"Testcases file not found: {path_arg}")

    default_path = script_dir / "testcases.json"
    if default_path.exists():
        return default_path
    return None


def load_env_if_exists(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_default_env_files() -> None:
    root_dir = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates = [
        root_dir / ".env",
        root_dir / "mvp" / ".env",
        script_dir / ".env",
    ]
    for candidate in candidates:
        load_env_if_exists(candidate)


def select_case(cases: list[dict[str, Any]], selector: str | None) -> tuple[dict[str, Any], int]:
    if selector is None:
        return cases[0], 1

    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(cases):
            raise ValueError(f"--case index out of range: {index} (1..{len(cases)})")
        return cases[index - 1], index

    for idx, case in enumerate(cases, start=1):
        if str(case.get("id")) == selector:
            return case, idx

    raise ValueError(f"Cannot find testcase with id: {selector}")


def main() -> int:
    args = parse_args()
    load_default_env_files()

    query_arg: str | None = args.query
    testcases_path = resolve_testcases_path(args.testcases)

    if args.list_cases:
        if testcases_path is None:
            raise ValueError("--list-cases requires --testcases (or default mvp/testcases.json)")
        cases = load_testcases(str(testcases_path))
        for idx, case in enumerate(cases, start=1):
            print(f"{idx}. {case['id']}")
        return 0

    personas = default_personas()
    llm_client = LLMClient(mode=args.mode, seed=args.seed)

    orchestrator = MVPOrchestrator(
        personas=personas,
        llm_client=llm_client,
        k=args.k,
        tau_fail=args.tau_fail,
        seed=args.seed,
        logs_root=args.logs_root,
    )

    def progress(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    if query_arg is not None and query_arg.strip():
        print("[ad-hoc] start", file=sys.stderr, flush=True)
        final_output, log_dir = orchestrator.run(
            query=query_arg,
            input_text=args.input_text,
            case_id=None,
            progress=progress,
        )
        print(final_output.final_answer)
        print(str(log_dir))
        return 0

    if testcases_path is None:
        raise ValueError("Provide --query or --testcases (or add mvp/testcases.json)")

    cases = load_testcases(str(testcases_path))

    if args.case:
        selected_case, selected_index = select_case(cases, args.case)
        case_id = str(selected_case.get("id", f"case_{selected_index}"))
        query = str(selected_case["query"])
        input_text = args.input_text
        if input_text is None and selected_case.get("input_text") is not None:
            input_text = str(selected_case["input_text"])

        print(f"[{case_id}] start", file=sys.stderr, flush=True)
        final_output, log_dir = orchestrator.run(
            query=query,
            input_text=input_text,
            case_id=case_id,
            progress=progress,
        )
        print(final_output.final_answer)
        print(str(log_dir))
        return 0

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case_{idx}"))
        query = str(case["query"])
        input_text = args.input_text
        if input_text is None and case.get("input_text") is not None:
            input_text = str(case["input_text"])

        print(f"[{case_id}] start ({idx}/{total})", file=sys.stderr, flush=True)
        final_output, log_dir = orchestrator.run(
            query=query,
            input_text=input_text,
            case_id=case_id,
            progress=progress,
        )
        print(f"=== [{idx}/{total}] {case_id} ===")
        print(final_output.final_answer)
        print(str(log_dir))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
