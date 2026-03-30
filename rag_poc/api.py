"""
FastAPI server for RAG PoC.
Serves /ask, /session, /session/{id}, /health.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from settings import (
    LLM_MODE, HOST, PORT, LOGS_ROOT, K, TAU_FAIL,
    WIN_BONUS, FAIL_PENALTY, POINT_SCORE_WEIGHT,
    MAX_POINTS_PER_ANSWER, MAX_USED_POINTS,
    NODE_A_ID, NODE_A_DOC, NODE_A_CHROMA,
    NODE_B_ID, NODE_B_DOC, NODE_B_CHROMA,
    NODE_C_ID, NODE_C_DOC, NODE_C_CHROMA,
)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.node import RAGNode
from orchestrator import RAGOrchestrator, RAGPersona, _rag_persona_prompt
from mvp.llm_client import LLMClient


# ==================== SESSION MANAGEMENT ====================
class SessionStore:
    """In-memory + disk session store, keyed by run<N>."""

    def __init__(self, logs_root: str):
        self.logs_root = Path(logs_root)
        self._sessions: dict[str, dict] = {}
        self._query_counters: dict = {}
        # Auto-detect next run number from existing directories
        existing = list(self.logs_root.glob("run*"))
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.replace("run", "")))
            except ValueError:
                pass
        self._run_counter: int = max(nums) if nums else 14

    def create_session(self, session_id: str | None = None) -> dict:
        # Auto-generate run<N> if no explicit id provided
        if session_id is None:
            self._run_counter += 1
            session_id = f"run{self._run_counter}"
        if session_id in self._sessions:
            return self._sessions[session_id]
        run_dir = self.logs_root / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = {
            "session_id": session_id,
            "session_dir": str(run_dir),
        }
        self._query_counters[session_id] = 0
        return self._sessions[session_id]

    def next_label(self, session_id: str) -> str:
        self._query_counters[session_id] = self._query_counters.get(session_id, 0) + 1
        return f"q{self._query_counters[session_id]:03d}"

    def get_history(self, session_id: str) -> dict | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        run_dir = Path(session["session_dir"])
        meta_path = run_dir / "session_meta.json"
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        return {"queries": []}


# ==================== BUILD ORCHESTRATOR ====================
def build_rag_nodes() -> list[RAGNode]:
    return [
        RAGNode(
            node_id=NODE_A_ID,
            doc_path=NODE_A_DOC,
            chroma_path=NODE_A_CHROMA,
            domain="medical_covid",
            style="medical researcher; evidence-based; cites sources",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, variants",
        ),
        RAGNode(
            node_id=NODE_B_ID,
            doc_path=NODE_B_DOC,
            chroma_path=NODE_B_CHROMA,
            domain="medical_covid",
            style="healthcare provider; practical guidance; patient-focused",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, vaccination",
        ),
        RAGNode(
            node_id=NODE_C_ID,
            doc_path=NODE_C_DOC,
            chroma_path=NODE_C_CHROMA,
            domain="medical_covid",
            style="public health official; WHO guidance; global perspective; informational",
            scope="COVID-19 disease, SARS-CoV-2 virus, symptoms, treatment, prevention, vaccination, variants, public health measures",
        ),
    ]


def build_rag_personas() -> list[RAGPersona]:
    nodes = build_rag_nodes()
    return [
        RAGPersona(
            node_id=n.node_id,
            rag_node=n,
            domain=n.domain,
            style=n.style,
            scope=n.scope,
            system_prompt=_rag_persona_prompt(
                domain=n.domain,
                style=n.style,
                scope=n.scope,
                doc_id=n.doc_path.name,
            ),
            rep_snapshot=10.0,
        )
        for n in nodes
    ]


_llm_client: LLMClient | None = None
_orchestrator: RAGOrchestrator | None = None
_session_store: SessionStore | None = None


def get_orchestrator() -> RAGOrchestrator:
    global _llm_client, _orchestrator
    if _orchestrator is None:
        _llm_client = LLMClient(mode=LLM_MODE)
        _orchestrator = RAGOrchestrator(
            rag_personas=build_rag_personas(),
            llm_client=_llm_client,
            k=K,
            tau_fail=TAU_FAIL,
            max_points_per_answer=MAX_POINTS_PER_ANSWER,
            max_used_points=MAX_USED_POINTS,
            win_bonus=WIN_BONUS,
            fail_penalty=FAIL_PENALTY,
            point_score_weight=POINT_SCORE_WEIGHT,
            logs_root=LOGS_ROOT,
        )
    return _orchestrator


def get_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(LOGS_ROOT)
    return _session_store


# ==================== FASTAPI APP ====================
app = FastAPI(title="Distributed LLM RAG PoC", version="0.1.0")


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None


class AskResponse(BaseModel):
    query: str
    session_id: str
    query_label: str
    query_dir: str
    winner: str
    final_answer: str
    citations: list[dict]
    metrics: dict
    reputation_updates: dict
    node_answers: list[dict]
    node_drafts: list[dict]
    grade_votes: list[dict]
    selected_nodes: list[str]


@app.get("/")
async def root():
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/health")
async def health():
    return {"status": "ok", "mode": LLM_MODE}


@app.post("/session")
async def create_session():
    """Create a new session."""
    store = get_store()
    session = store.create_session()
    return {"session_id": session["session_id"], "session_dir": session["session_dir"]}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session query history."""
    store = get_store()
    # Ensure session is loaded
    store.create_session(session_id)
    history = store.get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, **history}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    store = get_store()
    session = store.create_session(req.session_id)
    session_id = session["session_id"]
    session_dir = Path(session["session_dir"])
    query_label = store.next_label(session_id)

    orch = get_orchestrator()
    progress_logs: list[str] = []

    def progress(msg: str):
        progress_logs.append(msg)

    final_output, query_dir, extra = orch.run(
        query=req.query,
        progress=progress,
        session_dir=session_dir,
        query_label=query_label,
    )

    # Load stage data
    lp = Path(query_dir)
    raw_answers, raw_drafts, raw_grades = [], [], []
    if (lp / "stage1_answers.json").exists():
        with open(lp / "stage1_answers.json") as f:
            raw_answers = json.load(f)
    if (lp / "stage2_drafts.json").exists():
        with open(lp / "stage2_drafts.json") as f:
            raw_drafts = json.load(f)
    if (lp / "stage3_grades.json").exists():
        with open(lp / "stage3_grades.json") as f:
            raw_grades = json.load(f)

    # Deduplicate citations
    all_citations: dict[str, dict] = {}
    for cites in extra.get("citations_map", {}).values():
        for c in cites:
            key = f"{c['doc_id']}:{c['chunk_id']}"
            if key not in all_citations:
                all_citations[key] = c

    answers_with_citations = []
    for raw in raw_answers:
        cites = extra.get("citations_map", {}).get(raw["node_id"], [])
        answers_with_citations.append({
            "node_id": raw["node_id"],
            "abstain": len(raw.get("points", [])) == 0,
            "points": raw.get("points", []),
            "citations": cites,
        })

    return AskResponse(
        query=req.query,
        session_id=session_id,
        query_label=query_label,
        query_dir=str(query_dir),
        winner=final_output.winner,
        final_answer=final_output.final_answer,
        citations=list(all_citations.values()),
        metrics=final_output.metrics,
        reputation_updates=final_output.reputation_updates,
        node_answers=answers_with_citations,
        node_drafts=raw_drafts,
        grade_votes=raw_grades,
        selected_nodes=extra.get("selected_nodes", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
