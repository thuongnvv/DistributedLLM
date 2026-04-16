"""
FastAPI server for RAG PoC.
Serves /ask, /session, /session/{id}, /nodes/*, /health.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pdfplumber
import settings

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag.node import RAGNode
from rag.chunker import chunk_document
from rag.embedder import embed_texts
from orchestrator import RAGOrchestrator, RAGPersona, _rag_persona_prompt
from lib.llm_client import LLMClient
from lib.utils import extract_first_json_object


# ==================== SESSION MANAGEMENT ====================
class SessionStore:
    """In-memory + disk session store, keyed by run<N>."""

    def __init__(self, logs_root: str):
        self.logs_root = Path(logs_root)
        self._sessions: dict[str, dict] = {}
        self._query_counters: dict = {}
        existing = list(self.logs_root.glob("run*"))
        nums = []
        for d in existing:
            try:
                nums.append(int(d.name.replace("run", "")))
            except ValueError:
                pass
        self._run_counter: int = max(nums) if nums else 14

    def create_session(self, session_id: str | None = None) -> dict:
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


# ==================== NODE REGISTRY ====================
PROVIDER_DATA_DIR = Path(__file__).parent / "data" / "providers"
NODES_REGISTRY_PATH = PROVIDER_DATA_DIR / "nodes.json"


def load_node_registry() -> dict[str, Any]:
    if not NODES_REGISTRY_PATH.exists():
        return {"nodes": []}
    return json.loads(NODES_REGISTRY_PATH.read_text())


def save_node_registry(registry: dict[str, Any]) -> None:
    NODES_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    NODES_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def get_active_nodes(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if registry is None:
        registry = load_node_registry()
    return [n for n in registry.get("nodes", []) if n.get("status") == "active"]


def slugify(node_id: str) -> str:
    """Convert a string to a valid slug."""
    import re
    slug = re.sub(r"[^a-z0-9_-]", "_", node_id.lower().strip())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def _gen_node_id(file_bytes: bytes) -> str:
    """Generate a short, fixed 7-char node_id from file content hash."""
    import hashlib
    h = hashlib.sha256(file_bytes).hexdigest()[:7]
    return f"nd{h}"


# ==================== BUILD PERSONAS FROM REGISTRY ====================
def build_rag_nodes_from_registry(registry: dict[str, Any] | None = None) -> list[RAGNode]:
    if registry is None:
        registry = load_node_registry()
    nodes: list[RAGNode] = []
    for n in get_active_nodes(registry):
        doc_path = Path(__file__).parent / n["doc_path"]
        chroma_path = Path(__file__).parent / n["chroma_path"]
        nodes.append(
            RAGNode(
                node_id=n["node_id"],
                doc_path=str(doc_path),
                chroma_path=str(chroma_path),
                domain=n.get("domain", "general"),
                style=n.get("style", "knowledge provider"),
                scope=n.get("scope", ""),
            )
        )
    return nodes


def build_rag_personas_from_registry(
    registry: dict[str, Any] | None = None,
    rep_map: dict[str, float] | None = None,
) -> list[RAGPersona]:
    if registry is None:
        registry = load_node_registry()
    if rep_map is None:
        rep_map = {}
    nodes = build_rag_nodes_from_registry(registry)
    personas: list[RAGPersona] = []
    for n in nodes:
        personas.append(
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
                rep_snapshot=rep_map.get(n.node_id, 10.0),
            )
        )
    return personas


# ==================== GLOBAL STATE ====================
_llm_client: LLMClient | None = None
_orchestrator: RAGOrchestrator | None = None
_session_store: SessionStore | None = None
_node_registry: dict[str, Any] | None = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(mode=settings.LLM_MODE)
    return _llm_client


def get_orchestrator() -> RAGOrchestrator:
    global _llm_client, _orchestrator, _node_registry
    if _orchestrator is None:
        _node_registry = load_node_registry()
        _llm_client = get_llm_client()
        _orchestrator = RAGOrchestrator(
            rag_personas=build_rag_personas_from_registry(_node_registry),
            llm_client=_llm_client,
            k=settings.K,
            tau_fail=settings.TAU_FAIL,
            max_points_per_answer=settings.MAX_POINTS_PER_ANSWER,
            max_used_points=settings.MAX_USED_POINTS,
            win_bonus=settings.WIN_BONUS,
            fail_penalty=settings.FAIL_PENALTY,
            point_score_weight=settings.POINT_SCORE_WEIGHT,
            logs_root=settings.LOGS_ROOT,
        )
    return _orchestrator


def reload_orchestrator() -> None:
    """Reload orchestrator personas from updated registry."""
    global _orchestrator, _node_registry, _llm_client
    _node_registry = load_node_registry()
    if _orchestrator is None:
        get_orchestrator()
    else:
        _orchestrator.reload_personas(build_rag_personas_from_registry(_node_registry))


def get_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(settings.LOGS_ROOT)
    return _session_store


# ==================== FASTAPI APP ====================
app = FastAPI(title="Distributed LLM RAG PoC", version="0.1.0")


# ==================== REQUEST/RESPONSE MODELS ====================
class AskRequest(BaseModel):
    query: str
    node_ids: list[str] | None = None  # user-selected nodes
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
    point_evidence: dict
    synthesis_adjudications: list[dict]
    review_trace: list[dict]
    selected_nodes: list[str]


class NodeInfo(BaseModel):
    node_id: str
    domain: str
    style: str
    scope: str
    type: str  # "builtin" or "provider"
    status: str
    doc_path: str
    chroma_path: str
    chunk_count: int | None = None


class NodePreviewResponse(BaseModel):
    node_id: str
    extracted_snippet: str
    domain: str
    style: str
    scope: str
    suggested_chunk_count: int
    file_path: str  # temp file path for registration


class NodeRegisterRequest(BaseModel):
    node_id: str
    domain: str
    style: str
    scope: str
    file_path: str  # temp path of uploaded file (saved by /preview)


# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return FileResponse(str(ui_path))
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/health")
async def health():
    reg = load_node_registry()
    return {
        "status": "ok",
        "mode": settings.LLM_MODE,
        "total_nodes": len(get_active_nodes(reg)),
        "provider_nodes": len([n for n in get_active_nodes(reg) if n.get("type") == "provider"]),
    }


# ----- Sessions -----
@app.post("/session")
async def create_session():
    store = get_store()
    session = store.create_session()
    return {"session_id": session["session_id"], "session_dir": session["session_dir"]}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    store = get_store()
    store.create_session(session_id)
    history = store.get_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, **history}


# ----- Nodes -----
@app.get("/nodes")
async def list_nodes():
    """List all registered nodes."""
    registry = load_node_registry()
    nodes: list[dict[str, Any]] = []
    for n in get_active_nodes(registry):
        chroma_path = Path(__file__).parent / n["chroma_path"]
        chunk_count = None
        if chroma_path.exists():
            try:
                from rag.retriever import Retriever
                r = Retriever(str(chroma_path), f"chunks_{n['node_id']}")
                chunk_count = r.count()
            except Exception:
                pass
        nodes.append({**n, "chunk_count": chunk_count})
    return {"nodes": nodes, "total": len(nodes)}


@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Get details of a specific node."""
    registry = load_node_registry()
    for n in registry.get("nodes", []):
        if n["node_id"] == node_id:
            chroma_path = Path(__file__).parent / n["chroma_path"]
            chunk_count = None
            if chroma_path.exists():
                try:
                    from rag.retriever import Retriever
                    r = Retriever(str(chroma_path), f"chunks_{n['node_id']}")
                    chunk_count = r.count()
                except Exception:
                    pass
            return {**n, "chunk_count": chunk_count}
    raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")


@app.post("/nodes/preview")
async def preview_node(
    file: UploadFile = File(...),
    node_id: str | None = Form(None),
    domain: str | None = Form(None),
    style: str | None = Form(None),
    scope: str | None = Form(None),
):
    """
    Upload document, extract text, auto-detect metadata.
    Returns preview for provider review. Does NOT register yet.
    """
    from llm_utils import detect_node_metadata, extract_text_from_upload, estimate_chunk_count

    # Auto-generate node_id from file content hash
    file_bytes = await file.read()
    node_id = _gen_node_id(file_bytes)
    await file.seek(0)  # reset so file can be read again later

    registry = load_node_registry()
    for n in registry.get("nodes", []):
        if n["node_id"] == node_id:
            raise HTTPException(status_code=409, detail=f"Node '{node_id}' already exists")

    # Extract text
    content = await file.read()
    text = extract_text_from_upload(content, file.filename)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    # Snippet for preview
    snippet = text[:300].strip()

    # Estimate chunk count
    est_chunks = estimate_chunk_count(text)

    # LLM auto-detect if not all fields provided
    if not domain or not style or not scope:
        llm = get_llm_client()
        detected = detect_node_metadata(text, llm, node_id)
        domain = domain or detected.get("domain", "general")
        style = style or detected.get("style", "knowledge provider; informational")
        scope = scope or detected.get("scope", "general knowledge")

    # Save temp file for later registration
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    return NodePreviewResponse(
        node_id=node_id,
        extracted_snippet=snippet,
        domain=domain,
        style=style,
        scope=scope,
        suggested_chunk_count=est_chunks,
        file_path=tmp_path,
    )


@app.post("/nodes/register")
async def register_node(req: NodeRegisterRequest):
    """
    Register a provider node after provider reviews and approves metadata.
    """
    from llm_utils import extract_text_from_file

    node_id = slugify(req.node_id)
    if not node_id:
        raise HTTPException(status_code=400, detail="Invalid node_id")

    registry = load_node_registry()

    # Check not already registered
    for n in registry.get("nodes", []):
        if n["node_id"] == node_id:
            raise HTTPException(status_code=409, detail=f"Node '{node_id}' already exists")

    # Move doc to provider data dir
    provider_dir = PROVIDER_DATA_DIR / node_id
    provider_dir.mkdir(parents=True, exist_ok=True)
    doc_path = provider_dir / "doc.txt"

    temp_path = Path(req.file_path)
    if not temp_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")

    # Extract and save text
    text = extract_text_from_file(temp_path)
    doc_path.write_text(text, encoding="utf-8")
    temp_path.unlink(missing_ok=True)

    # Index into Chroma
    chroma_path = provider_dir / "chroma_store"
    doc_id = f"{node_id}_doc.txt"

    from rag.retriever import Retriever

    # Delete existing collection if any
    try:
        r = Retriever(str(chroma_path), f"chunks_{node_id}")
        r.client.delete_collection(name=f"chunks_{node_id}")
    except Exception:
        pass

    # Chunk and embed
    chunks = chunk_document(text, doc_id)
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts)

    r2 = Retriever(str(chroma_path), f"chunks_{node_id}")
    col = r2.collection
    col.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=[{"doc_id": doc_id} for _ in chunks],
    )

    # Register in registry
    chroma_rel = str(chroma_path.relative_to(Path(__file__).parent))
    doc_rel = str(doc_path.relative_to(Path(__file__).parent))
    node_entry = {
        "node_id": node_id,
        "doc_path": doc_rel,
        "chroma_path": chroma_rel,
        "domain": req.domain,
        "style": req.style,
        "scope": req.scope,
        "type": "provider",
        "status": "active",
    }
    registry.setdefault("nodes", [])
    registry["nodes"].append(node_entry)
    save_node_registry(registry)

    # Reload orchestrator
    reload_orchestrator()

    return {
        "node_id": node_id,
        "status": "registered",
        "chunks_indexed": len(chunks),
        "domain": req.domain,
        "style": req.style,
        "scope": req.scope,
    }


@app.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Unregister and delete a node."""
    registry = load_node_registry()
    found = False
    new_nodes = []
    for n in registry.get("nodes", []):
        if n["node_id"] == node_id:
            found = True
            provider_dir = PROVIDER_DATA_DIR / node_id
            if provider_dir.exists():
                shutil.rmtree(provider_dir)
        else:
            new_nodes.append(n)
    if not found:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    registry["nodes"] = new_nodes
    save_node_registry(registry)
    reload_orchestrator()
    return {"node_id": node_id, "status": "deleted"}


@app.post("/nodes/reload")
async def reload_nodes():
    """Reload orchestrator personas from registry."""
    reload_orchestrator()
    reg = load_node_registry()
    return {
        "status": "reloaded",
        "total_nodes": len(get_active_nodes(reg)),
    }


# ----- Ask -----
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")

    if not req.node_ids or len(req.node_ids) < 2:
        raise HTTPException(status_code=400, detail="node_ids must contain at least 2 nodes")

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
        selected_node_ids=req.node_ids,
    )

    # Load stage data
    lp = Path(query_dir)
    raw_answers, raw_drafts, raw_grades = [], [], []
    raw_point_evidence: dict = {}
    raw_synthesis_adjudications: list[dict] = []
    raw_review_trace: list[dict] = []
    if (lp / "stage1_answers.json").exists():
        with open(lp / "stage1_answers.json") as f:
            raw_answers = json.load(f)
    if (lp / "stage1_point_evidence.json").exists():
        with open(lp / "stage1_point_evidence.json") as f:
            raw_point_evidence = json.load(f)
    if (lp / "stage2_drafts.json").exists():
        with open(lp / "stage2_drafts.json") as f:
            raw_drafts = json.load(f)
    if (lp / "stage2_adjudications.json").exists():
        with open(lp / "stage2_adjudications.json") as f:
            raw_synthesis_adjudications = json.load(f)
    if (lp / "stage3_grades.json").exists():
        with open(lp / "stage3_grades.json") as f:
            raw_grades = json.load(f)
    if (lp / "stage3_review_trace.json").exists():
        with open(lp / "stage3_review_trace.json") as f:
            raw_review_trace = json.load(f)

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
        point_evidence=raw_point_evidence,
        synthesis_adjudications=raw_synthesis_adjudications,
        review_trace=raw_review_trace,
        selected_nodes=extra.get("selected_nodes", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
