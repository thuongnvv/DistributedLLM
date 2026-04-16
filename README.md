# Distributed LLM RAG PoC

**Multi-node peer-review consensus with real RAG retrieval.**

Each node answers queries using its own indexed document, then cross-evaluates peers' drafts using Local-First Evidence Sharing. Users manually select which nodes participate in each query.

## Architecture

```
rag_poc/
├── api.py              # FastAPI server — /ask, /nodes/* endpoints
├── app.py              # Streamlit UI — Query tab + Provider Portal tab
├── orchestrator.py     # 4-stage pipeline orchestrator
├── llm_utils.py        # LLM-based metadata detection for provider nodes
├── lib/                # Core library (config, protocol, scoring, prompts, llm_client)
├── rag/                # RAG components (node, chunker, embedder, retriever)
├── data/               # Node documents + Chroma stores
│   └── providers/
│       └── nodes.json  # Unified node registry (all nodes)
└── scripts/
    └── index_docs.py   # Index documents into Chroma
```

**4-stage pipeline:**
1. **Answer** — Each selected node retrieves from its document + generates points with structured citations
2. **Synthesize** — Each selected node creates a draft using LOCAL_SUPPORTED points first, EXTERNAL_SUPPORTED fallback
3. **Grade** — Selected nodes cross-evaluate each other's drafts with CONTRADICTED/LOCAL_SUPPORTED/EXTERNAL_SUPPORTED/UNKNOWN basis
4. **Finalize** — Winner selected, reputation updated (add-only, no penalty)

## Quick Start

```bash
cd rag_poc

# 1. Index documents into Chroma (one-time — required before first run)
python scripts/index_docs.py

# 2. Start API + UI
python scripts/run_all.py all

# Or start individually
python scripts/run_all.py api    # API only on port 8000
python scripts/run_all.py ui     # Streamlit on port 8501
```

> **Chưa có embeddings?** Nếu gặp lỗi `Collection does not exist`: chạy bước 1. Script tự tạo vector indexes bằng sentence-transformers (all-MiniLM-L6-v2).
>
> Re-index toàn bộ: `python scripts/index_docs.py --force`

Then open **http://localhost:8501** in your browser.

## Query UI — Manual Node Selection

On the **Query** tab, the user manually selects which nodes to include in the query via multiselect. At least 2 nodes must be selected.

## Provider Portal

On the **Provider Portal** tab, providers can:

1. **Upload** a PDF or TXT document
2. The system auto-detects **domain**, **style**, and **scope** via LLM
3. The provider reviews and edits the detected metadata
4. On confirm: document is embedded and registered as an active node
5. The new node immediately appears in the Query node selector

Registered nodes can also be deleted from the **Manage Nodes** tab.

Node IDs are auto-generated from document content hash (format: `nd` + 5 hex chars).

## Node Registry (`data/providers/nodes.json`)

All nodes are registered in JSON. Each node entry contains:
- `node_id` — unique identifier
- `doc_path` — path to document file
- `chroma_path` — path to Chroma vector store
- `domain`, `style`, `scope` — metadata
- `type` — `provider` or other
- `status` — `active` or `inactive`

The orchestrator hot-reloads personas when nodes are registered or deleted. No hardcoded node list.

## Configuration

All settings via environment variables or `rag_poc/.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `openai` | `mock`, `openai`, `mega`, `gemini`, `groq`, `openrouter` |
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model |
| `MEGALLM_API_KEY` | — | API key for ai.megallm.io |
| `GROQ_API_KEY` | — | API key for Groq |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter |
| `TOP_K` | `10` | Chunks retrieved per node |
| `OPENAI_TIMEOUT` | `120.0` | LLM request timeout (seconds) |
| `MAX_EXTERNAL_POINTS_SYNTH` | `6` | Max external points per synthesis |
| `MAX_EXTERNAL_POINTS_GRADE` | `8` | Max external points per grade |
| `MAX_EVIDENCE_CHUNKS_PER_POINT` | `2` | Max chunks per point evidence |
| `PORT` | `8000` | API server port |

## Logs

Query runs are saved to `logs/run<N>/q<M>/`:
- `stage1_answers.json` — Points with structured citations
- `stage1_point_evidence.json` — Point → origin evidence mapping
- `stage2_drafts.json` — Synthesis drafts with used_points
- `stage2_adjudications.json` — Per-point decisions (LOCAL/EXTERNAL/REJECTED/UNKNOWN)
- `stage3_grades.json` — Grade votes
- `stage3_review_trace.json` — Grade reasoning trace
- `llm_raw_responses.json` — Raw LLM inputs/outputs for debugging

## Local-First Evidence Sharing

Each point carries origin evidence from the node that created it. When a node evaluates an external point:
- **LOCAL_SUPPORTED** — local doc confirms the point
- **EXTERNAL_SUPPORTED** — local doc insufficient, but origin doc confirms
- **CONTRADICTED** — local doc explicitly contradicts the point
- **UNKNOWN** — neither local nor external evidence is sufficient

Rules:
- Local evidence always takes priority over external evidence
- Only LOCAL_SUPPORTED or EXTERNAL_SUPPORTED points may appear in a draft
- External evidence only fills gaps, never overrides local contradiction