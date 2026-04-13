# Distributed LLM RAG PoC

**6-node peer-review consensus with real RAG retrieval.**

Each node answers queries using its own indexed document, then cross-evaluates peers' drafts using Local-First Evidence Sharing.

## Architecture

```
RAGOrchestrator (1 process)
├── RAGNode A  →  doc1_covid19.txt (Wikipedia) (255 chunks)
├── RAGNode B  →  doc2_covid19.txt (Mayo Clinic) (28 chunks)
├── RAGNode C  →  doc3_covid19.txt (WHO) (32 chunks)
├── RAGNode D  →  Save and Grow (FAO) (461 chunks)
├── RAGNode E  →  Farm Management Practices (1055 chunks)
└── RAGNode F  →  Integrated Pest Management (544 chunks)
```

**4-stage pipeline:**
1. **Answer** — Each node retrieves from its document + generates points with structured citations
2. **Synthesize** — Each node creates a draft using LOCAL_SUPPORTED points first, EXTERNAL_SUPPORTED fallback
3. **Grade** — Nodes cross-evaluate each other's drafts with CONTRADICTED/LOCAL_SUPPORTED/EXTERNAL_SUPPORTED/UNKNOWN basis
4. **Finalize** — Winner selected, reputation updated

## Quick Start

```bash
cd rag_poc

# 1. Index documents into Chroma (one-time — required before first run)
python scripts/index_docs.py

# 2. Start API + UI
python scripts/run_all.py all

# Or start individually
python scripts/run_all.py api    # API only on port 8000
python scripts/run_all.py ui    # Streamlit on port 8501
```

> **Chưa có embeddings?** Nếu gặp lỗi `Collection does not exist`: chạy bước 1. Script tự tạo vector indexes bằng sentence-transformers (all-MiniLM-L6-v2).
>
> Re-index toàn bộ: `python scripts/index_docs.py --force`
>
> Re-index 1 node: `python scripts/index_docs.py --node node_d`

Then open **http://localhost:8501** in your browser.

## Configuration

All settings via environment variables (or `.env` file at `../mvp/.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `mega` | `mock`, `openai`, `groq`, `openrouter`, or `mega` |
| `MEGALLM_API_KEY` | — | API key for ai.megallm.io |
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `GROQ_API_KEY` | — | API key for Groq |
| `OPENROUTER_API_KEY` | — | API key for OpenRouter |
| `K` | `3` | Number of nodes participating per query |
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
- `stage2_drafts.json` — Synthesis drafts
- `stage2_adjudications.json` — Per-point decisions (LOCAL/EXTERNAL/REJECTED/UNKNOWN)
- `stage3_grades.json` — Grade votes
- `stage3_review_trace.json` — Grade reasoning trace
- `llm_raw_responses.json` — Raw LLM inputs/outputs
- `session_meta.json`

Run directories auto-increment: run15, run16, run17...

## 6-Node Setup

| Node | Document | Domain | Chunks |
|---|---|---|---|
| node_a | doc1_covid19.txt | COVID-19 (Wikipedia) | 255 |
| node_b | doc2_covid19.txt | COVID-19 (Mayo Clinic) | 28 |
| node_c | doc3_covid19.txt | COVID-19 (WHO) | 32 |
| node_d | Save and grow.pdf | Agriculture (FAO) | 461 |
| node_e | Farm Management Practices.pdf | Farm Management | 1055 |
| node_f | Integrated pest management.pdf | IPM Strategies | 544 |

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
