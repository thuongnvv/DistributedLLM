# Distributed LLM RAG PoC

**3-node peer-review consensus with real RAG retrieval.**

Each node answers queries using its own indexed document, then cross-evaluates peers' drafts.

## Architecture

```
RAGOrchestrator (1 process)
├── RAGNode A  →  doc1_covid19.txt (255 chunks)
├── RAGNode B  →  doc2_covid19.txt (28 chunks)
└── RAGNode C  →  doc3_covid19.txt (32 chunks)
```

**4-stage pipeline:**
1. **Answer** — Each node retrieves from its document + generates points
2. **Synthesize** — Each node creates a draft from all points
3. **Grade** — Nodes cross-evaluate each other's drafts (PASS/FAIL/UNKNOWN)
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

> **Chưa có embeddings?** Nếu gặp lỗi `Collection does not exist` hoặc không tìm thấy Chroma store: chạy bước 1. Script sẽ tự tạo vector indexes cho cả 3 nodes bằng sentence-transformers (all-MiniLM-L6-v2).
>
> Re-index toàn bộ: `python scripts/index_docs.py --force`

Then open **http://localhost:8501** in your browser.

## Configuration

All settings via environment variables (or `.env` file at `../mvp/.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_MODE` | `mega` | `mock`, `openai`, or `mega` |
| `MEGALLM_API_KEY` | — | API key for ai.megallm.io |
| `OPENAI_API_KEY` | — | API key for OpenAI |
| `K` | `3` | Number of nodes participating per query |
| `TOP_K` | `10` | Chunks retrieved per node |
| `OPENAI_TIMEOUT` | `120.0` | LLM request timeout (seconds) |
| `PORT` | `8000` | API server port |

## Logs

Query runs are saved to `logs/run<N>/q<M>/`:
- `stage1_answers.json`
- `stage2_drafts.json`
- `stage3_grades.json`
- `session_meta.json`

Run directories auto-increment: run15, run16, run17...

## 3-Node Setup

| Node | Document | Chunks |
|---|---|---|
| node_a | doc1_covid19.txt | 255 |
| node_b | doc2_covid19.txt | 28 |
| node_c | doc3_covid19.txt | 32 |
