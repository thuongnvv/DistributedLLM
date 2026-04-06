"""
Streamlit UI for Distributed LLM RAG PoC.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
import requests

sys.path.insert(0, str(Path(__file__).parent))

import settings

API_BASE = os.getenv("API_BASE", f"http://localhost:{settings.PORT}")

st.set_page_config(
    page_title="Distributed LLM RAG",
    page_icon="🤖",
    layout="wide",
)


def api_get(path: str, **kwargs) -> dict:
    r = requests.get(f"{API_BASE}{path}", timeout=30, **kwargs)
    r.raise_for_status()
    return r.json()


def api_post(path: str, data: dict | None = None, **kwargs) -> dict:
    r = requests.post(f"{API_BASE}{path}", json=data, timeout=300, **kwargs)
    r.raise_for_status()
    return r.json()


# ==================== SESSION INIT ====================
if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.session_history = []
    st.session_state.result = None
    st.session_state.query = ""
    st.session_state.elapsed = 0
    st.session_state.query_label = ""


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Config")
    try:
        health = api_get("/health")
        st.success(f"API: {health.get('status', 'ok')}")
        st.info(f"LLM Mode: `{health.get('mode', 'unknown')}`")
    except Exception as e:
        st.error(f"API offline: {e}")

    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.session_history = []
        st.session_state.result = None
        st.session_state.query = ""
        st.session_state.query_label = ""
        st.rerun()

    # Show session history
    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id}`")
        try:
            hist = api_get(f"/session/{st.session_state.session_id}")
            queries = hist.get("queries", [])
            if queries:
                st.markdown("**Query History:**")
                for q in queries:
                    label = q.get("label", "?")
                    query_text = q.get("query", "")[:50]
                    winner = q.get("winner", "?")
                    st.markdown(
                        f"`{label}` — *{query_text}*... → 🏆 `{winner}`"
                    )
        except Exception:
            pass

    st.divider()
    st.markdown("### Pipeline")
    st.markdown("""
    **4-Stage Consensus:**
    1. **Answer** — Each node retrieves from its document & generates points
    2. **Synthesize** — Each node creates a draft from the bundle of all points
    3. **Grade** — Nodes cross-evaluate each other's drafts (PASS/FAIL/UNKNOWN)
    4. **Finalize** — Winner selected, reputation updated
    """)
    st.divider()
    st.markdown("### Nodes")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Node A**\n`doc1_covid19.txt`")
    with col2:
        st.markdown("**Node B**\n`doc2_covid19.txt`")
    with col3:
        st.markdown("**Node C**\n`doc3_covid19.txt`")


# ==================== MAIN ====================
st.title("🤖 Distributed LLM RAG PoC")
st.markdown("*2-node peer-review consensus with real RAG retrieval*")

query = st.text_area(
    "Ask a question about COVID-19:",
    value=st.session_state.query,
    placeholder="e.g. What are the symptoms of COVID-19?",
    height=80,
)

col1, col2 = st.columns(2)
with col1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.result = None
    st.session_state.query = ""
    st.rerun()

if ask_clicked and query.strip():
    st.session_state.query = query
    st.session_state.result = None

    with st.spinner("Running 4-stage pipeline..."):
        start = time.time()
        try:
            payload = {"query": query.strip()}
            if st.session_state.session_id:
                payload["session_id"] = st.session_state.session_id

            result = api_post("/ask", data=payload)
            st.session_state.elapsed = time.time() - start
            st.session_state.result = result
            st.session_state.session_id = result.get("session_id", st.session_state.session_id)
            st.session_state.query_label = result.get("query_label", "")

            if st.session_state.session_id:
                try:
                    hist = api_get(f"/session/{st.session_state.session_id}")
                    st.session_state.session_history = hist.get("queries", [])
                except Exception:
                    pass

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.result = None


# ==================== RESULTS ====================
if st.session_state.result:
    result = st.session_state.result
    elapsed = st.session_state.elapsed

    st.divider()
    st.subheader(f"🏆 Winner: `{result['winner']}`")
    # Light background for readability in dark mode
    st.success(result["final_answer"])
    st.caption(
        f"⏱ {elapsed:.1f}s  |  Session `{result['session_id']}`  |  "
        f"`{result['query_label']}`  |  📁 {result['query_dir']}"
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Node Answers",
        "📝 Synthesis Drafts",
        "🔍 Cross-Grading",
        "📊 Metrics & Reputation",
        "📖 All Citations",
    ])

    with tab1:
        for ans in result.get("node_answers", []):
            with st.expander(f"**{ans['node_id']}** {'— ABSTAIN' if ans.get('abstain') else ''}", expanded=True):
                if ans.get("points"):
                    st.markdown(f"**Points ({len(ans['points'])})**")
                    for p in ans["points"]:
                        st.markdown(f"- `{p['point_id']}` — {p['text']}")
                        if p.get("citations"):
                            st.caption("origin citations: " + ", ".join(p["citations"]))
                else:
                    st.markdown("_No points (abstain or no retrieval)_")

                if ans.get("citations"):
                    st.markdown(f"**Retrieved Chunks ({len(ans['citations'])}):**")
                    for c in ans["citations"]:
                        score = c.get("score", 0)
                        text = c["text"][:300]
                        st.code(f"[{c['doc_id']}:{c['chunk_id']}] score={score:.3f}\n{text}", language=None)

    with tab2:
        adjudication_by_node = {
            item.get("node_id"): item.get("point_support", [])
            for item in result.get("synthesis_adjudications", [])
        }
        for draft in result.get("node_drafts", []):
            is_unknown = not draft.get("synthesis_text") or draft["synthesis_text"] == "UNKNOWN"
            with st.expander(f"**{draft['node_id']}**", expanded=True):
                if is_unknown:
                    st.markdown("_UNKNOWN — abstain or no valid points_")
                else:
                    st.markdown(draft["synthesis_text"])
                if draft.get("used_points"):
                    support_map = {
                        item.get("point_id"): item
                        for item in adjudication_by_node.get(draft["node_id"], [])
                    }
                    st.caption(
                        "Used points: "
                        + ", ".join(
                            f"{pid} ({'LOCAL' if support_map.get(pid, {}).get('decision') == 'LOCAL_SUPPORTED' else 'EXTERNAL' if support_map.get(pid, {}).get('decision') == 'EXTERNAL_SUPPORTED' else support_map.get(pid, {}).get('decision', 'UNKNOWN')})"
                            for pid in draft["used_points"]
                        )
                    )
                point_support = adjudication_by_node.get(draft["node_id"], [])
                if point_support:
                    st.markdown("**Point Adjudication**")
                    for item in point_support:
                        st.caption(
                            f"{item.get('point_id')} -> {item.get('decision')}"
                            + (f" | {item.get('reason')}" if item.get("reason") else "")
                        )

    with tab3:
        votes = result.get("grade_votes", [])
        review_trace_by_pair = {
            (item.get("grader_id"), item.get("target_id")): item.get("point_reviews", [])
            for item in result.get("review_trace", [])
        }
        if not votes:
            st.info("No grade votes recorded.")
        for g in votes:
            valid = g.get("valid", "UNKNOWN")
            icon = {"PASS": "🟢", "FAIL": "🔴", "UNKNOWN": "🟡"}.get(valid, "⚪")
            with st.expander(f"{icon} **{g['grader_id']}** → **{g['target_id']}**  **[{valid}]**", expanded=False):
                col_a, col_b, col_u = st.columns(3)
                with col_a:
                    st.markdown("**Agree:**")
                    for p in g.get("agree_points", []):
                        st.markdown(f"- `{p}` 🟢")
                with col_b:
                    st.markdown("**Reject:**")
                    for p in g.get("reject_points", []):
                        st.markdown(f"- `{p}` 🔴")
                with col_u:
                    st.markdown("**Unknown:**")
                    for p in g.get("unknown_points", []):
                        st.markdown(f"- `{p}` 🟡")
                if g.get("note"):
                    st.caption(f"_Note: {g['note']}_")
                point_reviews = review_trace_by_pair.get((g["grader_id"], g["target_id"]), [])
                if point_reviews:
                    st.markdown("**Review trace**")
                    for item in point_reviews:
                        st.caption(
                            f"{item.get('point_id')} -> {item.get('basis')}"
                            + (f" | {item.get('reason')}" if item.get("reason") else "")
                        )

    with tab4:
        metrics = result.get("metrics", {})
        if metrics:
            pass_counts = metrics.get("pass", {})
            fail_counts = metrics.get("fail", {})
            agree_counts = metrics.get("agree", {})
            reject_counts = metrics.get("reject", {})
            used_counts = metrics.get("used_points", {})
            agree_rates = metrics.get("agree_rate", {})

            nodes = list(pass_counts.keys())
            if nodes:
                cols = st.columns(len(nodes))
                for i, n in enumerate(nodes):
                    with cols[i]:
                        st.markdown(f"### `{n}`")
                        st.metric("PASS", pass_counts.get(n, 0))
                        st.metric("FAIL", fail_counts.get(n, 0))
                        st.metric("AGREE", agree_counts.get(n, 0))
                        st.metric("REJECT", reject_counts.get(n, 0))
                        st.metric("Used Points", used_counts.get(n, 0))
                        rate = agree_rates.get(n, 0)
                        st.progress(rate, text=f"Agree rate: {rate:.0%}")

        rep = result.get("reputation_updates", {})
        if rep:
            st.markdown("### Reputation Updates")
            for key, val in rep.get("node_rep_delta", {}).items():
                delta_str = f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
                icon = "🟢" if val >= 0 else "🔴"
                st.markdown(f"**{key} (winner/node)**: {icon} `{delta_str}`")
            for key, val in rep.get("point_rep_delta", {}).items():
                delta_str = f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
                icon = "🟢" if val >= 0 else "🔴"
                st.markdown(f"**{key} (point)**: {icon} `{delta_str}`")

    with tab5:
        citations = result.get("citations", [])
        if not citations:
            st.info("No citations.")
        for c in citations:
            score = c.get("score", 0)
            text = c["text"][:500]
            st.code(f"[{c['doc_id']}:{c['chunk_id']}] score={score:.3f}\n{text}", language=None)

elif not query.strip():
    st.info("👆 Enter a question above and click **Ask** to run the pipeline.")
