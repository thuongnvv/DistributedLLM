"""
RAGOrchestrator: extends MVPOrchestrator with RAG retrieval for Stage 1 and Stage 3.

Each node has its own document + vector store. Both answer and grade phases
retrieve from the node's own store.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field

# Add repo root for mvp imports
# Add repo root + mvp explicitly before any imports
_repo_root = str(Path(__file__).parent.parent)
sys.path.insert(0, _repo_root)
sys.path.insert(1, str(Path(_repo_root) / "mvp"))

from mvp.config import (
    FAIL_PENALTY,
    MAX_POINTS_PER_ANSWER,
    MAX_USED_POINTS,
    POINT_SCORE_WEIGHT,
    WIN_BONUS,
)
from mvp.llm_client import LLMClient
from mvp.prompts import Persona, _base_constitution
from mvp.protocol import (
    FinalizationOutput,
    GradeVote,
    SynthesisDraft,
    WorkerAnswer,
    normalize_grade_vote,
    normalize_used_points,
    validate_grade_vote,
    validate_synthesis_draft,
    validate_worker_answer,
)
from mvp.scoring import finalize_output
from mvp.utils import ensure_dir, run_timestamp, save_json, dedupe_keep_order

from rag.node import RAGNode
from models import Citation


def _rag_persona_prompt(domain: str, style: str, scope: str, doc_id: str) -> str:
    return (
        f"{_base_constitution()}\n"
        f"You are a knowledge provider node specializing in the document: {doc_id}\n"
        f"Domain: {domain}.\n"
        f"Scope: {scope}.\n"
        f"Working style: {style}.\n"
        "IMPORTANT: All your knowledge comes from the retrieved context. "
        "Cite sources as [doc_id:chunk_id] for factual claims."
    )


@dataclass
class RAGPersona:
    """A persona backed by a RAG node."""
    node_id: str
    rag_node: RAGNode
    domain: str
    style: str
    scope: str
    system_prompt: str
    rep_snapshot: float = 10.0

    def to_mvp_persona(self) -> Persona:
        return Persona(
            node_id=self.node_id,
            domain=self.domain,
            style=self.style,
            scope=self.scope,
            system_prompt=self.system_prompt,
            rep_snapshot=self.rep_snapshot,
        )


def _build_answer_prompt_rag(query: str, context: str, max_points: int | None) -> str:
    import json
    max_rule = f"- Return at most {max_points} points.\n" if max_points is not None else ""
    return (
        "TASK: ANSWER\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "abstain": true|false,\n'
        '  "points": [{"text": "string"}]\n'
        "}\n"
        "Rules:\n"
        "- Answer ONLY using the provided context below.\n"
        "- Cite sources inline as [doc_id:chunk_id] for factual claims.\n"
        "- If the context does not contain enough information, set abstain=true.\n"
        "- Each point must be atomic and <= 160 chars.\n"
        f"{max_rule}"
        f"CONTEXT:\n{context}\n\n"
        f"Input JSON:\n"
        f'{{"query": {json.dumps(query)}}}'
    )


def _build_grade_prompt_rag(
    query: str,
    context: str,
    target_draft_text: str,
    target_used_points: list[str],
    points_map: dict[str, str],
) -> str:
    import json
    relevant_points = {pid: points_map[pid] for pid in target_used_points if pid in points_map}
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
        "- Use ONLY the provided context to evaluate the draft.\n"
        "- FAIL only on clear errors or contradictions you can verify from context.\n"
        "- UNKNOWN if insufficient information in context.\n"
        "- Only classify IDs present in target_used_points.\n"
        "- Every target_used_points ID must appear in exactly one list.\n"
        f"CONTEXT:\n{context}\n\n"
        f"Input JSON:\n"
        f'{{"query": {json.dumps(query)}, "target_draft": {json.dumps(target_draft_text)}, "target_used_points": {json.dumps(target_used_points)}, "points_map": {json.dumps(relevant_points)}}}'
    )


def _build_synthesize_prompt_rag(
    query: str,
    points_map: dict[str, str],
    context: str,
    max_used_points: int | None,
    repair_feedback: str | None = None,
) -> str:
    import json
    max_rule = f"- Select at most {max_used_points} point IDs.\n" if max_used_points is not None else ""
    repair_rule = f"- Contract repair note: {repair_feedback}\n" if repair_feedback else ""
    return (
        "TASK: SYNTHESIZE\n"
        "Return EXACT JSON schema:\n"
        "{\n"
        '  "synthesis_text": "string",\n'
        '  "used_points": ["point_id", "..."]\n'
        "}\n"
        "Rules:\n"
        "- Use ONLY the provided context to ground your synthesis.\n"
        "- Cite sources as [doc_id:chunk_id] in synthesis_text.\n"
        "- Synthesis must be grounded in provided points_map AND context.\n"
        "- Do not invent IDs; used_points must be subset of keys(points_map).\n"
        "- Avoid redundant points that repeat the same idea.\n"
        "- If synthesis_text is not 'UNKNOWN', used_points must not be empty.\n"
        "- Every major claim in synthesis_text must be traceable to at least one point_id in used_points.\n"
        "- If you cannot support the answer with point IDs, return synthesis_text='UNKNOWN' and used_points=[].\n"
        f"{max_rule}"
        f"{repair_rule}"
        f"CONTEXT (from your document):\n{context}\n\n"
        f"Input JSON:\n"
        f'{{"query": {json.dumps(query)}, "points_map": {json.dumps(points_map)}}}'
    )


class RAGOrchestrator:
    def __init__(
        self,
        rag_personas: list[RAGPersona],
        llm_client: LLMClient,
        k: int,
        tau_fail: int = 2,
        seed: int = 42,
        logs_root: str = "logs",
        max_points_per_answer: int | None = MAX_POINTS_PER_ANSWER,
        max_used_points: int | None = MAX_USED_POINTS,
        win_bonus: float = WIN_BONUS,
        fail_penalty: float = FAIL_PENALTY,
        point_score_weight: float = POINT_SCORE_WEIGHT,
    ):
        if k <= 0:
            raise ValueError("k must be > 0")
        if k > len(rag_personas):
            raise ValueError(f"k={k} exceeds available rag_personas={len(rag_personas)}")

        self.rag_personas = rag_personas
        self.llm_client = llm_client
        self.k = k
        self.tau_fail = tau_fail
        self.seed = seed
        self.logs_root = logs_root
        self.max_points_per_answer = max_points_per_answer
        self.max_used_points = max_used_points
        self.win_bonus = win_bonus
        self.fail_penalty = fail_penalty
        self.point_score_weight = point_score_weight

        # Map node_id → RAGPersona for quick lookup
        self._persona_map: dict[str, RAGPersona] = {rp.node_id: rp for rp in rag_personas}

    def run(
        self,
        query: str,
        progress: Callable[[str], None] | None = None,
        session_dir: Path | None = None,
        query_label: str | None = None,
    ) -> tuple[FinalizationOutput, Path, dict[str, Any]]:
        """Run the full 4-stage RAG pipeline. Returns (final_output, query_dir, extra_data)."""
        if not query.strip():
            raise ValueError("Query must be non-empty")

        self.llm_client.reset_raw_events()
        selected = self._select_workers()
        self._emit(progress, f"selected_nodes={','.join(rp.node_id for rp in selected)}")

        # Build per-query log dir inside session
        if session_dir is None:
            session_dir = Path(self.logs_root) / f"session_{run_timestamp()}"
        if query_label is None:
            query_label = f"q_{run_timestamp()}"

        # ==================== Stage 1: RAG Answering ====================
        self._emit(progress, "stage=1/4 phase=answer start")
        answers: list[WorkerAnswer] = []
        citations_map: dict[str, list[Citation]] = {}
        for rp in selected:
            self._emit(progress, f"stage=1/4 node={rp.node_id} retrieving")
            citations = rp.rag_node.retrieve(query)
            citations_map[rp.node_id] = citations
            context = rp.rag_node.build_rag_context(citations)

            if not citations or not context.strip():
                # No retrieval → abstain
                self._emit(progress, f"stage=1/4 node={rp.node_id} abstain(no_retrieval)")
                answers.append(WorkerAnswer(node_id=rp.node_id, points=[]))
                continue

            # Call LLM with RAG context
            self._emit(progress, f"stage=1/4 node={rp.node_id} calling_llm")
            mvp_persona = rp.to_mvp_persona()
            answer = self._llm_answer_rag(
                persona=mvp_persona,
                query=query,
                context=context,
                max_points=self.max_points_per_answer,
            )
            try:
                validate_worker_answer(answer, max_points=self.max_points_per_answer)
            except ValueError:
                answer = WorkerAnswer(node_id=rp.node_id, points=[])
            answers.append(answer)
        self._emit(progress, "stage=1/4 phase=answer done")

        # Build maps
        points_map: dict[str, str] = {}
        point_owner_map: dict[str, str] = {}
        for ans in answers:
            for pt in ans.points:
                points_map[pt.point_id] = pt.text
                point_owner_map[pt.point_id] = ans.node_id
        active_nodes = {ans.node_id for ans in answers if ans.points}
        known_point_ids = set(points_map.keys())

        # ==================== Stage 2: Synthesis ====================
        self._emit(progress, "stage=2/4 phase=synthesize start")
        drafts: list[SynthesisDraft] = []
        for rp in selected:
            if rp.node_id not in active_nodes:
                self._emit(progress, f"stage=2/4 node={rp.node_id} abstain_synthesis")
                draft = SynthesisDraft(node_id=rp.node_id, synthesis_text="UNKNOWN", used_points=[])
                drafts.append(draft)
                continue

            # Retrieve RAG context from this node's own document
            self._emit(progress, f"stage=2/4 node={rp.node_id} retrieving")
            synthesis_citations = rp.rag_node.retrieve(query)
            synthesis_context = rp.rag_node.build_rag_context(synthesis_citations)

            self._emit(progress, f"stage=2/4 node={rp.node_id} synthesizing")
            draft = self._llm_synthesize_rag(
                persona=rp.to_mvp_persona(),
                bundle=answers,
                query=query,
                context=synthesis_context,
                max_used_points=self.max_used_points,
            )
            draft.used_points = normalize_used_points(draft.used_points, known_point_ids)
            try:
                validate_synthesis_draft(draft, known_point_ids)
            except ValueError:
                draft = SynthesisDraft(node_id=rp.node_id, synthesis_text="UNKNOWN", used_points=[])
            drafts.append(draft)
        self._emit(progress, "stage=2/4 phase=synthesize done")
        active_drafts = [d for d in drafts if self._is_active_draft(d)]
        draft_by_node = {d.node_id: d for d in drafts}

        # ==================== Stage 3: RAG Cross-grading ====================
        self._emit(progress, "stage=3/4 phase=grade start")
        grades: list[GradeVote] = []
        for rp in selected:
            if rp.node_id not in active_nodes:
                self._emit(progress, f"stage=3/4 grader={rp.node_id} skip(no in-scope points)")
                continue

            # Retrieve from THIS node's own document
            self._emit(progress, f"stage=3/4 grader={rp.node_id} retrieving")
            grading_context = rp.rag_node.retrieve(query)
            context_str = rp.rag_node.build_rag_context(grading_context)

            target_drafts = [d for d in active_drafts if d.node_id != rp.node_id]
            if not target_drafts:
                continue

            self._emit(progress, f"stage=3/4 grader={rp.node_id} grading_batch")
            batch_votes = self._llm_grade_batch_rag(
                persona=rp.to_mvp_persona(),
                query=query,
                context=context_str,
                targets=target_drafts,
                points_map=points_map,
            )

            vote_map: dict[str, GradeVote] = {}
            for vote in batch_votes:
                if vote.target_id in vote_map:
                    continue
                target_draft = draft_by_node.get(vote.target_id)
                if target_draft is None:
                    continue
                norm = normalize_grade_vote(vote, target_draft.used_points)
                try:
                    validate_grade_vote(norm, target_draft.used_points)
                except ValueError:
                    continue
                vote_map[vote.target_id] = norm

            for target_draft in target_drafts:
                vote = vote_map.get(target_draft.node_id)
                if vote is None:
                    vote = GradeVote(
                        grader_id=rp.node_id,
                        target_id=target_draft.node_id,
                        valid="UNKNOWN",
                        agree_points=[],
                        reject_points=[],
                        unknown_points=list(target_draft.used_points),
                        note="Missing target in batch response",
                    )
                    try:
                        validate_grade_vote(vote, target_draft.used_points)
                    except ValueError:
                        continue
                grades.append(vote)
        self._emit(progress, "stage=3/4 phase=grade done")

        # ==================== Stage 4: Finalization ====================
        self._emit(progress, "stage=4/4 phase=finalize start")
        final_output = finalize_output(
            drafts=active_drafts,
            grades=grades,
            point_owner_map=point_owner_map,
            tau_fail=self.tau_fail,
            win_bonus=self.win_bonus,
            fail_penalty=self.fail_penalty,
            point_score_weight=self.point_score_weight,
        )
        self._emit(progress, f"stage=4/4 phase=finalize done winner={final_output.winner}")

        # ==================== Write Logs ====================
        query_dir = self._write_logs(
            query=query,
            selected=selected,
            answers=answers,
            citations_map=citations_map,
            drafts=drafts,
            grades=grades,
            final_output=final_output,
            session_dir=session_dir,
            query_label=query_label,
        )

        # Build extra data for UI
        extra: dict[str, Any] = {
            "citations_map": {k: [c.model_dump() for c in v] for k, v in citations_map.items()},
            "active_nodes": list(active_nodes),
            "selected_nodes": [rp.node_id for rp in selected],
            "session_dir": str(session_dir),
            "query_label": query_label,
        }

        return final_output, query_dir, extra

    # ---- LLM calls with RAG context ----

    def _llm_answer_rag(
        self,
        persona: Persona,
        query: str,
        context: str,
        max_points: int | None,
    ) -> WorkerAnswer:
        """Call LLM answer with RAG context (not using input_text)."""
        from mvp.utils import extract_first_json_object

        if self.llm_client.mode == "mock":
            # Mock mode: use persona domain for template-based answer
            point_texts = self.llm_client._mock_answer(persona, query, context, max_points)
        else:
            user_prompt = _build_answer_prompt_rag(query, context, max_points)
            response = self.llm_client._chat_completion(
                system_prompt=persona.system_prompt,
                user_prompt=user_prompt,
                trace={"phase": "stage1_answer_rag", "node_id": persona.node_id},
            )
            obj = extract_first_json_object(response)
            abstain_raw = obj.get("abstain", False)
            abstain = False
            if isinstance(abstain_raw, bool):
                abstain = abstain_raw
            elif isinstance(abstain_raw, str):
                abstain = abstain_raw.strip().lower() in {"true", "1", "yes"}
            if abstain:
                point_texts = []
            else:
                field = obj.get("points", [])
                point_texts = self.llm_client._normalize_point_texts(field, max_points=max_points)

        # Dedupe and cap
        point_texts = [pt[:160].strip() for pt in point_texts if pt and pt.strip()]
        point_texts = [pt for pt in point_texts if not self.llm_client._is_abstain_point(pt)]
        point_texts = dedupe_keep_order(point_texts)
        if max_points and max_points > 0:
            point_texts = point_texts[:max_points]

        points = []
        for idx, text in enumerate(point_texts):
            pid = self.llm_client._stable_seed(f"{persona.node_id}|{idx}|{text}")
            points.append(
                __import__("mvp.protocol", fromlist=["Point"]).Point(
                    point_id=f"{persona.node_id}:{idx}:{hex(pid)[2:10]}",
                    text=text,
                )
            )
        return WorkerAnswer(node_id=persona.node_id, points=points)

    def _llm_synthesize_rag(
        self,
        persona: Persona,
        bundle: list[WorkerAnswer],
        query: str,
        context: str,
        max_used_points: int | None,
    ) -> SynthesisDraft:
        """Synthesize using bundle of points and RAG context from the node's own document."""
        from mvp.utils import extract_first_json_object

        points_map: dict[str, str] = {}
        for ans in bundle:
            for p in ans.points:
                points_map[p.point_id] = p.text

        if self.llm_client.mode == "mock":
            synthesis_text, used = self.llm_client._mock_synthesis(persona, query, points_map, max_used_points)
        else:
            synthesis_text, used = self._openai_synthesize_rag(
                persona=persona,
                query=query,
                points_map=points_map,
                max_used_points=max_used_points,
                context=context,
            )

        used = [pid for pid in dedupe_keep_order(used) if pid in points_map]
        if max_used_points and max_used_points > 0:
            used = used[:max_used_points]
        if not synthesis_text.strip():
            synthesis_text = "UNKNOWN"
        if synthesis_text.strip().upper() != "UNKNOWN" and not used:
            synthesis_text = "UNKNOWN"

        return SynthesisDraft(node_id=persona.node_id, synthesis_text=synthesis_text.strip(), used_points=used)

    def _openai_synthesize_rag(
        self,
        persona: Persona,
        query: str,
        points_map: dict[str, str],
        max_used_points: int | None,
        context: str,
    ) -> tuple[str, list[str]]:
        synthesis_text, used = self._synthesize_attempt(
            persona=persona,
            query=query,
            points_map=points_map,
            max_used_points=max_used_points,
            context=context,
            repair=None,
            trace_phase="stage2_synthesize_rag",
        )
        if synthesis_text.upper() != "UNKNOWN" and not used:
            synthesis_text, used = self._synthesize_attempt(
                persona=persona,
                query=query,
                points_map=points_map,
                max_used_points=max_used_points,
                context=context,
                repair=(
                    "Contract violation: synthesis_text was not UNKNOWN but used_points was empty. "
                    "Return corrected JSON or synthesis_text='UNKNOWN'."
                ),
                trace_phase="stage2_synthesize_rag_retry",
            )
        if synthesis_text.upper() != "UNKNOWN" and not used:
            return "UNKNOWN", []
        return synthesis_text, used

    def _synthesize_attempt(
        self,
        persona: Persona,
        query: str,
        points_map: dict[str, str],
        max_used_points: int | None,
        context: str,
        repair: str | None,
        trace_phase: str,
    ) -> tuple[str, list[str]]:
        from mvp.utils import extract_first_json_object
        user_prompt = _build_synthesize_prompt_rag(query, points_map, context, max_used_points, repair)
        response = self.llm_client._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            trace={"phase": trace_phase, "node_id": persona.node_id},
        )
        obj = extract_first_json_object(response)
        synthesis_text = str(obj.get("synthesis_text", "UNKNOWN")).strip() or "UNKNOWN"
        raw_ids = obj.get("used_points", [])
        used = []
        if isinstance(raw_ids, list):
            for x in raw_ids:
                if isinstance(x, str) and x in points_map:
                    used.append(x)
        used = dedupe_keep_order(used)
        if max_used_points and max_used_points > 0:
            used = used[:max_used_points]
        return synthesis_text, used

    def _llm_grade_batch_rag(
        self,
        persona: Persona,
        query: str,
        context: str,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
    ) -> list[GradeVote]:
        """Grade batch using RAG context."""
        from mvp.utils import extract_first_json_object

        if self.llm_client.mode == "mock":
            return self.llm_client._mock_grade_batch(persona, query, context, targets, points_map)

        target_payload = [
            {
                "target_id": t.node_id,
                "target_draft": t.synthesis_text,
                "target_used_points": list(t.used_points),
            }
            for t in targets
        ]
        user_prompt = _build_grade_batch_prompt_rag(query, context, target_payload, points_map)
        response = self.llm_client._chat_completion(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            trace={"phase": "stage3_grade_batch_rag", "node_id": persona.node_id, "target_count": len(targets)},
        )
        obj = extract_first_json_object(response)
        votes_field = obj.get("votes", [])
        normalized_items = []
        if isinstance(votes_field, list):
            normalized_items = [e for e in votes_field if isinstance(e, dict)]
        elif isinstance(votes_field, dict):
            for tid, detail in votes_field.items():
                if isinstance(detail, dict):
                    item = dict(detail)
                    item["target_id"] = str(tid)
                    normalized_items.append(item)

        votes = []
        for entry in normalized_items:
            tid = str(entry.get("target_id", "")).strip()
            if not tid:
                continue
            valid = str(entry.get("valid", "UNKNOWN")).upper()
            agree = entry.get("agree_points", [])
            reject = entry.get("reject_points", [])
            unknown = entry.get("unknown_points", [])
            note = str(entry.get("note", "")).strip()
            votes.append(
                GradeVote(
                    grader_id=persona.node_id,
                    target_id=tid,
                    valid=(valid if valid in {"PASS", "FAIL", "UNKNOWN"} else "UNKNOWN"),
                    agree_points=[str(x) for x in agree] if isinstance(agree, list) else [],
                    reject_points=[str(x) for x in reject] if isinstance(reject, list) else [],
                    unknown_points=[str(x) for x in unknown] if isinstance(unknown, list) else [],
                    note=note,
                )
            )
        return votes

    # ---- Helpers ----

    def _select_workers(self) -> list[RAGPersona]:
        return self.rag_personas[: self.k]

    def _emit(self, progress: Callable[[str], None] | None, message: str) -> None:
        if progress:
            progress(message)

    def _is_active_draft(self, draft: SynthesisDraft) -> bool:
        return draft.synthesis_text.strip().upper() != "UNKNOWN" and bool(draft.used_points)

    def _write_logs(
        self,
        query: str,
        selected: list[RAGPersona],
        answers: list[WorkerAnswer],
        citations_map: dict[str, list[Citation]],
        drafts: list[SynthesisDraft],
        grades: list[GradeVote],
        final_output: FinalizationOutput,
        session_dir: Path,
        query_label: str,
    ) -> Path:
        session_dir = ensure_dir(session_dir)
        query_dir = ensure_dir(session_dir / query_label)

        save_json(query_dir / "stage1_answers.json", [a.to_dict() for a in answers])
        save_json(query_dir / "stage2_drafts.json", [d.to_dict() for d in drafts])
        save_json(query_dir / "stage3_grades.json", [g.to_dict() for g in grades])
        save_json(query_dir / "stage4_final.json", final_output.to_dict())
        save_json(query_dir / "llm_raw_responses.json", self.llm_client.get_raw_events())

        citations_json = {k: [c.model_dump() for c in v] for k, v in citations_map.items()}
        save_json(query_dir / "stage1_citations.json", citations_json)

        run_meta: dict[str, Any] = {
            "query": query,
            "mode": self.llm_client.mode,
            "k": self.k,
            "tau_fail": self.tau_fail,
            "selected_nodes": [
                {
                    "node_id": rp.node_id,
                    "domain": rp.domain,
                    "style": rp.style,
                    "scope": rp.scope,
                    "doc_path": str(rp.rag_node.doc_path),
                    "rep_snapshot": rp.rep_snapshot,
                }
                for rp in selected
            ],
        }
        save_json(query_dir / "run_meta.json", run_meta)

        # Write session index (list of all queries in this session)
        session_meta_path = session_dir / "session_meta.json"
        existing_meta = {}
        if session_meta_path.exists():
            try:
                existing_meta = json.loads(session_meta_path.read_text())
            except Exception:
                existing_meta = {}
        if "queries" not in existing_meta:
            existing_meta["queries"] = []
        # Avoid duplicates on retry
        existing_meta["queries"] = [
            q for q in existing_meta["queries"]
            if q.get("label") != query_label
        ]
        existing_meta["queries"].append({
            "label": query_label,
            "query": query,
            "winner": final_output.winner,
            "q_dir": str(query_dir.relative_to(session_dir)),
        })
        session_meta_path.write_text(json.dumps(existing_meta, indent=2))

        return query_dir


def _build_grade_batch_prompt_rag(
    query: str,
    context: str,
    targets: list[dict[str, Any]],
    points_map: dict[str, str],
) -> str:
    import json
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
        "- Use ONLY the provided context to evaluate drafts.\n"
        "- Include exactly one vote per target.\n"
        "- FAIL only on clear errors from context.\n"
        "- UNKNOWN if insufficient context.\n"
        "- Only classify IDs present in each target's used_points.\n"
        "- Every target_used_points ID must appear in exactly one list.\n"
        f"CONTEXT:\n{context}\n\n"
        f"Input JSON:\n"
        f'{{"query": {json.dumps(query)}, "targets": {json.dumps(targets)}, "points_map": {json.dumps(points_map)}}}'
    )
