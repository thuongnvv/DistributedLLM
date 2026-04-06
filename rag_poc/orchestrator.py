"""
RAGOrchestrator: extends MVPOrchestrator with RAG retrieval for Stage 1 and Stage 3.

Each node has its own document + vector store. Both answer and grade phases
retrieve from the node's own store.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Sequence
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
    Point as MVPPoint,
    SynthesisDraft,
    WorkerAnswer,
    normalize_grade_vote,
    normalize_used_points,
    validate_grade_vote,
    validate_synthesis_draft,
    validate_worker_answer,
)
from mvp.scoring import finalize_output
from mvp.utils import ensure_dir, run_timestamp, save_json, dedupe_keep_order, extract_first_json_object, generate_point_id

from rag.node import RAGNode
from rag.embedder import embed_query, embed_texts
from settings import (
    MAX_EVIDENCE_CHUNKS_PER_POINT,
    MAX_EXTERNAL_POINTS_GRADE,
    MAX_EXTERNAL_POINTS_SYNTH,
)
from models import (
    Citation,
    PointAdjudication,
    PointEvidenceRef,
    ReviewTrace,
    ReviewTraceItem,
    SynthesisAdjudication,
)


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


_CITATION_PATTERN = re.compile(r"\[([^\[\]]+:[^\[\]]+)\]")


def _build_answer_prompt_rag(query: str, context: str, max_points: int | None) -> str:
    max_rule = f"- Return at most {max_points} points.\n" if max_points is not None else "- Return as many valid points as possible.\n"
    return (
        f"QUERY: {query}\n\n"
        "TASK: Answer the QUERY using ONLY the CONTEXT below. If the CONTEXT does not contain relevant information, set abstain=true.\n"
        "Return EXACT JSON:\n"
        "{\n"
        '  "abstain": true|false,\n'
        '  "points": [\n'
        '    {"text": "string", "citations": ["doc_id:chunk_id", "..."]}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Answer ONLY from the CONTEXT. Do NOT use your own knowledge.\n"
        "- Each point must be atomic and at most 160 characters.\n"
        "- Do NOT include inline citation markers inside point.text.\n"
        "- Every point must include at least one citation from the CONTEXT using the string form doc_id:chunk_id.\n"
        "- If abstain=true, points must be empty.\n"
        f"{max_rule}"
        f"CONTEXT:\n{context}\n"
    )


def _build_grade_prompt_rag(
    query: str,
    context: str,
    target_draft_text: str,
    target_used_points: list[str],
    points_map: dict[str, str],
) -> str:
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
    external_evidence_map: dict[str, dict[str, Any]],
    max_used_points: int | None,
    repair_feedback: str | None = None,
) -> str:
    repair_rule = f"- Contract repair note: {repair_feedback}\n" if repair_feedback else ""
    max_rule = f"- Select at most {max_used_points} points.\n" if max_used_points is not None else "- Use as many relevant points as possible.\n"
    points_text = "\n".join(f"  {pid}: {ptext}" for pid, ptext in points_map.items())
    return (
        f"QUERY: {query}\n\n"
        "TASK: Synthesize a final answer to the QUERY using the provided POINTS.\n"
        "Return EXACT JSON:\n"
        "{\n"
        '  "synthesis_text": "string",\n'
        '  "used_points": ["point_id", "..."],\n'
        '  "point_support": [\n'
        '    {"point_id": "string", "decision": "LOCAL_SUPPORTED|EXTERNAL_SUPPORTED|REJECTED|UNKNOWN", "reason": "string"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Build synthesis_text from the POINTS below. Do NOT use your own knowledge.\n"
        "- First evaluate each point against your LOCAL CONTEXT.\n"
        "- If LOCAL CONTEXT is insufficient for a point, you may consult EXTERNAL EVIDENCE for that point.\n"
        "- Local contradiction is stronger than external support; do not use contradicted points.\n"
        "- Every point_id in POINTS must appear exactly once in point_support.\n"
        "- Every factual claim in synthesis_text must be traceable to a point_id in used_points.\n"
        "- Cite sources as [doc_id:chunk_id] in synthesis_text for every claim.\n"
        "- Avoid redundant points that repeat the same idea.\n"
        "- If synthesis_text is not 'UNKNOWN', used_points must not be empty.\n"
        "- Only point_support decisions LOCAL_SUPPORTED or EXTERNAL_SUPPORTED may appear in used_points.\n"
        "- If no point supports the answer, return synthesis_text='UNKNOWN' and used_points=[]; do NOT make up content.\n"
        f"{max_rule}"
        f"{repair_rule}"
        f"POINTS (all provided claims):\n{points_text}\n\n"
        f"LOCAL CONTEXT (your document):\n{context}\n\n"
        f"EXTERNAL EVIDENCE (origin citations for non-local points):\n{json.dumps(external_evidence_map, ensure_ascii=False)}\n"
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
        self.max_evidence_chunks_per_point = MAX_EVIDENCE_CHUNKS_PER_POINT
        self.max_external_points_synth = MAX_EXTERNAL_POINTS_SYNTH
        self.max_external_points_grade = MAX_EXTERNAL_POINTS_GRADE

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
        point_evidence_map: dict[str, PointEvidenceRef] = {}
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
            answer, point_evidence = self._llm_answer_rag(
                persona=mvp_persona,
                query=query,
                context=context,
                citations=citations,
                max_points=self.max_points_per_answer,
            )
            try:
                validate_worker_answer(answer, max_points=self.max_points_per_answer)
            except ValueError:
                answer = WorkerAnswer(node_id=rp.node_id, points=[])
                point_evidence = []

            for ref in point_evidence:
                point_evidence_map[ref.point_id] = ref
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
        point_evidence_map = {
            pid: ref for pid, ref in point_evidence_map.items() if pid in known_point_ids
        }
        query_point_scores = self._compute_query_point_scores(query=query, points_map=points_map)

        # ==================== Stage 2: Synthesis ====================
        self._emit(progress, "stage=2/4 phase=synthesize start")
        drafts: list[SynthesisDraft] = []
        synthesis_adjudications: list[SynthesisAdjudication] = []
        for rp in selected:
            if rp.node_id not in active_nodes:
                self._emit(progress, f"stage=2/4 node={rp.node_id} abstain_synthesis")
                draft = SynthesisDraft(node_id=rp.node_id, synthesis_text="UNKNOWN", used_points=[])
                drafts.append(draft)
                synthesis_adjudications.append(SynthesisAdjudication(node_id=rp.node_id, point_support=[]))
                continue

            # Retrieve RAG context from this node's own document
            self._emit(progress, f"stage=2/4 node={rp.node_id} retrieving")
            synthesis_citations = rp.rag_node.retrieve(query)
            synthesis_context = rp.rag_node.build_rag_context(synthesis_citations)
            candidate_points_map = self._build_synthesis_candidate_points(
                node_id=rp.node_id,
                points_map=points_map,
                point_owner_map=point_owner_map,
                query_point_scores=query_point_scores,
            )
            external_evidence_map = self._build_external_evidence_map(
                point_ids=[
                    pid for pid in candidate_points_map
                    if point_owner_map.get(pid) != rp.node_id
                ],
                point_evidence_map=point_evidence_map,
            )

            self._emit(progress, f"stage=2/4 node={rp.node_id} synthesizing")
            draft, adjudication = self._llm_synthesize_rag(
                persona=rp.to_mvp_persona(),
                query=query,
                context=synthesis_context,
                candidate_points_map=candidate_points_map,
                external_evidence_map=external_evidence_map,
                max_used_points=self.max_used_points,
            )
            draft.used_points = normalize_used_points(draft.used_points, known_point_ids)
            adjudication = self._normalize_synthesis_adjudication(
                node_id=rp.node_id,
                point_support=adjudication.point_support,
                candidate_point_ids=list(candidate_points_map.keys()),
            )
            allowed_support_ids = {
                item.point_id
                for item in adjudication.point_support
                if item.decision in {"LOCAL_SUPPORTED", "EXTERNAL_SUPPORTED"}
            }
            draft.used_points = [pid for pid in draft.used_points if pid in allowed_support_ids]
            try:
                validate_synthesis_draft(draft, known_point_ids)
            except ValueError:
                draft = SynthesisDraft(node_id=rp.node_id, synthesis_text="UNKNOWN", used_points=[])
                adjudication = self._coerce_unknown_synthesis_adjudication(
                    node_id=rp.node_id,
                    candidate_point_ids=list(candidate_points_map.keys()),
                    reason="Draft failed validation",
                )
            drafts.append(draft)
            synthesis_adjudications.append(adjudication)
        self._emit(progress, "stage=2/4 phase=synthesize done")
        active_drafts = [d for d in drafts if self._is_active_draft(d)]
        draft_by_node = {d.node_id: d for d in drafts}

        # ==================== Stage 3: RAG Cross-grading ====================
        self._emit(progress, "stage=3/4 phase=grade start")
        grades: list[GradeVote] = []
        review_traces: list[ReviewTrace] = []
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
            batch_votes, batch_review_traces = self._llm_grade_batch_rag(
                persona=rp.to_mvp_persona(),
                query=query,
                context=context_str,
                targets=target_drafts,
                points_map=points_map,
                point_owner_map=point_owner_map,
                point_evidence_map=point_evidence_map,
            )

            vote_map: dict[str, GradeVote] = {}
            trace_map: dict[str, ReviewTrace] = {}
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

            for trace in batch_review_traces:
                if trace.target_id in trace_map:
                    continue
                trace_map[trace.target_id] = self._normalize_review_trace(
                    grader_id=rp.node_id,
                    target_id=trace.target_id,
                    point_reviews=trace.point_reviews,
                    target_used_points=list(draft_by_node.get(trace.target_id, SynthesisDraft(node_id="", synthesis_text="UNKNOWN", used_points=[])).used_points),
                )

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
                review_trace = trace_map.get(target_draft.node_id)
                if review_trace is None:
                    review_trace = self._build_default_review_trace(
                        grader_id=rp.node_id,
                        target_id=target_draft.node_id,
                        point_ids=list(target_draft.used_points),
                        basis="UNKNOWN",
                        reason="Missing target in batch response",
                    )
                grades.append(vote)
                review_traces.append(review_trace)
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
            point_evidence_map=point_evidence_map,
            drafts=drafts,
            synthesis_adjudications=synthesis_adjudications,
            grades=grades,
            review_traces=review_traces,
            final_output=final_output,
            session_dir=session_dir,
            query_label=query_label,
        )

        # Build extra data for UI
        extra: dict[str, Any] = {
            "citations_map": {k: [c.model_dump() for c in v] for k, v in citations_map.items()},
            "point_evidence_map": {k: ref.model_dump() for k, ref in point_evidence_map.items()},
            "synthesis_adjudications": [item.model_dump() for item in synthesis_adjudications],
            "review_trace": [item.model_dump() for item in review_traces],
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
        citations: list[Citation],
        max_points: int | None,
    ) -> tuple[WorkerAnswer, list[PointEvidenceRef]]:
        """Call LLM answer with RAG context (not using input_text)."""
        citation_lookup = self._build_citation_lookup(citations)
        normalized_points: list[tuple[str, list[str]]] = []

        if self.llm_client.mode == "mock":
            point_texts = self.llm_client._mock_answer(persona, query, context, max_points)
            available_refs = list(citation_lookup.keys())
            for idx, text in enumerate(point_texts):
                if not available_refs:
                    break
                cleaned = self._strip_inline_citations(text)
                if not cleaned:
                    continue
                normalized_points.append((cleaned, [available_refs[idx % len(available_refs)]]))
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
                normalized_points = []
            else:
                normalized_points = self._normalize_answer_points(
                    obj.get("points", []),
                    citation_lookup=citation_lookup,
                    max_points=max_points,
                )

        points: list[MVPPoint] = []
        point_evidence: list[PointEvidenceRef] = []
        for idx, (text, citation_refs) in enumerate(normalized_points):
            if self.llm_client._is_abstain_point(text):
                continue
            evidence_chunks = [
                citation_lookup[ref]
                for ref in citation_refs
                if ref in citation_lookup
            ][: self.max_evidence_chunks_per_point]
            if not evidence_chunks:
                continue
            point_id = generate_point_id(persona.node_id, len(points), text)
            points.append(MVPPoint(point_id=point_id, text=text))
            point_evidence.append(
                PointEvidenceRef(
                    point_id=point_id,
                    origin_node_id=persona.node_id,
                    text=text,
                    citations=[self._citation_ref(chunk) for chunk in evidence_chunks],
                    evidence_chunks=evidence_chunks,
                )
            )

        return WorkerAnswer(node_id=persona.node_id, points=points), point_evidence

    def _llm_synthesize_rag(
        self,
        persona: Persona,
        query: str,
        context: str,
        candidate_points_map: dict[str, str],
        external_evidence_map: dict[str, dict[str, Any]],
        max_used_points: int | None,
    ) -> tuple[SynthesisDraft, SynthesisAdjudication]:
        """Synthesize using bundle of points and RAG context from the node's own document."""
        if self.llm_client.mode == "mock":
            synthesis_text, used, point_support = self._mock_synthesize_with_evidence(
                persona=persona,
                query=query,
                candidate_points_map=candidate_points_map,
                external_evidence_map=external_evidence_map,
                max_used_points=max_used_points,
            )
        else:
            synthesis_text, used, point_support = self._openai_synthesize_rag(
                persona=persona,
                query=query,
                points_map=candidate_points_map,
                external_evidence_map=external_evidence_map,
                max_used_points=max_used_points,
                context=context,
            )

        used = [pid for pid in dedupe_keep_order(used) if pid in candidate_points_map]
        if max_used_points and max_used_points > 0:
            used = used[:max_used_points]
        if not synthesis_text.strip():
            synthesis_text = "UNKNOWN"
        if synthesis_text.strip().upper() != "UNKNOWN" and not used:
            synthesis_text = "UNKNOWN"

        return (
            SynthesisDraft(node_id=persona.node_id, synthesis_text=synthesis_text.strip(), used_points=used),
            SynthesisAdjudication(node_id=persona.node_id, point_support=point_support),
        )

    def _openai_synthesize_rag(
        self,
        persona: Persona,
        query: str,
        points_map: dict[str, str],
        external_evidence_map: dict[str, dict[str, Any]],
        max_used_points: int | None,
        context: str,
    ) -> tuple[str, list[str], list[PointAdjudication]]:
        synthesis_text, used, point_support = self._synthesize_attempt(
            persona=persona,
            query=query,
            points_map=points_map,
            external_evidence_map=external_evidence_map,
            max_used_points=max_used_points,
            context=context,
            repair=None,
            trace_phase="stage2_synthesize_rag",
        )
        if synthesis_text.upper() != "UNKNOWN" and not used:
            synthesis_text, used, point_support = self._synthesize_attempt(
                persona=persona,
                query=query,
                points_map=points_map,
                external_evidence_map=external_evidence_map,
                max_used_points=max_used_points,
                context=context,
                repair=(
                    "Contract violation: synthesis_text was not UNKNOWN but used_points was empty. "
                    "Return corrected JSON or synthesis_text='UNKNOWN'."
                ),
                trace_phase="stage2_synthesize_rag_retry",
            )
        if synthesis_text.upper() != "UNKNOWN" and not used:
            return "UNKNOWN", [], []
        return synthesis_text, used, point_support

    def _synthesize_attempt(
        self,
        persona: Persona,
        query: str,
        points_map: dict[str, str],
        external_evidence_map: dict[str, dict[str, Any]],
        max_used_points: int | None,
        context: str,
        repair: str | None,
        trace_phase: str,
    ) -> tuple[str, list[str], list[PointAdjudication]]:
        user_prompt = _build_synthesize_prompt_rag(
            query=query,
            points_map=points_map,
            context=context,
            external_evidence_map=external_evidence_map,
            max_used_points=max_used_points,
            repair_feedback=repair,
        )
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
        point_support = self._parse_point_adjudications(
            obj.get("point_support", []),
            candidate_point_ids=list(points_map.keys()),
        )
        return synthesis_text, used, point_support

    def _llm_grade_batch_rag(
        self,
        persona: Persona,
        query: str,
        context: str,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
        point_owner_map: dict[str, str],
        point_evidence_map: dict[str, PointEvidenceRef],
    ) -> tuple[list[GradeVote], list[ReviewTrace]]:
        """Grade batch using RAG context."""
        if self.llm_client.mode == "mock":
            return self._mock_grade_batch_with_evidence(
                persona=persona,
                query=query,
                targets=targets,
                points_map=points_map,
                point_owner_map=point_owner_map,
                point_evidence_map=point_evidence_map,
            )

        target_payload = [
            {
                "target_id": t.node_id,
                "target_draft": t.synthesis_text,
                "target_used_points": list(t.used_points),
                "external_evidence": self._build_external_evidence_map(
                    point_ids=self._cap_grade_point_ids(list(t.used_points)),
                    point_evidence_map=point_evidence_map,
                ),
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
        review_traces: list[ReviewTrace] = []
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
            review_traces.append(
                ReviewTrace(
                    grader_id=persona.node_id,
                    target_id=tid,
                    point_reviews=self._parse_review_trace_items(
                        entry.get("review_trace", []),
                        target_used_points=next((list(t.used_points) for t in targets if t.node_id == tid), []),
                    ),
                )
            )
        return votes, review_traces

    # ---- Helpers ----

    def _citation_ref(self, citation: Citation) -> str:
        return f"{citation.doc_id}:{citation.chunk_id}"

    def _build_citation_lookup(self, citations: Sequence[Citation]) -> dict[str, Citation]:
        lookup: dict[str, Citation] = {}
        for citation in citations:
            lookup[self._citation_ref(citation)] = citation
        return lookup

    def _extract_inline_citations(self, text: str) -> list[str]:
        return dedupe_keep_order(match.strip() for match in _CITATION_PATTERN.findall(text or "") if match.strip())

    def _strip_inline_citations(self, text: str) -> str:
        cleaned = _CITATION_PATTERN.sub("", text or "")
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()[:160]

    def _normalize_answer_points(
        self,
        points_field: Any,
        citation_lookup: dict[str, Citation],
        max_points: int | None,
    ) -> list[tuple[str, list[str]]]:
        normalized: list[tuple[str, list[str]]] = []
        seen_texts: set[str] = set()
        entries = points_field if isinstance(points_field, list) else []
        for entry in entries:
            text = ""
            citations: list[str] = []
            if isinstance(entry, dict):
                raw_text = entry.get("text", "")
                if isinstance(raw_text, str):
                    text = self._strip_inline_citations(raw_text)
                    citations = self._extract_inline_citations(raw_text)
                raw_citations = entry.get("citations", [])
                if isinstance(raw_citations, list):
                    citations.extend(str(item).strip() for item in raw_citations if str(item).strip())
            elif isinstance(entry, str):
                text = self._strip_inline_citations(entry)
                citations = self._extract_inline_citations(entry)

            citations = [
                ref for ref in dedupe_keep_order(citations)
                if ref in citation_lookup
            ][: self.max_evidence_chunks_per_point]
            if not text or not citations or text in seen_texts:
                continue
            seen_texts.add(text)
            normalized.append((text, citations))
            if max_points is not None and max_points > 0 and len(normalized) >= max_points:
                break
        return normalized

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _token_overlap_score(self, query: str, text: str) -> float:
        query_terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
        text_terms = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
        if not query_terms or not text_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)

    def _compute_query_point_scores(self, query: str, points_map: dict[str, str]) -> dict[str, float]:
        if not points_map:
            return {}
        point_ids = list(points_map.keys())
        point_texts = [points_map[pid] for pid in point_ids]
        try:
            query_embedding = embed_query(query)
            point_embeddings = embed_texts(point_texts)
            return {
                pid: self._cosine_similarity(query_embedding, embedding)
                for pid, embedding in zip(point_ids, point_embeddings)
            }
        except Exception:
            return {
                pid: self._token_overlap_score(query, points_map[pid])
                for pid in point_ids
            }

    def _build_synthesis_candidate_points(
        self,
        node_id: str,
        points_map: dict[str, str],
        point_owner_map: dict[str, str],
        query_point_scores: dict[str, float],
    ) -> dict[str, str]:
        own_ids = [pid for pid in points_map if point_owner_map.get(pid) == node_id]
        external_ids = [pid for pid in points_map if point_owner_map.get(pid) != node_id]
        external_ids.sort(key=lambda pid: (-query_point_scores.get(pid, 0.0), pid))
        selected_external = external_ids[: self.max_external_points_synth]
        selected_ids = own_ids + selected_external
        return {pid: points_map[pid] for pid in selected_ids}

    def _build_external_evidence_map(
        self,
        point_ids: Sequence[str],
        point_evidence_map: dict[str, PointEvidenceRef],
    ) -> dict[str, dict[str, Any]]:
        external: dict[str, dict[str, Any]] = {}
        for pid in point_ids:
            ref = point_evidence_map.get(pid)
            if ref is None:
                continue
            evidence_chunks = ref.evidence_chunks[: self.max_evidence_chunks_per_point]
            external[pid] = {
                "origin_node_id": ref.origin_node_id,
                "point_text": ref.text,
                "citations": list(ref.citations[: self.max_evidence_chunks_per_point]),
                "evidence_chunks": [
                    {
                        "citation": self._citation_ref(chunk),
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                    }
                    for chunk in evidence_chunks
                ],
            }
        return external

    def _parse_point_adjudications(
        self,
        raw: Any,
        candidate_point_ids: Sequence[str],
    ) -> list[PointAdjudication]:
        allowed_ids = set(candidate_point_ids)
        allowed_decisions = {"LOCAL_SUPPORTED", "EXTERNAL_SUPPORTED", "REJECTED", "UNKNOWN"}
        items = raw if isinstance(raw, list) else []
        parsed: list[PointAdjudication] = []
        seen: set[str] = set()
        for entry in items:
            if not isinstance(entry, dict):
                continue
            point_id = str(entry.get("point_id", "")).strip()
            if not point_id or point_id not in allowed_ids or point_id in seen:
                continue
            decision = str(entry.get("decision", "UNKNOWN")).upper()
            if decision not in allowed_decisions:
                decision = "UNKNOWN"
            reason = str(entry.get("reason", "")).strip()
            parsed.append(PointAdjudication(point_id=point_id, decision=decision, reason=reason))
            seen.add(point_id)
        return parsed

    def _normalize_synthesis_adjudication(
        self,
        node_id: str,
        point_support: Sequence[PointAdjudication],
        candidate_point_ids: Sequence[str],
    ) -> SynthesisAdjudication:
        support_map = {item.point_id: item for item in point_support}
        normalized: list[PointAdjudication] = []
        for pid in candidate_point_ids:
            normalized.append(
                support_map.get(pid, PointAdjudication(point_id=pid, decision="UNKNOWN", reason="No adjudication returned"))
            )
        return SynthesisAdjudication(node_id=node_id, point_support=normalized)

    def _coerce_unknown_synthesis_adjudication(
        self,
        node_id: str,
        candidate_point_ids: Sequence[str],
        reason: str,
    ) -> SynthesisAdjudication:
        return SynthesisAdjudication(
            node_id=node_id,
            point_support=[
                PointAdjudication(point_id=pid, decision="UNKNOWN", reason=reason)
                for pid in candidate_point_ids
            ],
        )

    def _mock_synthesize_with_evidence(
        self,
        persona: Persona,
        query: str,
        candidate_points_map: dict[str, str],
        external_evidence_map: dict[str, dict[str, Any]],
        max_used_points: int | None,
    ) -> tuple[str, list[str], list[PointAdjudication]]:
        point_ids = list(candidate_points_map.keys())
        own_ids = [pid for pid in point_ids if pid.startswith(f"{persona.node_id}:")]
        external_ids = [pid for pid in point_ids if pid not in own_ids]
        limit = max_used_points if max_used_points is not None and max_used_points > 0 else len(point_ids)
        used = own_ids[:limit]
        if len(used) < limit and external_ids:
            used.append(external_ids[0])
        used = dedupe_keep_order(used)[:limit]

        point_support: list[PointAdjudication] = []
        for pid in point_ids:
            if pid in used:
                if pid in own_ids:
                    decision = "LOCAL_SUPPORTED"
                    reason = "Selected because local context supports this point."
                else:
                    decision = "EXTERNAL_SUPPORTED"
                    reason = "Selected because origin evidence was available."
            else:
                decision = "UNKNOWN"
                reason = "Not selected for the final draft."
            point_support.append(PointAdjudication(point_id=pid, decision=decision, reason=reason))

        if not used:
            return "UNKNOWN", [], point_support

        snippets = [candidate_points_map[pid] for pid in used]
        synthesis_text = " ".join(snippets)
        return synthesis_text, used, point_support

    def _cap_grade_point_ids(self, point_ids: Sequence[str]) -> list[str]:
        if self.max_external_points_grade <= 0:
            return list(point_ids)
        return list(point_ids)[: self.max_external_points_grade]

    def _parse_review_trace_items(
        self,
        raw: Any,
        target_used_points: Sequence[str],
    ) -> list[ReviewTraceItem]:
        allowed_ids = set(target_used_points)
        allowed_basis = {"LOCAL_SUPPORTED", "EXTERNAL_SUPPORTED", "CONTRADICTED", "UNKNOWN"}
        items = raw if isinstance(raw, list) else []
        parsed: list[ReviewTraceItem] = []
        seen: set[str] = set()
        for entry in items:
            if not isinstance(entry, dict):
                continue
            point_id = str(entry.get("point_id", "")).strip()
            if not point_id or point_id not in allowed_ids or point_id in seen:
                continue
            basis = str(entry.get("basis", "UNKNOWN")).upper()
            if basis not in allowed_basis:
                basis = "UNKNOWN"
            reason = str(entry.get("reason", "")).strip()
            parsed.append(ReviewTraceItem(point_id=point_id, basis=basis, reason=reason))
            seen.add(point_id)
        return parsed

    def _normalize_review_trace(
        self,
        grader_id: str,
        target_id: str,
        point_reviews: Sequence[ReviewTraceItem],
        target_used_points: Sequence[str],
    ) -> ReviewTrace:
        review_map = {item.point_id: item for item in point_reviews}
        normalized = [
            review_map.get(pid, ReviewTraceItem(point_id=pid, basis="UNKNOWN", reason="No trace returned"))
            for pid in target_used_points
        ]
        return ReviewTrace(grader_id=grader_id, target_id=target_id, point_reviews=normalized)

    def _build_default_review_trace(
        self,
        grader_id: str,
        target_id: str,
        point_ids: Sequence[str],
        basis: str,
        reason: str,
    ) -> ReviewTrace:
        return ReviewTrace(
            grader_id=grader_id,
            target_id=target_id,
            point_reviews=[
                ReviewTraceItem(point_id=pid, basis=basis, reason=reason)
                for pid in point_ids
            ],
        )

    def _mock_grade_batch_with_evidence(
        self,
        persona: Persona,
        query: str,
        targets: list[SynthesisDraft],
        points_map: dict[str, str],
        point_owner_map: dict[str, str],
        point_evidence_map: dict[str, PointEvidenceRef],
    ) -> tuple[list[GradeVote], list[ReviewTrace]]:
        votes: list[GradeVote] = []
        traces: list[ReviewTrace] = []
        for target in targets:
            capped_external = set(self._cap_grade_point_ids(target.used_points))
            agree: list[str] = []
            reject: list[str] = []
            unknown: list[str] = []
            point_reviews: list[ReviewTraceItem] = []

            for pid in target.used_points:
                text = points_map.get(pid, "")
                owner = point_owner_map.get(pid)
                if self.llm_client._contains_clear_error(text):
                    reject.append(pid)
                    point_reviews.append(ReviewTraceItem(point_id=pid, basis="CONTRADICTED", reason="Clear contradiction heuristic"))
                elif owner == persona.node_id:
                    agree.append(pid)
                    point_reviews.append(ReviewTraceItem(point_id=pid, basis="LOCAL_SUPPORTED", reason="Point originated from the grader node"))
                elif pid in capped_external and pid in point_evidence_map and point_evidence_map[pid].evidence_chunks:
                    agree.append(pid)
                    point_reviews.append(ReviewTraceItem(point_id=pid, basis="EXTERNAL_SUPPORTED", reason="Origin evidence available for this point"))
                else:
                    unknown.append(pid)
                    point_reviews.append(ReviewTraceItem(point_id=pid, basis="UNKNOWN", reason="No local or external evidence available in mock mode"))

            if reject:
                valid = "FAIL"
                note = "At least one point was contradicted."
            elif agree and not unknown:
                valid = "PASS"
                note = "All points supported by local or external evidence."
            else:
                valid = "UNKNOWN"
                note = "Some points could not be verified."

            votes.append(
                GradeVote(
                    grader_id=persona.node_id,
                    target_id=target.node_id,
                    valid=valid,  # type: ignore[arg-type]
                    agree_points=agree,
                    reject_points=reject,
                    unknown_points=unknown,
                    note=note,
                )
            )
            traces.append(
                ReviewTrace(
                    grader_id=persona.node_id,
                    target_id=target.node_id,
                    point_reviews=point_reviews,
                )
            )
        return votes, traces

    def _serialize_stage1_answers(
        self,
        answers: Sequence[WorkerAnswer],
        point_evidence_map: dict[str, PointEvidenceRef],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for answer in answers:
            points_payload: list[dict[str, Any]] = []
            for point in answer.points:
                evidence = point_evidence_map.get(point.point_id)
                points_payload.append(
                    {
                        "point_id": point.point_id,
                        "text": point.text,
                        "citations": list(evidence.citations) if evidence else [],
                    }
                )
            payload.append({"node_id": answer.node_id, "points": points_payload})
        return payload

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
        point_evidence_map: dict[str, PointEvidenceRef],
        drafts: list[SynthesisDraft],
        synthesis_adjudications: list[SynthesisAdjudication],
        grades: list[GradeVote],
        review_traces: list[ReviewTrace],
        final_output: FinalizationOutput,
        session_dir: Path,
        query_label: str,
    ) -> Path:
        session_dir = ensure_dir(session_dir)
        query_dir = ensure_dir(session_dir / query_label)

        save_json(query_dir / "stage1_answers.json", self._serialize_stage1_answers(answers, point_evidence_map))
        save_json(query_dir / "stage2_drafts.json", [d.to_dict() for d in drafts])
        save_json(query_dir / "stage3_grades.json", [g.to_dict() for g in grades])
        save_json(query_dir / "stage4_final.json", final_output.to_dict())
        save_json(query_dir / "llm_raw_responses.json", self.llm_client.get_raw_events())

        citations_json = {k: [c.model_dump() for c in v] for k, v in citations_map.items()}
        save_json(query_dir / "stage1_citations.json", citations_json)
        save_json(
            query_dir / "stage1_point_evidence.json",
            {pid: ref.model_dump() for pid, ref in point_evidence_map.items()},
        )
        save_json(
            query_dir / "stage2_adjudications.json",
            [item.model_dump() for item in synthesis_adjudications],
        )
        save_json(
            query_dir / "stage3_review_trace.json",
            [item.model_dump() for item in review_traces],
        )

        run_meta: dict[str, Any] = {
            "query": query,
            "mode": self.llm_client.mode,
            "k": self.k,
            "tau_fail": self.tau_fail,
            "max_evidence_chunks_per_point": self.max_evidence_chunks_per_point,
            "max_external_points_synth": self.max_external_points_synth,
            "max_external_points_grade": self.max_external_points_grade,
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
        '      "review_trace": [\n'
        '        {"point_id": "string", "basis": "LOCAL_SUPPORTED|EXTERNAL_SUPPORTED|CONTRADICTED|UNKNOWN", "reason": "string"}\n'
        "      ],\n"
        '      "note": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use LOCAL CONTEXT first to evaluate each point in a draft.\n"
        "- If LOCAL CONTEXT is insufficient for a point, you may use the target's EXTERNAL EVIDENCE for that point.\n"
        "- Local contradiction is stronger than external support.\n"
        "- Include exactly one vote per target.\n"
        "- FAIL only on clear errors from context.\n"
        "- UNKNOWN only if both LOCAL CONTEXT and EXTERNAL EVIDENCE are insufficient.\n"
        "- Only classify IDs present in each target's used_points.\n"
        "- Every target_used_points ID must appear in exactly one list.\n"
        "- Every target_used_points ID must also appear exactly once in review_trace.\n"
        f"CONTEXT:\n{context}\n\n"
        f"Input JSON:\n"
        f'{{"query": {json.dumps(query)}, "targets": {json.dumps(targets)}, "points_map": {json.dumps(points_map)}}}'
    )
