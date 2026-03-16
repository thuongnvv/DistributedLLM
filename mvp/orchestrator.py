from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from config import (
    FAIL_PENALTY,
    MAX_POINTS_PER_ANSWER,
    MAX_USED_POINTS,
    POINT_SCORE_WEIGHT,
    WIN_BONUS,
)
from llm_client import LLMClient
from prompts import Persona
from protocol import (
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
from scoring import finalize_output
from utils import ensure_dir, run_timestamp, save_json


class MVPOrchestrator:
    def __init__(
        self,
        personas: list[Persona],
        llm_client: LLMClient,
        k: int,
        tau_fail: int,
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
        if tau_fail < 0:
            raise ValueError("tau_fail must be >= 0")
        if k > len(personas):
            raise ValueError(f"k={k} exceeds available personas={len(personas)}")

        self.personas = personas
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

    def run(
        self,
        query: str,
        input_text: str | None = None,
        case_id: str | None = None,
        progress: Callable[[str], None] | None = None,
        log_dir_override: Path | None = None,
    ) -> tuple[FinalizationOutput, Path]:
        if not query.strip():
            raise ValueError("Query must be non-empty")

        case_label = case_id or "ad-hoc"
        self.llm_client.reset_raw_events()
        selected, _routing_scores = self._select_workers()
        self._emit_progress(progress, f"[{case_label}] selected_nodes={','.join(p.node_id for p in selected)}")

        # Stage 1: independent answering
        self._emit_progress(progress, f"[{case_label}] stage=1/4 phase=answer start")
        answers: list[WorkerAnswer] = []
        for persona in selected:
            self._emit_progress(progress, f"[{case_label}] stage=1/4 node={persona.node_id} answering")
            answer = self.llm_client.answer(
                persona=persona,
                query=query,
                input_text=input_text,
                max_points=self.max_points_per_answer,
            )
            validate_worker_answer(answer, max_points=self.max_points_per_answer)
            answers.append(answer)
        self._emit_progress(progress, f"[{case_label}] stage=1/4 phase=answer done")

        points_map: dict[str, str] = {}
        point_owner_map: dict[str, str] = {}
        for ans in answers:
            for point in ans.points:
                points_map[point.point_id] = point.text
                point_owner_map[point.point_id] = ans.node_id
        active_nodes = {ans.node_id for ans in answers if ans.points}
        abstain_nodes = [p.node_id for p in selected if p.node_id not in active_nodes]
        if abstain_nodes:
            self._emit_progress(progress, f"[{case_label}] abstain_nodes_stage1={','.join(abstain_nodes)}")

        known_point_ids = set(points_map.keys())

        # Stage 2: distributed synthesis
        self._emit_progress(progress, f"[{case_label}] stage=2/4 phase=synthesize start")
        drafts: list[SynthesisDraft] = []
        for persona in selected:
            if persona.node_id not in active_nodes:
                self._emit_progress(
                    progress,
                    f"[{case_label}] stage=2/4 node={persona.node_id} abstain_synthesis(no in-scope points)",
                )
                draft = SynthesisDraft(
                    node_id=persona.node_id,
                    synthesis_text="UNKNOWN",
                    used_points=[],
                )
                validate_synthesis_draft(draft, known_point_ids)
                drafts.append(draft)
                continue

            self._emit_progress(progress, f"[{case_label}] stage=2/4 node={persona.node_id} synthesizing")
            draft = self.llm_client.synthesize(
                persona=persona,
                bundle=answers,
                query=query,
                input_text=input_text,
                max_used_points=self.max_used_points,
            )
            draft.used_points = normalize_used_points(draft.used_points, known_point_ids)
            validate_synthesis_draft(draft, known_point_ids)
            drafts.append(draft)
        self._emit_progress(progress, f"[{case_label}] stage=2/4 phase=synthesize done")

        # Stage 3: cross-grading
        self._emit_progress(progress, f"[{case_label}] stage=3/4 phase=grade start")
        grades: list[GradeVote] = []
        draft_by_node = {d.node_id: d for d in drafts}
        for grader in selected:
            target_drafts = [draft_by_node[node.node_id] for node in selected if node.node_id != grader.node_id]
            if grader.node_id not in active_nodes:
                self._emit_progress(
                    progress,
                    f"[{case_label}] stage=3/4 grader={grader.node_id} abstain_grade(no in-scope points)",
                )
                for target_draft in target_drafts:
                    vote = GradeVote(
                        grader_id=grader.node_id,
                        target_id=target_draft.node_id,
                        valid="UNKNOWN",
                        agree_points=[],
                        reject_points=[],
                        unknown_points=list(target_draft.used_points),
                        note="Grader abstained: no in-scope points in stage1",
                    )
                    validate_grade_vote(vote, target_draft.used_points)
                    grades.append(vote)
                continue

            self._emit_progress(progress, f"[{case_label}] stage=3/4 grader={grader.node_id} grading_batch")
            batch_votes = self.llm_client.grade_batch(
                persona=grader,
                query=query,
                input_text=input_text,
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
                normalized_vote = normalize_grade_vote(vote, target_draft.used_points)
                validate_grade_vote(normalized_vote, target_draft.used_points)
                vote_map[vote.target_id] = normalized_vote

            for target_draft in target_drafts:
                vote = vote_map.get(target_draft.node_id)
                if vote is None:
                    vote = GradeVote(
                        grader_id=grader.node_id,
                        target_id=target_draft.node_id,
                        valid="UNKNOWN",
                        agree_points=[],
                        reject_points=[],
                        unknown_points=list(target_draft.used_points),
                        note="Missing target in batch response",
                    )
                    validate_grade_vote(vote, target_draft.used_points)
                grades.append(vote)
        self._emit_progress(progress, f"[{case_label}] stage=3/4 phase=grade done")

        # Stage 4: finalization + reputation updates
        self._emit_progress(progress, f"[{case_label}] stage=4/4 phase=finalize start")
        final_output = finalize_output(
            drafts=drafts,
            grades=grades,
            point_owner_map=point_owner_map,
            tau_fail=self.tau_fail,
            win_bonus=self.win_bonus,
            fail_penalty=self.fail_penalty,
            point_score_weight=self.point_score_weight,
        )
        self._emit_progress(progress, f"[{case_label}] stage=4/4 phase=finalize done winner={final_output.winner}")

        log_dir = self._write_logs(
            query=query,
            input_text=input_text,
            case_id=case_id,
            selected=selected,
            answers=answers,
            drafts=drafts,
            grades=grades,
            final_output=final_output,
            log_dir_override=log_dir_override,
        )

        return final_output, log_dir

    def _write_logs(
        self,
        query: str,
        input_text: str | None,
        case_id: str | None,
        selected: list[Persona],
        answers: list[WorkerAnswer],
        drafts: list[SynthesisDraft],
        grades: list[GradeVote],
        final_output: FinalizationOutput,
        log_dir_override: Path | None = None,
    ) -> Path:
        if log_dir_override is not None:
            run_dir = ensure_dir(log_dir_override)
        else:
            run_dir = ensure_dir(Path(self.logs_root) / f"run_{run_timestamp()}")

        save_json(run_dir / "stage1_answers.json", [a.to_dict() for a in answers])
        save_json(run_dir / "stage2_drafts.json", [d.to_dict() for d in drafts])
        save_json(run_dir / "stage3_grades.json", [g.to_dict() for g in grades])
        save_json(run_dir / "stage4_final.json", final_output.to_dict())
        save_json(run_dir / "llm_raw_responses.json", self.llm_client.get_raw_events())

        run_meta: dict[str, Any] = {
            "query": query,
            "input_text": input_text,
            "case_id": case_id,
            "mode": self.llm_client.mode,
            "k": self.k,
            "tau_fail": self.tau_fail,
            "selection_strategy": "fixed_first_k",
            "selected_nodes": [
                {
                    "node_id": p.node_id,
                    "domain": p.domain,
                    "style": p.style,
                    "scope": p.scope,
                    "rep_snapshot": p.rep_snapshot,
                }
                for p in selected
            ],
        }
        save_json(run_dir / "run_meta.json", run_meta)

        return run_dir

    def _select_workers(self) -> tuple[list[Persona], dict[str, float]]:
        selected = self.personas[: self.k]
        return selected, {}

    def _emit_progress(self, progress: Callable[[str], None] | None, message: str) -> None:
        if progress is not None:
            progress(message)
