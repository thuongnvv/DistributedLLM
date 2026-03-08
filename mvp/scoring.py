from __future__ import annotations

from collections import defaultdict

from protocol import FinalizationOutput, GradeVote, SynthesisDraft


def compute_metrics(node_ids: list[str], grades: list[GradeVote]) -> dict[str, dict[str, int]]:
    metrics: dict[str, dict[str, int]] = {
        "pass": {nid: 0 for nid in node_ids},
        "fail": {nid: 0 for nid in node_ids},
        "agree": {nid: 0 for nid in node_ids},
        "reject": {nid: 0 for nid in node_ids},
    }

    node_set = set(node_ids)
    for vote in grades:
        if vote.target_id not in node_set:
            continue
        if vote.valid == "PASS":
            metrics["pass"][vote.target_id] += 1
        elif vote.valid == "FAIL":
            metrics["fail"][vote.target_id] += 1

        metrics["agree"][vote.target_id] += len(vote.agree_points)
        metrics["reject"][vote.target_id] += len(vote.reject_points)

    return metrics


def select_winner(
    node_ids: list[str],
    metrics: dict[str, dict[str, int]],
    tau_fail: int,
) -> tuple[str, list[str]]:
    discarded = [nid for nid in node_ids if metrics["fail"][nid] >= tau_fail]
    survivors = [nid for nid in node_ids if nid not in discarded]

    def rank_key(nid: str) -> tuple[int, int, int, int, str]:
        return (
            metrics["pass"][nid],
            metrics["agree"][nid],
            -metrics["reject"][nid],
            -metrics["fail"][nid],
            nid,
        )

    candidates = survivors if survivors else node_ids
    winner = max(candidates, key=rank_key)
    return winner, discarded


def compute_reputation_updates(
    node_ids: list[str],
    grades: list[GradeVote],
    point_owner_map: dict[str, str],
    winner: str,
    metrics: dict[str, dict[str, int]],
    tau_fail: int,
    win_bonus: float,
    fail_penalty: float,
    point_score_weight: float,
) -> dict[str, dict[str, float]]:
    node_rep_delta: dict[str, float] = {nid: 0.0 for nid in node_ids}
    point_rep_delta: dict[str, float] = {nid: 0.0 for nid in node_ids}

    node_rep_delta[winner] += win_bonus

    for nid in node_ids:
        if metrics["fail"][nid] >= tau_fail:
            node_rep_delta[nid] -= fail_penalty

    point_scores: dict[str, int] = defaultdict(int)
    for vote in grades:
        for pid in vote.agree_points:
            point_scores[pid] += 1
        for pid in vote.reject_points:
            point_scores[pid] -= 1

    for pid, score in point_scores.items():
        owner = point_owner_map.get(pid)
        if owner is None:
            continue
        point_rep_delta[owner] += point_score_weight * float(score)

    node_rep_delta = {k: v for k, v in node_rep_delta.items() if abs(v) > 1e-12}
    point_rep_delta = {k: v for k, v in point_rep_delta.items() if abs(v) > 1e-12}

    return {
        "node_rep_delta": node_rep_delta,
        "point_rep_delta": point_rep_delta,
    }


def finalize_output(
    drafts: list[SynthesisDraft],
    grades: list[GradeVote],
    point_owner_map: dict[str, str],
    tau_fail: int,
    win_bonus: float,
    fail_penalty: float,
    point_score_weight: float,
) -> FinalizationOutput:
    if not drafts:
        raise ValueError("Cannot finalize without drafts")

    node_ids = [d.node_id for d in drafts]
    metrics = compute_metrics(node_ids=node_ids, grades=grades)
    winner, _discarded = select_winner(node_ids=node_ids, metrics=metrics, tau_fail=tau_fail)

    winner_draft = next((d for d in drafts if d.node_id == winner), None)
    if winner_draft is None:
        raise RuntimeError(f"Winner draft not found for node {winner}")

    rep_updates = compute_reputation_updates(
        node_ids=node_ids,
        grades=grades,
        point_owner_map=point_owner_map,
        winner=winner,
        metrics=metrics,
        tau_fail=tau_fail,
        win_bonus=win_bonus,
        fail_penalty=fail_penalty,
        point_score_weight=point_score_weight,
    )

    return FinalizationOutput(
        winner=winner,
        final_answer=winner_draft.synthesis_text,
        metrics=metrics,
        reputation_updates=rep_updates,
    )
