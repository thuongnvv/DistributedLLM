from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single citation extracted from a node's RAG retrieval."""
    doc_id: str
    chunk_id: str
    text: str
    score: float


class Point(BaseModel):
    """An atomic claim from a worker node."""
    point_id: str
    text: str


class WorkerAnswer(BaseModel):
    """Response from a node's RAG answering phase."""
    node_id: str
    abstain: bool = False
    points: list[Point] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class SynthesisDraft(BaseModel):
    """A node's synthesized draft from the bundle of points."""
    node_id: str
    synthesis_text: str
    used_points: list[str] = Field(default_factory=list)


class GradeVote(BaseModel):
    """A grader's evaluation of a target's draft."""
    grader_id: str
    target_id: str
    valid: str = "UNKNOWN"  # PASS, FAIL, UNKNOWN
    agree_points: list[str] = Field(default_factory=list)
    reject_points: list[str] = Field(default_factory=list)
    unknown_points: list[str] = Field(default_factory=list)
    note: str = ""


class FinalizationOutput(BaseModel):
    """Final output from the pipeline."""
    winner: str
    final_answer: str
    metrics: dict = Field(default_factory=dict)
    reputation_updates: dict = Field(default_factory=dict)


class NodeResponse(BaseModel):
    """Full response from a node including all stages."""
    node_id: str
    abstain: bool = False
    citations: list[Citation] = Field(default_factory=list)
    # Stage 1
    stage1_points: list[Point] = Field(default_factory=list)
    # Stage 2
    synthesis_text: str = ""
    used_points: list[str] = Field(default_factory=list)
    # Stage 3 (as grader)
    grade_votes: list[GradeVote] = Field(default_factory=list)
    # Stage 3 (as target, received from other nodes)
    received_votes: list[GradeVote] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """Complete result from the full pipeline."""
    query: str
    node_responses: dict[str, NodeResponse]
    winner: str
    final_answer: str
    citations: list[Citation] = Field(default_factory=list)
    metrics: dict = Field(default_factory=dict)
    reputation_updates: dict = Field(default_factory=dict)
    logs: dict = Field(default_factory=dict)  # stage logs for debug UI
