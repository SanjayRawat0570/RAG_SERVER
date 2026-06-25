"""Pydantic models describing a workflow definition (F1).

A workflow is a Directed Acyclic Graph of typed nodes connected by edges.
Edges may carry a condition (F2) so that branching is expressed declaratively.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    INPUT = "input"
    PROCESSING = "processing"
    DECISION = "decision"
    SWITCH = "switch"
    MERGE = "merge"
    EXTERNAL = "external"
    SUBWORKFLOW = "subworkflow"
    LOOP = "loop"
    INGEST = "ingest"
    CHUNK = "chunk"
    EMBED = "embed"
    UPSERT = "upsert"
    VECTOR_SEARCH = "vector_search"
    QUERY_PROCESS = "query_process"
    KEYWORD_SEARCH = "keyword_search"
    RERANK = "rerank"
    AUGMENT = "augment"
    GENERATE = "generate"
    DECOMPOSE = "decompose"
    SYNTHESIZE = "synthesize"
    ENTITY_SEARCH = "entity_search"
    CLASSIFY = "classify"
    OUTPUT = "output"


class RetryPolicy(BaseModel):
    """Retry with exponential backoff (F7 foundation)."""

    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: float = Field(default=0.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class NodeDef(BaseModel):
    id: str
    type: NodeType
    # Free-form per-node configuration; validated by each node implementation.
    config: dict[str, Any] = Field(default_factory=dict)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    # If a node fails after retries, fall back to this static output instead of
    # failing the whole workflow (F7).
    fallback: Any | None = None


class EdgeDef(BaseModel):
    source: str
    target: str
    # Optional branching condition (F2). When present, the edge is only
    # "active" if the condition evaluates to True against the run context.
    condition: dict[str, Any] | None = None
    # Optional label, useful for decision-node fan-out and debugging.
    label: str | None = None


class WorkflowDef(BaseModel):
    name: str
    version: str = "1"
    description: str | None = None
    nodes: list[NodeDef]
    edges: list[EdgeDef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> "WorkflowDef":
        ids = [n.id for n in self.nodes]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"Duplicate node ids: {sorted(dupes)}")
        idset = set(ids)
        for e in self.edges:
            if e.source not in idset:
                raise ValueError(f"Edge source '{e.source}' is not a known node")
            if e.target not in idset:
                raise ValueError(f"Edge target '{e.target}' is not a known node")
        return self


class RunRequest(BaseModel):
    workflow: WorkflowDef
    # Initial payload made available to input nodes / the run context.
    inputs: dict[str, Any] = Field(default_factory=dict)


class NodeResult(BaseModel):
    node_id: str
    type: NodeType
    status: str  # success | error | skipped | fallback
    output: Any | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0


class RunResponse(BaseModel):
    run_id: str
    workflow: str
    status: str  # success | error
    outputs: dict[str, Any]
    results: list[NodeResult]
    duration_ms: float
