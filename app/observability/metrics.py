"""Prometheus metrics (F8).

Counters and histograms capture throughput, error rate, and latency for both
whole workflows and individual nodes. Histograms let Prometheus/Grafana compute
P50/P95/P99 via ``histogram_quantile`` over the exported buckets.
"""
from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

WORKFLOW_RUNS = Counter(
    "workflow_runs_total",
    "Total workflow executions.",
    ["workflow", "status"],
)
WORKFLOW_DURATION = Histogram(
    "workflow_duration_seconds",
    "Workflow execution duration.",
    ["workflow"],
)
NODE_RUNS = Counter(
    "node_runs_total",
    "Total node executions.",
    ["type", "status"],
)
NODE_DURATION = Histogram(
    "node_duration_seconds",
    "Per-node execution duration.",
    ["type"],
)
CACHE_OPS = Counter(
    "cache_operations_total",
    "Cache lookups by result (F17).",
    ["cache", "result"],  # result: hit | miss
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "LLM tokens consumed (F24).",
    ["provider", "model", "kind"],  # kind: input | output
)
LLM_COST = Counter(
    "llm_cost_usd_total",
    "Estimated LLM spend in USD (F24).",
    ["provider", "model"],
)


def record_cache(cache: str, hit: bool) -> None:
    CACHE_OPS.labels(cache=cache, result="hit" if hit else "miss").inc()


def record_llm_usage(
    provider: str, model: str, input_tokens: int, output_tokens: int, cost: float
) -> None:
    LLM_TOKENS.labels(provider=provider, model=model, kind="input").inc(input_tokens or 0)
    LLM_TOKENS.labels(provider=provider, model=model, kind="output").inc(output_tokens or 0)
    if cost:
        LLM_COST.labels(provider=provider, model=model).inc(cost)


def record_node(node_type: str, status: str, duration_ms: float) -> None:
    NODE_RUNS.labels(type=node_type, status=status).inc()
    NODE_DURATION.labels(type=node_type).observe(duration_ms / 1000.0)


def record_workflow(workflow: str, status: str, duration_ms: float) -> None:
    WORKFLOW_RUNS.labels(workflow=workflow, status=status).inc()
    WORKFLOW_DURATION.labels(workflow=workflow).observe(duration_ms / 1000.0)


def render_latest() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
