"""In-memory span store for local trace inspection (F8).

Attaches an InMemorySpanExporter to the TracerProvider during init so every
span is also captured locally — no Jaeger / OTLP collector needed for dev or
test.  Provides ``list_traces()`` and ``get_trace()`` for the monitoring API.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_span_id, format_trace_id

_exporter: InMemorySpanExporter = InMemorySpanExporter()
_MAX_SPANS = 2000


def get_exporter() -> InMemorySpanExporter:
    return _exporter


def init_span_store(provider) -> None:
    """Attach the in-memory exporter to a TracerProvider (call once)."""
    provider.add_span_processor(SimpleSpanProcessor(_exporter))


def _span_to_dict(span) -> dict[str, Any]:
    ctx = span.get_span_context()
    start_ns = span.start_time or 0
    end_ns = span.end_time or start_ns
    return {
        "trace_id":    format_trace_id(ctx.trace_id),
        "span_id":     format_span_id(ctx.span_id),
        "name":        span.name,
        "start_ns":    start_ns,
        "end_ns":      end_ns,
        "duration_ms": round((end_ns - start_ns) / 1_000_000, 2),
        "status":      span.status.status_code.name,
        "attributes":  dict(span.attributes or {}),
    }


def list_traces(limit: int = 20) -> list[dict[str, Any]]:
    """Return a list of recent unique trace summaries (most recent first)."""
    all_spans = _exporter.get_finished_spans()[-_MAX_SPANS:]

    by_trace: dict[str, list] = defaultdict(list)
    for span in all_spans:
        ctx = span.get_span_context()
        by_trace[format_trace_id(ctx.trace_id)].append(span)

    summaries: list[dict[str, Any]] = []
    for trace_id, spans in by_trace.items():
        root = min(spans, key=lambda s: s.start_time or 0)
        node_spans = [s for s in spans if s.name.startswith("node.")]
        has_error = any(s.status.status_code.name == "ERROR" for s in spans)
        summaries.append({
            "trace_id":    trace_id,
            "root_name":   root.name,
            "span_count":  len(spans),
            "start_ns":    root.start_time,
            "duration_ms": round(
                sum((s.end_time or 0) - (s.start_time or 0) for s in node_spans) / 1_000_000, 2
            ),
            "status":      "ERROR" if has_error else "OK",
            "workflow":    (root.attributes or {}).get("workflow.name", ""),
        })

    summaries.sort(key=lambda t: t.get("start_ns") or 0, reverse=True)
    return summaries[:limit]


def get_trace(trace_id: str) -> list[dict[str, Any]]:
    """Return all spans for a specific trace_id, sorted by start time."""
    spans = _exporter.get_finished_spans()
    result = [
        _span_to_dict(s)
        for s in spans
        if format_trace_id(s.get_span_context().trace_id) == trace_id
    ]
    result.sort(key=lambda s: s["start_ns"])
    return result
