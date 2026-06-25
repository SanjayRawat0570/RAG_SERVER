"""Distributed tracing via OpenTelemetry (F8).

A single TracerProvider is configured once. If ``otel_exporter_endpoint`` is set
(e.g. ``http://jaeger:4317``) spans are batched and exported over OTLP/gRPC to
Jaeger; otherwise spans are still created (so instrumentation code is always
exercised) but not shipped — keeping local and test runs dependency-free.

Trace/span ids are exposed via :func:`current_trace_id` so they can be attached
to structured logs, giving trace<->log correlation.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

from app.config import settings

_initialised = False


def init_tracing() -> None:
    global _initialised
    if _initialised:
        return
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.service_name})
    )
    if settings.otel_exporter_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    # Always capture spans in memory for local trace inspection (F8).
    from app.observability.span_store import init_span_store
    init_span_store(provider)
    trace.set_tracer_provider(provider)
    _initialised = True


def get_tracer(name: str = "engine"):
    return trace.get_tracer(name)


def current_trace_id() -> str | None:
    """Hex trace id of the active span, for log correlation."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return None
    return format_trace_id(ctx.trace_id)


def current_span_id() -> str | None:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return None
    return format_span_id(ctx.span_id)
