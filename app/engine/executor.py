"""Workflow executor — runs a validated DAG (F1, F2, F3, F4, F6, F7).

Execution model
---------------
* Nodes run in topological generations; nodes within a generation are
  independent and run concurrently (parallel workflows, F3).
* Each edge may carry a condition. After a source node runs, every outgoing
  edge's condition is evaluated; the edge becomes *active* only if it passes
  (branching, F2).
* A node runs only if it is a source (no incoming edges) or at least one of its
  incoming edges is active. Otherwise it is *skipped*, and its outgoing edges
  are inactive — pruning unreachable branches.
* Each node only sees upstream outputs arriving via active edges, so context
  flows along the chosen path (chaining, F3). Merge nodes combine them (F4).
* Per-node retry with exponential backoff, an optional static fallback, and a
  shared circuit breaker give resilience (F7).
* Execution is exposed as an async event stream (F6); :meth:`run` simply
  collects that stream into a single response.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

from opentelemetry.trace import Status, StatusCode

from app.config import settings
from app.engine.circuit_breaker import registry as breaker_registry
from app.engine.conditions import evaluate
from app.engine.context import ExecutionContext
from app.engine.graph import build_graph, generations
from app.engine.nodes import get_node
from app.models.workflow import NodeResult, RunResponse, WorkflowDef
from app.observability.metrics import record_node, record_workflow
from app.observability.tracing import get_tracer

logger = logging.getLogger("engine.executor")
tracer = get_tracer("engine.executor")


class CircuitOpenError(RuntimeError):
    """Raised internally when a node's circuit breaker is open."""


class WorkflowExecutor:
    def __init__(self, wf: WorkflowDef) -> None:
        if len(wf.nodes) > settings.max_nodes:
            raise ValueError(f"Workflow exceeds max_nodes ({settings.max_nodes})")
        self.wf = wf
        self.graph = build_graph(wf)

    async def events(self, inputs: dict | None = None) -> AsyncIterator[dict]:
        """Execute the workflow, yielding events as they happen (F6).

        Event shapes::

            {"event": "workflow_start", "run_id", "workflow"}
            {"event": "node_complete", "result": NodeResult}
            {"event": "workflow_end", "status", "outputs", "duration_ms"}
        """
        run_id = str(uuid.uuid4())
        ctx = ExecutionContext(run_id=run_id, inputs=inputs or {})
        active_edges: dict[int, bool] = {}
        wf_start = time.perf_counter()
        log = logger.getChild(run_id)

        with tracer.start_as_current_span("workflow.run") as wf_span:
            wf_span.set_attribute("workflow.name", self.wf.name)
            wf_span.set_attribute("workflow.version", self.wf.version)
            wf_span.set_attribute("run.id", run_id)
            log.info("workflow_start", extra={"workflow": self.wf.name, "run_id": run_id})
            yield {"event": "workflow_start", "run_id": run_id, "workflow": self.wf.name}

            had_error = False
            for generation in generations(self.graph):
                runnable = [n for n in generation if self._is_active(n, active_edges)]
                skipped = [n for n in generation if n not in runnable]

                for node_id in skipped:
                    ctx.set_output(node_id, None, status="skipped")
                    result = NodeResult(
                        node_id=node_id,
                        type=self.graph.nodes[node_id]["definition"].type,
                        status="skipped",
                    )
                    record_node(result.type.value, "skipped", 0.0)
                    self._deactivate_outgoing(node_id, active_edges)
                    yield {"event": "node_complete", "result": result}

                # Emit node_start for every node about to execute (F6).
                for node_id in runnable:
                    yield {
                        "event": "node_start",
                        "node_id": node_id,
                        "type": self.graph.nodes[node_id]["definition"].type.value,
                    }

                gen_results = await asyncio.gather(
                    *(self._run_node(node_id, ctx) for node_id in runnable)
                )
                for node_result in gen_results:
                    if node_result.status == "error":
                        had_error = True
                    self._resolve_outgoing(node_result.node_id, ctx, active_edges)
                    yield {"event": "node_complete", "result": node_result}

            status = "error" if had_error else "success"
            outputs = self._collect_outputs(ctx)
            duration_ms = (time.perf_counter() - wf_start) * 1000
            wf_span.set_attribute("workflow.status", status)
            record_workflow(self.wf.name, status, duration_ms)
            log.info(
                "workflow_end",
                extra={"workflow": self.wf.name, "status": status, "duration_ms": duration_ms},
            )
            yield {
                "event": "workflow_end",
                "run_id": run_id,
                "status": status,
                "outputs": outputs,
                "duration_ms": duration_ms,
            }

    async def run(self, inputs: dict | None = None) -> RunResponse:
        """Collect the event stream into a single response."""
        results: list[NodeResult] = []
        end: dict = {}
        run_id = ""
        async for ev in self.events(inputs):
            if ev["event"] == "workflow_start":
                run_id = ev["run_id"]
            elif ev["event"] == "node_complete":
                results.append(ev["result"])
            elif ev["event"] == "workflow_end":
                end = ev
        return RunResponse(
            run_id=run_id,
            workflow=self.wf.name,
            status=end.get("status", "error"),
            outputs=end.get("outputs", {}),
            results=results,
            duration_ms=end.get("duration_ms", 0.0),
        )

    # -- helpers -----------------------------------------------------------

    def _collect_outputs(self, ctx: ExecutionContext) -> dict:
        outputs = {
            n.id: ctx.outputs.get(n.id)
            for n in self.wf.nodes
            if n.type.value == "output"
        }
        if not outputs:
            outputs = {
                node_id: ctx.outputs.get(node_id)
                for node_id in self.graph.nodes
                if self.graph.out_degree(node_id) == 0
            }
        return outputs

    def _is_active(self, node_id: str, active_edges: dict[int, bool]) -> bool:
        in_edges = list(self.graph.in_edges(node_id, data=True))
        if not in_edges:
            return True  # source node
        return any(active_edges.get(data["index"], False) for _, _, data in in_edges)

    def _deactivate_outgoing(self, node_id: str, active_edges: dict[int, bool]) -> None:
        for _, _, data in self.graph.out_edges(node_id, data=True):
            active_edges[data["index"]] = False

    def _resolve_outgoing(
        self, node_id: str, ctx: ExecutionContext, active_edges: dict[int, bool]
    ) -> None:
        source_ok = ctx.status.get(node_id) in ("success", "fallback")
        for _, _, data in self.graph.out_edges(node_id, data=True):
            edge = data["definition"]
            active_edges[data["index"]] = source_ok and evaluate(edge.condition, ctx)

    async def _run_node(self, node_id: str, ctx: ExecutionContext) -> NodeResult:
        definition = self.graph.nodes[node_id]["definition"]
        node = get_node(definition)
        node_type = definition.type.value
        upstream = {
            src: ctx.outputs[src]
            for src, _, _ in self.graph.in_edges(node_id, data=True)
            if ctx.status.get(src) in ("success", "fallback")
        }

        log = logger.getChild(ctx.run_id)
        start = time.perf_counter()

        with tracer.start_as_current_span(f"node.{node_type}") as span:
            span.set_attribute("node.id", node_id)
            span.set_attribute("node.type", node_type)
            result = await self._execute_with_resilience(
                node, definition, node_id, ctx, upstream, log, start
            )
            span.set_attribute("node.status", result.status)
            span.set_attribute("node.attempts", result.attempts)
            if result.status == "error":
                span.set_status(Status(StatusCode.ERROR, result.error or "node failed"))
            record_node(node_type, result.status, result.duration_ms)
            return result

    async def _execute_with_resilience(
        self, node, definition, node_id, ctx, upstream, log, start
    ) -> NodeResult:
        attempts = 0
        delay = definition.retry.backoff_seconds
        last_error: Exception | None = None

        # Circuit breaker (F7): keyed by config.breaker_key, shared across runs.
        breaker_key = definition.config.get("breaker_key")
        breaker = (
            breaker_registry.get(
                breaker_key,
                failure_threshold=definition.config.get("breaker_threshold", 5),
                recovery_timeout=definition.config.get("breaker_recovery", 30.0),
            )
            if breaker_key
            else None
        )

        while attempts < definition.retry.max_attempts:
            attempts += 1
            try:
                if breaker is not None and not breaker.allow():
                    raise CircuitOpenError(f"circuit '{breaker_key}' is open")
                output = await node.run(ctx, upstream)
                if breaker is not None:
                    breaker.record_success()
                ctx.set_output(node_id, output, status="success")
                duration_ms = (time.perf_counter() - start) * 1000
                log.info(
                    "node_success",
                    extra={"node_id": node_id, "type": definition.type.value,
                           "attempts": attempts, "duration_ms": duration_ms},
                )
                return NodeResult(
                    node_id=node_id, type=definition.type, status="success",
                    output=output, attempts=attempts, duration_ms=duration_ms,
                )
            except Exception as exc:  # noqa: BLE001 - node failures are data, not crashes
                last_error = exc
                # A CircuitOpenError means we never actually called the node, so
                # don't count it again against the breaker and don't keep retrying.
                if breaker is not None and not isinstance(exc, CircuitOpenError):
                    breaker.record_failure()
                log.warning(
                    "node_attempt_failed",
                    extra={"node_id": node_id, "attempt": attempts, "error": str(exc)},
                )
                if isinstance(exc, CircuitOpenError):
                    break
                if attempts < definition.retry.max_attempts and delay > 0:
                    await asyncio.sleep(delay)
                    delay *= definition.retry.backoff_multiplier

        duration_ms = (time.perf_counter() - start) * 1000
        if definition.fallback is not None:
            ctx.set_output(node_id, definition.fallback, status="fallback")
            log.warning("node_fallback", extra={"node_id": node_id})
            return NodeResult(
                node_id=node_id, type=definition.type, status="fallback",
                output=definition.fallback, error=str(last_error),
                attempts=attempts, duration_ms=duration_ms,
            )

        ctx.set_output(node_id, None, status="error")
        log.error("node_error", extra={"node_id": node_id, "error": str(last_error)})
        return NodeResult(
            node_id=node_id, type=definition.type, status="error",
            error=str(last_error), attempts=attempts, duration_ms=duration_ms,
        )
