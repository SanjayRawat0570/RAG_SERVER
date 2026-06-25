"""F3: Sequential chain execution.

A chain is an ordered sequence of workflow steps where each step's output
automatically feeds the next step's inputs. Builds on the workflow engine
(F1/F2) and adds:

  * fail_fast (default) — stop immediately when any step fails
  * continue            — log the failure, skip that step, keep going
  * Per-step output forwarding via flattening of the "out" node dict

Execution history is persisted to Supabase ``workflow_executions`` when
configured; falls back to in-memory-only when Supabase is absent (offline
mode stays fully functional).
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef


# ── models ────────────────────────────────────────────────────────────────────

class ChainStep(BaseModel):
    name: str
    workflow: WorkflowDef
    # Explicit output keys to forward to the next step.
    # Leave empty to forward all keys from the flattened "out" node.
    forward_keys: list[str] = Field(default_factory=list)


class ChainDef(BaseModel):
    name: str
    description: str | None = None
    steps: list[ChainStep]
    # "fail_fast" (default) | "continue"
    on_error: str = "fail_fast"


class StepResult(BaseModel):
    step: str
    status: str        # success | error
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class ChainResult(BaseModel):
    chain_id: str
    chain: str
    status: str        # success | error | partial
    steps: list[StepResult]
    final_output: dict[str, Any]
    duration_ms: float


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten(raw_outputs: dict[str, Any]) -> dict[str, Any]:
    """Flatten {"out": {k: v}} → {k: v}, merging all output-node dicts."""
    flat: dict[str, Any] = {}
    for val in raw_outputs.values():
        if isinstance(val, dict):
            flat.update(val)
    return flat or dict(raw_outputs)


async def _persist(
    chain_id: str,
    chain_name: str,
    result: "ChainResult",
    user_id: str | None,
) -> None:
    """Write execution record to Supabase; silent no-op when unconfigured."""
    try:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return
        from supabase import create_client  # type: ignore[import]
        sb = create_client(settings.supabase_url, settings.supabase_key)
        sb.table("workflow_executions").insert({
            "id":           chain_id,
            "user_id":      user_id or "anonymous",
            "tenant":       result.final_output.get("tenant", "default"),
            "filename":     chain_name,
            "status":       result.status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "step_outputs": result.model_dump(),
        }).execute()
    except Exception:  # noqa: BLE001 — observability must never break execution
        pass


# ── executor ──────────────────────────────────────────────────────────────────

class ChainExecutor:
    """Run a ChainDef sequentially, forwarding outputs between steps."""

    def __init__(self, chain: ChainDef) -> None:
        self.chain = chain

    async def run(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> ChainResult:
        chain_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        ctx: dict[str, Any] = dict(inputs or {})
        step_results: list[StepResult] = []
        had_error = False

        for step in self.chain.steps:
            t_step = time.perf_counter()
            try:
                run = await WorkflowExecutor(step.workflow).run(ctx)
                ms = (time.perf_counter() - t_step) * 1000

                if run.status == "error":
                    had_error = True
                    step_results.append(StepResult(
                        step=step.name, status="error",
                        outputs=run.outputs,
                        error="workflow returned error status",
                        duration_ms=ms,
                    ))
                    if self.chain.on_error == "fail_fast":
                        break
                    continue

                # Merge this step's outputs into context for the next step.
                flat = _flatten(run.outputs)
                forwarded = (
                    {k: flat[k] for k in step.forward_keys if k in flat}
                    if step.forward_keys else flat
                )
                ctx = {**ctx, **forwarded}

                step_results.append(StepResult(
                    step=step.name, status="success",
                    outputs=run.outputs, duration_ms=ms,
                ))

            except Exception as exc:  # noqa: BLE001
                ms = (time.perf_counter() - t_step) * 1000
                had_error = True
                step_results.append(StepResult(
                    step=step.name, status="error",
                    error=str(exc), duration_ms=ms,
                ))
                if self.chain.on_error == "fail_fast":
                    break

        n_ok = sum(1 for s in step_results if s.status == "success")
        if not had_error:
            status = "success"
        elif n_ok > 0 and self.chain.on_error == "continue":
            status = "partial"
        else:
            status = "error"

        result = ChainResult(
            chain_id=chain_id,
            chain=self.chain.name,
            status=status,
            steps=step_results,
            final_output=ctx,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        await _persist(chain_id, self.chain.name, result, user_id)
        return result
