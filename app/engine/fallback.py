"""Cascading fallback chains (F7: Error Handling & Fallbacks).

A FallbackChain tries each option workflow in order until one succeeds.
Unlike the executor's per-node ``fallback`` (which returns a static value),
this runs *completely different workflows*, enabling the four spec scenarios:

  Scenario 1 — Vector DB down     : semantic search → keyword search → entity search
  Scenario 2 — LLM API timeout    : primary LLM → secondary LLM → cached answer
  Scenario 3 — No search results  : direct query → expanded query → related terms
  Scenario 4 — Document corrupted : PDF extract → OCR → re-upload prompt

``FallbackExecutor.run()`` iterates options, considers an option failed when:
  * Its workflow returns status == "error", OR
  * It returns an empty list result (``skip_empty_results=True``)

The result includes the full attempt history, which option won, how deep the
fallback chain had to go, and a human-readable explanation. Everything is
logged to Supabase ``audit_logs``; silent no-op when offline.
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

class FallbackOption(BaseModel):
    name: str
    description: str = ""
    workflow: WorkflowDef


class FallbackChainDef(BaseModel):
    name: str
    description: str | None = None
    # Tried in order; first success wins.
    options: list[FallbackOption]
    # Behaviour when every option fails:
    #   "error" — raise RuntimeError (caller must handle)
    #   "empty" — return a FallbackResult with succeeded=False (graceful)
    on_all_fail: str = "error"
    # Treat an option that returns [] as a failure — keep trying.
    skip_empty_results: bool = True


class FallbackAttempt(BaseModel):
    option: str
    status: str                    # "success" | "error" | "empty"
    error: str | None = None
    duration_ms: float = 0.0


class FallbackResult(BaseModel):
    chain_id: str
    chain: str
    succeeded: bool
    used_option: str               # name of winning option, or "none"
    fallback_depth: int            # 0 = primary, 1 = first fallback, …
    degraded: bool                 # True when a fallback (not primary) won
    outputs: dict[str, Any]
    attempts: list[FallbackAttempt]
    # Human-readable explanation of what happened — shown to the user.
    message: str
    # Partial result metadata for graceful degradation display.
    partial_note: str | None = None
    duration_ms: float


# ── executor ──────────────────────────────────────────────────────────────────

class FallbackExecutor:
    """Execute a FallbackChainDef, cascading through options on failure."""

    def __init__(self, chain: FallbackChainDef) -> None:
        self.chain = chain

    async def run(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> FallbackResult:
        chain_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        attempts: list[FallbackAttempt] = []

        for depth, option in enumerate(self.chain.options):
            t_opt = time.perf_counter()
            try:
                run = await WorkflowExecutor(option.workflow).run(inputs or {})
                dur = (time.perf_counter() - t_opt) * 1000

                if run.status == "error":
                    attempts.append(FallbackAttempt(
                        option=option.name, status="error",
                        error="workflow returned error status",
                        duration_ms=dur,
                    ))
                    continue

                # Empty-result check (search returns [])
                if self.chain.skip_empty_results:
                    out = run.outputs.get("out")
                    if isinstance(out, list) and len(out) == 0:
                        attempts.append(FallbackAttempt(
                            option=option.name, status="empty",
                            error="no results returned",
                            duration_ms=dur,
                        ))
                        continue

                attempts.append(FallbackAttempt(
                    option=option.name, status="success", duration_ms=dur,
                ))

                desc = option.description or option.name
                msg = (
                    f"Succeeded with {desc}."
                    if depth == 0
                    else f"Primary search failed — results from fallback: {desc}."
                )

                # Build partial-degradation note when not primary.
                partial_note: str | None = None
                if depth > 0:
                    failed = ", ".join(a.option for a in attempts[:-1])
                    partial_note = (
                        f"Note: primary option(s) [{failed}] did not return results. "
                        f"Results shown are from '{option.name}' and may be less precise."
                    )

                result = FallbackResult(
                    chain_id=chain_id,
                    chain=self.chain.name,
                    succeeded=True,
                    used_option=option.name,
                    fallback_depth=depth,
                    degraded=(depth > 0),
                    outputs=run.outputs,
                    attempts=attempts,
                    message=msg,
                    partial_note=partial_note,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
                await _persist(result, user_id)
                return result

            except Exception as exc:  # noqa: BLE001
                dur = (time.perf_counter() - t_opt) * 1000
                attempts.append(FallbackAttempt(
                    option=option.name, status="error",
                    error=str(exc), duration_ms=dur,
                ))

        # ── All options failed ────────────────────────────────────────────────
        tried = ", ".join(a.option for a in attempts)
        result = FallbackResult(
            chain_id=chain_id,
            chain=self.chain.name,
            succeeded=False,
            used_option="none",
            fallback_depth=len(self.chain.options),
            degraded=True,
            outputs={},
            attempts=attempts,
            message=f"All fallback options exhausted. Tried: {tried}.",
            partial_note=(
                "Service is currently unavailable. "
                "Please try again later or contact support."
            ),
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        await _persist(result, user_id)

        if self.chain.on_all_fail == "error":
            raise RuntimeError(result.message)
        return result


# ── Supabase persistence ──────────────────────────────────────────────────────

async def _persist(result: FallbackResult, user_id: str | None) -> None:
    """Write fallback/error event to Supabase audit_logs; silent no-op offline."""
    try:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return
        from supabase import create_client  # type: ignore[import]
        sb = create_client(settings.supabase_url, settings.supabase_key)
        sb.table("audit_logs").insert({
            "id":            result.chain_id,
            "user_id":       user_id or "anonymous",
            "operation":     "fallback" if result.succeeded else "error",
            "decision_tree": result.chain,
            "decision_path": [a.option for a in result.attempts],
            "outcome":       result.used_option,
            "confidence":    float(not result.degraded),
            "created_at":    datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:  # noqa: BLE001
        pass
