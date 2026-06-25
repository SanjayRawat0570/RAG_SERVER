"""Cost Optimization API (F24).

Endpoints
---------
GET  /cost/pricing               Pricing table (model → $/1K tokens)
POST /cost/estimate              Estimate cost for a model + token counts
GET  /cost/budgets               All budget keys with spend/remaining
POST /cost/budgets               Create or update a budget limit
POST /cost/budgets/reserve       Pre-flight: check if estimated cost fits budget
POST /cost/usage                 Record a usage event
GET  /cost/usage                 List recent usage events (filterable)
GET  /cost/summary               Aggregate totals by model / operation
POST /cost/recommend             Recommend cheapest/best model for budget
POST /cost/compare               Side-by-side cost comparison of models
POST /cost/savings/cache         Estimate savings from a cache hit-rate
DELETE /cost/reset               Wipe usage log + all budgets (test helper)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query as QParam
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.cost import (
    BudgetExceededError,
    budget_stats,
    cache_savings,
    count_tokens,
    estimate_cost,
    get_usage,
    model_comparison,
    record_spend,
    record_usage,
    recommend_model,
    reserve,
    reset_budgets,
    reset_tracker,
    usage_summary,
    upsert_budget,
)
from app.rag.cost.pricing import PRICING, price_for

router = APIRouter(prefix="/cost", tags=["cost"])


# ── Request models ─────────────────────────────────────────────────────────────

class EstimateRequest(BaseModel):
    model:         str
    input_tokens:  int   = Field(..., ge=0)
    output_tokens: int   = Field(..., ge=0)


class RecordUsageRequest(BaseModel):
    model:         str   = ""
    operation:     str   = "llm"
    input_tokens:  int   = Field(0, ge=0)
    output_tokens: int   = Field(0, ge=0)
    cached:        bool  = False
    tenant:        str   = "default"


class BudgetRequest(BaseModel):
    key:   str   = Field(..., min_length=1)
    limit: float = Field(..., gt=0)


class ReserveRequest(BaseModel):
    key:            str   = Field(..., min_length=1)
    limit:          float = Field(..., gt=0)
    estimated_cost: float = Field(..., ge=0)


class RecommendRequest(BaseModel):
    budget_remaining:  float = Field(..., ge=0)
    avg_input_tokens:  int   = Field(1000, ge=1)
    avg_output_tokens: int   = Field(200,  ge=0)
    prefer_quality:    bool  = True


class CompareRequest(BaseModel):
    models:        list[str] = Field(..., min_length=1)
    input_tokens:  int       = Field(1000, ge=1)
    output_tokens: int       = Field(200,  ge=0)


class CacheSavingsRequest(BaseModel):
    total_queries:     int   = Field(..., ge=1)
    cache_hit_rate:    float = Field(0.8, ge=0.0, le=1.0)
    model:             str
    avg_input_tokens:  int   = Field(500, ge=1)
    avg_output_tokens: int   = Field(200, ge=0)


# ── Pricing / estimation ───────────────────────────────────────────────────────

@router.get("/pricing")
async def get_pricing(_: CurrentUser) -> dict[str, Any]:
    return {
        "pricing": {
            model: {"input_per_1k": inp, "output_per_1k": out}
            for model, (inp, out) in PRICING.items()
        }
    }


@router.post("/estimate")
async def estimate(_: CurrentUser, req: EstimateRequest) -> dict[str, Any]:
    cost = estimate_cost(req.model, req.input_tokens, req.output_tokens)
    inp, out = price_for(req.model)
    return {
        "model":              req.model,
        "input_tokens":       req.input_tokens,
        "output_tokens":      req.output_tokens,
        "estimated_cost":     cost,
        "price_per_1k_input": inp,
        "price_per_1k_output": out,
    }


# ── Budgets ────────────────────────────────────────────────────────────────────

@router.get("/budgets")
async def list_budgets(_: CurrentUser) -> dict[str, Any]:
    return {"budgets": budget_stats()}


@router.post("/budgets", status_code=201)
async def create_budget(req: BudgetRequest, _: CurrentUser) -> dict[str, Any]:
    return upsert_budget(req.key, req.limit)


@router.post("/budgets/reserve")
async def check_reserve(req: ReserveRequest, _: CurrentUser) -> dict[str, Any]:
    try:
        reserve(req.key, req.limit, req.estimated_cost)
        return {"allowed": True, "key": req.key, "estimated_cost": req.estimated_cost}
    except BudgetExceededError as exc:
        return {"allowed": False, "key": req.key, "reason": str(exc)}


# ── Usage tracking ─────────────────────────────────────────────────────────────

@router.post("/usage", status_code=201)
async def record(req: RecordUsageRequest, user: CurrentUser) -> dict[str, Any]:
    event = record_usage(
        user_id=user["id"],
        tenant=req.tenant,
        model=req.model,
        operation=req.operation,
        input_tokens=req.input_tokens,
        output_tokens=req.output_tokens,
        cached=req.cached,
    )
    return {
        "id":        event.id,
        "model":     event.model,
        "operation": event.operation,
        "cost":      event.cost,
        "cached":    event.cached,
        "timestamp": event.timestamp.isoformat(),
    }


@router.get("/usage")
async def get_usage_endpoint(
    _:         CurrentUser,
    user_id:   str | None = QParam(None),
    tenant:    str | None = QParam(None),
    model:     str | None = QParam(None),
    operation: str | None = QParam(None),
    limit:     int        = QParam(50, ge=1, le=500),
) -> dict[str, Any]:
    events = get_usage(
        user_id=user_id, tenant=tenant, model=model, operation=operation, limit=limit
    )
    return {
        "total": len(events),
        "events": [
            {
                "id":            e.id,
                "user_id":       e.user_id,
                "tenant":        e.tenant,
                "model":         e.model,
                "operation":     e.operation,
                "input_tokens":  e.input_tokens,
                "output_tokens": e.output_tokens,
                "cost":          e.cost,
                "cached":        e.cached,
                "timestamp":     e.timestamp.isoformat(),
            }
            for e in events
        ],
    }


@router.get("/summary")
async def summary(
    _:       CurrentUser,
    user_id: str | None = QParam(None),
    tenant:  str | None = QParam(None),
) -> dict[str, Any]:
    return usage_summary(user_id=user_id, tenant=tenant)


# ── Optimization ───────────────────────────────────────────────────────────────

@router.post("/recommend")
async def recommend(_: CurrentUser, req: RecommendRequest) -> dict[str, Any]:
    return recommend_model(
        budget_remaining=req.budget_remaining,
        avg_input_tokens=req.avg_input_tokens,
        avg_output_tokens=req.avg_output_tokens,
        prefer_quality=req.prefer_quality,
    )


@router.post("/compare")
async def compare(_: CurrentUser, req: CompareRequest) -> dict[str, Any]:
    results = model_comparison(req.models, req.input_tokens, req.output_tokens)
    return {
        "models":   results,
        "cheapest": results[0]["model"] if results else None,
    }


@router.post("/savings/cache")
async def cache_savings_estimate(_: CurrentUser, req: CacheSavingsRequest) -> dict[str, Any]:
    return cache_savings(
        total_queries=req.total_queries,
        cache_hit_rate=req.cache_hit_rate,
        model=req.model,
        avg_input_tokens=req.avg_input_tokens,
        avg_output_tokens=req.avg_output_tokens,
    )


# ── Reset (test helper) ────────────────────────────────────────────────────────

@router.delete("/reset")
async def reset(_: CurrentUser) -> dict[str, Any]:
    reset_tracker()
    reset_budgets()
    return {"status": "ok"}
