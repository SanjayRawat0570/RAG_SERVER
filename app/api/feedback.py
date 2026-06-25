"""Feedback Loop & Continuous Improvement API (F23).

Endpoints
---------
POST   /feedback                   Submit a star rating + optional comment
GET    /feedback                   List feedback (filterable)
GET    /feedback/stats             Aggregate statistics
GET    /feedback/insights          Pattern-based improvement insights
POST   /feedback/ab                Create an A/B test
GET    /feedback/ab                List all A/B tests
GET    /feedback/ab/{test_id}      Results + winner for a test
POST   /feedback/ab/{test_id}/assign   Assign a user to a variant (deterministic)
POST   /feedback/ab/{test_id}/record   Record a rated result for a variant
DELETE /feedback/reset             Wipe all feedback + A/B state (test helper)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query as QParam
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.feedback import (
    ABTest, FeedbackRecord,
    analyze_patterns, assign_variant, create_ab_test,
    feedback_stats, get_ab_results, get_feedback,
    get_insights, list_ab_tests, record_ab_result,
    reset_feedback, submit_feedback,
)

router = APIRouter(prefix="/feedback", tags=["feedback"])


# ── Request models ─────────────────────────────────────────────────────────────

class SubmitFeedbackRequest(BaseModel):
    query_id: str         = Field(..., min_length=1)
    rating:   int         = Field(..., ge=1, le=5)
    comment:  str         = ""
    query:    str         = ""
    answer:   str         = ""
    sources:  list[str]   = Field(default_factory=list)
    signals:  dict[str, Any] = Field(default_factory=dict)
    tags:     list[str]   = Field(default_factory=list)


class CreateABTestRequest(BaseModel):
    name:        str       = Field(..., min_length=1)
    variants:    list[str] = Field(..., min_length=2)
    description: str       = ""


class AssignVariantRequest(BaseModel):
    user_id: str = Field(..., min_length=1)


class RecordABResultRequest(BaseModel):
    variant: str   = Field(..., min_length=1)
    rating:  float = Field(..., ge=1.0, le=5.0)


# ── Feedback endpoints ─────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def submit(req: SubmitFeedbackRequest, user: CurrentUser) -> dict[str, Any]:
    record = FeedbackRecord(
        query_id=req.query_id,
        user_id=user["id"],
        rating=req.rating,
        comment=req.comment,
        query=req.query,
        answer=req.answer,
        sources=req.sources,
        signals=req.signals,
        tags=req.tags,
    )
    result = submit_feedback(record)
    return result.model_dump(mode="json")


@router.get("")
async def list_feedback_endpoint(
    _:          CurrentUser,
    query_id:   str | None = QParam(None),
    rating_min: int | None = QParam(None, ge=1, le=5),
    rating_max: int | None = QParam(None, ge=1, le=5),
    limit:      int        = QParam(50, ge=1, le=500),
) -> dict[str, Any]:
    records = get_feedback(
        query_id=query_id,
        rating_min=rating_min,
        rating_max=rating_max,
        limit=limit,
    )
    return {
        "total":    len(records),
        "feedback": [r.model_dump(mode="json") for r in records],
    }


@router.get("/stats")
async def stats(_: CurrentUser) -> dict[str, Any]:
    return feedback_stats()


@router.get("/insights")
async def insights(_: CurrentUser) -> dict[str, Any]:
    return get_insights()


# ── A/B test endpoints ─────────────────────────────────────────────────────────

@router.post("/ab", status_code=201)
async def create_test(req: CreateABTestRequest, _: CurrentUser) -> dict[str, Any]:
    try:
        test = create_ab_test(req.name, req.variants, req.description)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "id":          test.id,
        "name":        test.name,
        "variants":    list(test.variants.keys()),
        "description": test.description,
    }


@router.get("/ab")
async def list_tests(_: CurrentUser) -> dict[str, Any]:
    tests = list_ab_tests()
    return {
        "total": len(tests),
        "tests": [
            {"id": t.id, "name": t.name, "variants": list(t.variants.keys())}
            for t in tests
        ],
    }


@router.get("/ab/{test_id}")
async def get_test_results(test_id: str, _: CurrentUser) -> dict[str, Any]:
    try:
        return get_ab_results(test_id)
    except KeyError:
        raise HTTPException(404, f"A/B test '{test_id}' not found")


@router.post("/ab/{test_id}/assign")
async def assign(test_id: str, req: AssignVariantRequest, _: CurrentUser) -> dict[str, Any]:
    try:
        variant = assign_variant(test_id, req.user_id)
    except KeyError:
        raise HTTPException(404, f"A/B test '{test_id}' not found")
    return {"test_id": test_id, "user_id": req.user_id, "variant": variant}


@router.post("/ab/{test_id}/record", status_code=201)
async def record_result(
    test_id: str, req: RecordABResultRequest, _: CurrentUser
) -> dict[str, Any]:
    try:
        record_ab_result(test_id, req.variant, req.rating)
    except KeyError:
        raise HTTPException(404, f"A/B test '{test_id}' not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "ok", "test_id": test_id, "variant": req.variant, "rating": req.rating}


@router.delete("/reset")
async def reset(_: CurrentUser) -> dict[str, Any]:
    reset_feedback()
    return {"status": "ok"}
