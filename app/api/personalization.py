"""Personalization API (F22).

Endpoints
---------
GET  /personalization/profile                 Get my preference profile
PUT  /personalization/profile                 Update my preference profile
DELETE /personalization/profile               Reset profile to defaults
POST /personalization/history                 Record a query interaction
GET  /personalization/history                 My recent query history
GET  /personalization/interests               Inferred topics from history
POST /personalization/search                  Personalized hybrid search
GET  /personalization/recommendations         Recommended documents
POST /personalization/recommendations/refresh Rebuild recommendations
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query as QParam
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.personalization import (
    QueryRecord, UserProfile,
    detect_topics, get_history, get_profile, infer_interests,
    personalize_hits, recommend_documents, record_query,
    reset_profile, upsert_profile,
)

router = APIRouter(prefix="/personalization", tags=["personalization"])


# ── Request models ─────────────────────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    search:  dict[str, Any] | None = None
    content: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class RecordQueryRequest(BaseModel):
    query:    str = Field(..., min_length=1)
    doc_ids:  list[str] = Field(default_factory=list)
    rating:   int | None = Field(None, ge=1, le=5)
    provider: str = ""


class PersonalizedSearchRequest(BaseModel):
    query:          str = Field(..., min_length=1)
    store:          str = "kb"
    namespace:      str = "default"
    top_k:          int = Field(5, ge=1, le=50)
    weight_profile: str | None = None
    auto_weight:    bool = True
    text_field:     str = "text"
    record:         bool = True    # whether to log this query to history


# ── Profile endpoints ──────────────────────────────────────────────────────────

@router.get("/profile")
async def get_my_profile(user: CurrentUser) -> dict[str, Any]:
    profile = get_profile(user["id"])
    return profile.model_dump()


@router.put("/profile")
async def update_my_profile(req: UpdateProfileRequest, user: CurrentUser) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if req.search   is not None: updates["search"]   = req.search
    if req.content  is not None: updates["content"]  = req.content
    if req.metadata is not None: updates["metadata"] = req.metadata
    profile = upsert_profile(user["id"], **updates)
    return profile.model_dump()


@router.delete("/profile")
async def reset_my_profile(user: CurrentUser) -> dict[str, Any]:
    reset_profile(user["id"])
    return {"status": "ok", "message": "Profile reset to defaults"}


# ── History endpoints ──────────────────────────────────────────────────────────

@router.post("/history", status_code=201)
async def record_interaction(req: RecordQueryRequest, user: CurrentUser) -> dict[str, Any]:
    """Record a query interaction (called automatically by /personalization/search)."""
    topics = detect_topics(req.query)
    qr = QueryRecord(
        query=req.query, doc_ids=req.doc_ids, rating=req.rating,
        provider=req.provider, topics=topics,
    )
    record_query(user["id"], qr)
    return {"recorded": True, "topics_detected": topics}


@router.get("/history")
async def get_my_history(
    user:  CurrentUser,
    limit: int = QParam(20, ge=1, le=100),
) -> dict[str, Any]:
    history = get_history(user["id"], limit=limit)
    return {
        "total":   len(history),
        "history": [h.model_dump() for h in history],
    }


@router.get("/interests")
async def get_my_interests(
    user:  CurrentUser,
    top_n: int = QParam(5, ge=1, le=10),
) -> dict[str, Any]:
    """Inferred topic interests from query history."""
    profile   = get_profile(user["id"])
    inferred  = infer_interests(user["id"], top_n=top_n)
    return {
        "explicit_interests":    profile.content.interests,
        "explicit_disinterests": profile.content.disinterests,
        "inferred_from_history": inferred,
    }


# ── Personalised search ────────────────────────────────────────────────────────

@router.post("/search")
async def personalized_search(req: PersonalizedSearchRequest,
                              user: CurrentUser) -> dict[str, Any]:
    """Hybrid search with results re-ranked by the caller's preference profile."""
    from app.rag.search.hybrid import hybrid_search

    result = hybrid_search(
        req.query,
        store_name=req.store,
        namespace=req.namespace,
        top_k=req.top_k * 2,   # fetch extra; personalized re-rank trims to top_k
        text_field=req.text_field,
        weight_profile=req.weight_profile,
        auto_weight=req.auto_weight,
    )

    profile      = get_profile(user["id"])
    raw_hits     = result.get("hits", [])
    ranked_hits  = personalize_hits(raw_hits, profile, req.text_field)
    final_hits   = ranked_hits[:req.top_k]

    # Auto-record to history if requested.
    if req.record:
        topics = detect_topics(req.query)
        record_query(user["id"], QueryRecord(query=req.query, topics=topics))

    return {
        **result,
        "hits":           final_hits,
        "total":          len(final_hits),
        "personalized":   True,
        "profile_applied": {
            "interests":    profile.content.interests,
            "disinterests": profile.content.disinterests,
            "prefer_recent": profile.content.prefer_recent,
        },
    }


# ── Recommendations ────────────────────────────────────────────────────────────

@router.get("/recommendations")
async def get_recommendations(
    user:      CurrentUser,
    store:     str = QParam("kb"),
    namespace: str = QParam("default"),
    top_k:     int = QParam(5, ge=1, le=20),
) -> dict[str, Any]:
    """Documents recommended for this user based on their history and interests."""
    recs = recommend_documents(
        user_id=user["id"],
        store_name=store,
        namespace=namespace,
        top_k=top_k,
    )
    inferred = infer_interests(user["id"], top_n=3)
    return {
        "user_id":         user["id"],
        "recommendations": recs,
        "total":           len(recs),
        "based_on":        inferred,
        "message":         "You might be interested in these documents"
                           if recs else "Search more to get personalised recommendations",
    }
