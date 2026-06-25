"""Personalised re-ranking of search hits (F22).

Applies score adjustments based on:
  1. Topic interest match   — boost hits whose text covers user's interests
  2. Recency boost          — newer docs ranked higher if prefer_recent
  3. Authority boost        — trusted sources ranked higher if prefer_authoritative
  4. Disinterest penalty    — demote hits matching user's disinterests
  5. Document type filter   — preferred_doc_types move to top

All adjustments are additive on top of the original score so the
baseline retrieval order is preserved when preferences have no signal.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.rag.personalization.profile import UserProfile
from app.rag.personalization.store   import _TOPIC_KEYWORDS


def _text(hit: dict[str, Any], text_field: str = "text") -> str:
    return str(hit.get("metadata", {}).get(text_field, "")).lower()


def _doc_date(hit: dict[str, Any]) -> datetime | None:
    raw = hit.get("metadata", {}).get("date") or hit.get("metadata", {}).get("created_at")
    if not raw:
        return None
    try:
        if isinstance(raw, datetime):
            return raw
        if re.match(r"\d{4}", str(raw)):
            year = int(str(raw)[:4])
            return datetime(year, 1, 1, tzinfo=timezone.utc)
    except Exception:
        pass
    return None


def personalize_hits(
    hits:        list[dict[str, Any]],
    profile:     UserProfile,
    text_field:  str = "text",
) -> list[dict[str, Any]]:
    """Re-rank *hits* according to *profile*. Returns a new sorted list."""
    if not hits:
        return hits

    now = datetime.now(timezone.utc)
    scored: list[tuple[float, dict[str, Any]]] = []

    for hit in hits:
        base   = float(hit.get("score", 0.0))
        boost  = 0.0
        text   = _text(hit, text_field)
        meta   = hit.get("metadata", {})

        # 1. Interest boost
        for interest in profile.content.interests:
            kws = _TOPIC_KEYWORDS.get(interest, [interest.lower()])
            if any(kw in text for kw in kws):
                boost += 0.15

        # 2. Disinterest penalty
        for disinterest in profile.content.disinterests:
            kws = _TOPIC_KEYWORDS.get(disinterest, [disinterest.lower()])
            if any(kw in text for kw in kws):
                boost -= 0.40

        # 3. Recency boost
        if profile.content.prefer_recent:
            doc_date = _doc_date(hit)
            if doc_date:
                age_days = (now - doc_date.replace(tzinfo=timezone.utc)).days
                # Decay over 2-year window so documents within 2 years get a boost.
                recency_factor = max(0.0, 1.0 - age_days / 730)
                boost += profile.content.recency_weight * recency_factor

        # 4. Authority boost
        if profile.content.prefer_authoritative:
            source = str(meta.get("source", "")).lower()
            for auth in profile.content.authority_sources:
                if auth.lower() in source:
                    boost += 0.10
                    break

        # 5. Preferred document types
        doc_type = str(meta.get("doc_type", meta.get("format", ""))).lower()
        if profile.content.preferred_doc_types:
            if doc_type in [t.lower() for t in profile.content.preferred_doc_types]:
                boost += 0.08

        personalized_score = round(base + boost, 6)
        entry = {
            **hit,
            "original_score":    base,
            "personalized_score": personalized_score,
            "score":             personalized_score,
            "personalization": {
                "boost": round(boost, 6),
                "interests_matched":   [i for i in profile.content.interests
                                        if any(kw in text for kw in
                                               _TOPIC_KEYWORDS.get(i, [i.lower()]))],
                "disinterests_matched": [d for d in profile.content.disinterests
                                         if any(kw in text for kw in
                                                _TOPIC_KEYWORDS.get(d, [d.lower()]))],
            },
        }
        scored.append((personalized_score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored]


def recommend_documents(
    user_id:   str,
    store_name: str = "kb",
    namespace:  str = "default",
    top_k:     int = 5,
    text_field: str = "text",
) -> list[dict[str, Any]]:
    """Return documents recommended for *user_id* based on their history.

    Strategy:
      1. Pull the user's inferred interests.
      2. Build a pseudo-query from the most frequent search terms.
      3. Run semantic search with personalized re-ranking.
    """
    from app.rag.personalization.store import get_history, infer_interests
    from app.rag.vectorstore import lookup_store
    from app.rag.embeddings import embed_texts
    from app.rag.embeddings.registry import DEFAULT_DIMENSION, DEFAULT_MODEL

    profile    = get_profile_fn(user_id)
    history    = get_history(user_id, limit=50)
    interests  = infer_interests(user_id, top_n=3)

    if not history and not interests:
        return []

    # Build pseudo-query from recent searches + interests
    recent_queries = " ".join(r.query for r in history[:5])
    interest_terms = " ".join(i["topic"] for i in interests)
    pseudo_query   = f"{recent_queries} {interest_terms}".strip() or "general"

    store = lookup_store(store_name)
    if store is None:
        return []

    dim       = DEFAULT_DIMENSION
    query_vec = embed_texts([pseudo_query], DEFAULT_MODEL, dim)[0]
    raw_hits  = store.search(query_vec, top_k=top_k * 2, namespace=namespace)

    hits = [{"id": h.id, "score": h.score, "metadata": h.metadata} for h in raw_hits]
    ranked = personalize_hits(hits, profile, text_field)

    return [
        {
            **h,
            "match_pct": round(min(h["personalized_score"] * 100, 100), 1),
            "reason":    _recommend_reason(h, interests),
        }
        for h in ranked[:top_k]
    ]


def _recommend_reason(hit: dict, interests: list[dict]) -> str:
    matched = hit.get("personalization", {}).get("interests_matched", [])
    if matched:
        return f"Matches your interest in {matched[0]}"
    if interests:
        return f"Related to your frequent topic: {interests[0]['topic']}"
    return "Based on your search history"


def get_profile_fn(user_id: str) -> UserProfile:
    from app.rag.personalization.store import get_profile
    return get_profile(user_id)
