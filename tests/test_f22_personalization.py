"""Tests for F22: Personalization."""
from __future__ import annotations

import pytest

from app.rag.personalization import (
    QueryRecord, UserProfile,
    detect_topics, get_history, get_profile, infer_interests,
    personalize_hits, record_query, reset_store, upsert_profile,
)
from app.rag.personalization.profile import ContentPreferences, SearchPreferences
from app.rag.vectorstore import VectorRecord, get_store, reset_stores
from app.rag.embeddings import embed_texts


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean():
    reset_store()
    reset_stores()
    yield
    reset_store()
    reset_stores()


# ── UserProfile defaults ───────────────────────────────────────────────────────

def test_f22_profile_created_on_first_access():
    p = get_profile("new-user")
    assert isinstance(p, UserProfile)
    assert p.user_id == "new-user"


def test_f22_profile_same_instance_returned():
    p1 = get_profile("alice")
    p2 = get_profile("alice")
    assert p1 is p2


def test_f22_profile_default_search_prefs():
    p = get_profile("alice")
    assert p.search.cite_sources is True
    assert p.search.preferred_format == "prose"


def test_f22_profile_default_content_prefs():
    p = get_profile("alice")
    assert p.content.interests == []
    assert p.content.prefer_recent is True


# ── upsert_profile ─────────────────────────────────────────────────────────────

def test_f22_upsert_search_preferences():
    upsert_profile("alice", search={"cite_sources": False, "preferred_format": "bullets"})
    p = get_profile("alice")
    assert p.search.cite_sources is False
    assert p.search.preferred_format == "bullets"


def test_f22_upsert_content_preferences():
    upsert_profile("alice", content={"interests": ["Finance", "Sales"]})
    p = get_profile("alice")
    assert "Finance" in p.content.interests
    assert "Sales"   in p.content.interests


def test_f22_upsert_disinterests():
    upsert_profile("bob", content={"disinterests": ["HR"]})
    p = get_profile("bob")
    assert "HR" in p.content.disinterests


def test_f22_upsert_metadata():
    upsert_profile("alice", metadata={"team": "analytics"})
    assert get_profile("alice").metadata["team"] == "analytics"


# ── Query history ──────────────────────────────────────────────────────────────

def test_f22_record_and_retrieve_history():
    record_query("alice", QueryRecord(query="revenue report", topics=["Finance"]))
    history = get_history("alice")
    assert len(history) == 1
    assert history[0].query == "revenue report"


def test_f22_history_most_recent_first():
    record_query("alice", QueryRecord(query="first"))
    record_query("alice", QueryRecord(query="second"))
    history = get_history("alice")
    assert history[0].query == "second"


def test_f22_history_limit():
    for i in range(10):
        record_query("alice", QueryRecord(query=f"q{i}"))
    assert len(get_history("alice", limit=3)) == 3


def test_f22_history_empty_for_new_user():
    assert get_history("unknown-user") == []


# ── Topic detection ────────────────────────────────────────────────────────────

def test_f22_detect_finance_topic():
    topics = detect_topics("What is the quarterly revenue and profit margin?")
    assert "Finance" in topics


def test_f22_detect_sales_topic():
    topics = detect_topics("Show me the sales pipeline and deal conversion rates.")
    assert "Sales" in topics


def test_f22_detect_hr_topic():
    topics = detect_topics("What is the employee headcount and hiring plan?")
    assert "HR" in topics


def test_f22_detect_technology_topic():
    topics = detect_topics("How do we deploy the new API infrastructure?")
    assert "Technology" in topics


def test_f22_detect_no_topic():
    topics = detect_topics("hello world")
    assert topics == []


def test_f22_detect_multiple_topics():
    topics = detect_topics("Sales revenue and employee performance review")
    assert len(topics) >= 2


# ── infer_interests ────────────────────────────────────────────────────────────

def test_f22_infer_interests_from_history():
    for _ in range(5):
        record_query("alice", QueryRecord(query="revenue profit", topics=["Finance"]))
    for _ in range(2):
        record_query("alice", QueryRecord(query="sales pipeline", topics=["Sales"]))
    interests = infer_interests("alice", top_n=2)
    assert interests[0]["topic"] == "Finance"
    assert interests[0]["count"] == 5


def test_f22_infer_interests_empty_when_no_history():
    assert infer_interests("nobody") == []


# ── personalize_hits ───────────────────────────────────────────────────────────

def _hit(hid: str, score: float, text: str, **meta) -> dict:
    return {"id": hid, "score": score, "metadata": {"text": text, **meta}}


def test_f22_interest_boosts_matching_hit():
    upsert_profile("alice", content={"interests": ["Finance"]})
    profile = get_profile("alice")
    hits = [
        _hit("finance-doc", 0.5, "quarterly revenue and profit analysis"),
        _hit("other-doc",   0.6, "employee hiring and onboarding process"),
    ]
    ranked = personalize_hits(hits, profile)
    # Finance doc should jump above other-doc despite lower base score.
    assert ranked[0]["id"] == "finance-doc"


def test_f22_disinterest_penalises_matching_hit():
    upsert_profile("bob", content={"disinterests": ["HR"]})
    profile = get_profile("bob")
    hits = [
        _hit("hr-doc",    0.8, "employee headcount and payroll review"),
        _hit("tech-doc",  0.5, "cloud infrastructure deployment"),
    ]
    ranked = personalize_hits(hits, profile)
    assert ranked[0]["id"] == "tech-doc"


def test_f22_personalized_score_field_added():
    profile = get_profile("alice")
    hits    = [_hit("d1", 0.7, "revenue analysis")]
    ranked  = personalize_hits(hits, profile)
    assert "personalized_score" in ranked[0]
    assert "original_score"     in ranked[0]
    assert "personalization"    in ranked[0]


def test_f22_personalization_boost_is_zero_for_empty_profile():
    profile = get_profile("nobody")
    hits    = [_hit("d1", 0.7, "some document")]
    ranked  = personalize_hits(hits, profile)
    # No preferences set — disinterest list empty so penalty = 0,
    # no interests so boost = 0. prefer_recent=True but no date field.
    assert ranked[0]["personalization"]["boost"] == 0.0


def test_f22_recency_boost_applied():
    upsert_profile("alice", content={"prefer_recent": True, "recency_weight": 0.5})
    profile = get_profile("alice")
    from datetime import datetime, timezone
    recent_date = "2025-01-01"
    hits = [
        _hit("recent", 0.5, "text", date=recent_date),
        _hit("old",    0.6, "text"),   # no date → no recency
    ]
    ranked = personalize_hits(hits, profile)
    recent_entry = next(h for h in ranked if h["id"] == "recent")
    assert recent_entry["personalized_score"] > recent_entry["original_score"]


def test_f22_hits_order_preserved_when_no_preference_signal():
    profile = get_profile("neutral")
    hits = [
        _hit("d1", 0.9, "some content"),
        _hit("d2", 0.7, "other content"),
        _hit("d3", 0.5, "third content"),
    ]
    ranked = personalize_hits(hits, profile)
    ids = [h["id"] for h in ranked]
    assert ids == ["d1", "d2", "d3"]


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f22_api_get_profile():
    with _client() as c:
        resp = c.get("/api/v1/personalization/profile", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "dev"
    assert "search"  in data
    assert "content" in data


def test_f22_api_update_profile():
    with _client() as c:
        resp = c.put("/api/v1/personalization/profile", json={
            "content": {"interests": ["Finance", "Sales"], "prefer_recent": True},
            "search":  {"preferred_format": "bullets"},
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "Finance" in data["content"]["interests"]
    assert data["search"]["preferred_format"] == "bullets"


def test_f22_api_reset_profile():
    with _client() as c:
        c.put("/api/v1/personalization/profile",
              json={"content": {"interests": ["Finance"]}}, headers=AUTH)
        resp = c.delete("/api/v1/personalization/profile", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    # Profile is back to defaults.
    with _client() as c:
        p = c.get("/api/v1/personalization/profile", headers=AUTH).json()
    assert p["content"]["interests"] == []


def test_f22_api_record_history():
    with _client() as c:
        resp = c.post("/api/v1/personalization/history", json={
            "query": "quarterly revenue report", "doc_ids": ["d1"],
        }, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["recorded"] is True
    assert "Finance" in data["topics_detected"]


def test_f22_api_get_history():
    with _client() as c:
        c.post("/api/v1/personalization/history",
               json={"query": "revenue analysis"}, headers=AUTH)
        c.post("/api/v1/personalization/history",
               json={"query": "sales pipeline"}, headers=AUTH)
        resp = c.get("/api/v1/personalization/history", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2


def test_f22_api_get_interests():
    with _client() as c:
        c.post("/api/v1/personalization/history",
               json={"query": "revenue profit budget"}, headers=AUTH)
        resp = c.get("/api/v1/personalization/interests", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "inferred_from_history" in data
    assert "explicit_interests"    in data


def test_f22_api_personalized_search():
    store = get_store("kb", 256)
    docs = [
        ("d1", "Quarterly revenue grew 15 percent year over year."),
        ("d2", "Employee headcount increased by 200 last quarter."),
        ("d3", "Sales pipeline conversion rate improved significantly."),
    ]
    vecs = embed_texts([t for _, t in docs], "local-hash", 256)
    for (did, text), vec in zip(docs, vecs):
        store.upsert([VectorRecord(id=did, vector=vec,
                                   metadata={"text": text})])

    with _client() as c:
        # Set Finance interest.
        c.put("/api/v1/personalization/profile",
              json={"content": {"interests": ["Finance"]}}, headers=AUTH)
        resp = c.post("/api/v1/personalization/search", json={
            "query":     "performance results",
            "store":     "kb",
            "namespace": "default",
            "top_k":     3,
            "record":    True,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["personalized"] is True
    assert "profile_applied" in data
    assert len(data["hits"]) <= 3


def test_f22_api_recommendations_empty_without_history():
    with _client() as c:
        resp = c.get("/api/v1/personalization/recommendations?store=kb", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recommendations"] == []


def test_f22_api_recommendations_with_history():
    store = get_store("kb", 256)
    vecs  = embed_texts(["Revenue and profit analysis report."], "local-hash", 256)
    store.upsert([VectorRecord(id="rec-d1", vector=vecs[0],
                               metadata={"text": "Revenue and profit analysis report."})])

    with _client() as c:
        c.post("/api/v1/personalization/history",
               json={"query": "revenue profit"}, headers=AUTH)
        c.post("/api/v1/personalization/history",
               json={"query": "quarterly earnings"}, headers=AUTH)
        resp = c.get("/api/v1/personalization/recommendations?store=kb", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    assert "based_on"        in data


def test_f22_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/personalization/profile")
    assert resp.status_code == 401
