"""Tests for F23: Feedback Loop & Continuous Improvement."""
from __future__ import annotations

import pytest

from app.rag.feedback import (
    ABTest, FeedbackRecord, PatternReport,
    analyze_patterns, assign_variant, create_ab_test,
    feedback_stats, get_ab_results, get_feedback,
    get_insights, list_ab_tests, record_ab_result,
    reset_feedback, submit_feedback,
)
from app.rag.feedback.models import ABTest, PatternReport


@pytest.fixture(autouse=True)
def _clean():
    reset_feedback()
    yield
    reset_feedback()


# ── FeedbackRecord model ───────────────────────────────────────────────────────

def test_f23_feedback_defaults():
    f = FeedbackRecord(query_id="q1", rating=4)
    assert f.rating   == 4
    assert f.comment  == ""
    assert f.id       != ""


def test_f23_feedback_rating_too_high():
    with pytest.raises(Exception):
        FeedbackRecord(query_id="q1", rating=6)


def test_f23_feedback_rating_too_low():
    with pytest.raises(Exception):
        FeedbackRecord(query_id="q1", rating=0)


def test_f23_feedback_with_comment():
    f = FeedbackRecord(query_id="q1", rating=3, comment="Missing Q4 data")
    assert f.comment == "Missing Q4 data"


# ── submit / get_feedback ─────────────────────────────────────────────────────

def test_f23_submit_and_retrieve():
    submit_feedback(FeedbackRecord(query_id="q1", rating=4))
    results = get_feedback(query_id="q1")
    assert len(results) == 1
    assert results[0].rating == 4


def test_f23_get_feedback_by_user():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5, user_id="alice"))
    submit_feedback(FeedbackRecord(query_id="q2", rating=3, user_id="bob"))
    alice_fb = get_feedback(user_id="alice")
    assert len(alice_fb) == 1
    assert alice_fb[0].user_id == "alice"


def test_f23_get_feedback_by_rating_min():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5))
    submit_feedback(FeedbackRecord(query_id="q2", rating=2))
    high = get_feedback(rating_min=4)
    assert all(f.rating >= 4 for f in high)


def test_f23_get_feedback_most_recent_first():
    submit_feedback(FeedbackRecord(query_id="q1", rating=3))
    submit_feedback(FeedbackRecord(query_id="q2", rating=4))
    results = get_feedback()
    assert results[0].query_id == "q2"


# ── feedback_stats ─────────────────────────────────────────────────────────────

def test_f23_stats_empty():
    stats = feedback_stats()
    assert stats["count"]          == 0
    assert stats["average_rating"] == 0.0


def test_f23_stats_average_rating():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5))
    submit_feedback(FeedbackRecord(query_id="q2", rating=3))
    stats = feedback_stats()
    assert abs(stats["average_rating"] - 4.0) < 0.01


def test_f23_stats_rating_distribution():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5))
    submit_feedback(FeedbackRecord(query_id="q2", rating=5))
    submit_feedback(FeedbackRecord(query_id="q3", rating=3))
    stats = feedback_stats()
    assert stats["distribution"][5] == 2
    assert stats["distribution"][3] == 1


def test_f23_stats_high_low_counts():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5))
    submit_feedback(FeedbackRecord(query_id="q2", rating=4))
    submit_feedback(FeedbackRecord(query_id="q3", rating=2))
    stats = feedback_stats()
    assert stats["high_rated_count"] == 2
    assert stats["low_rated_count"]  == 1


# ── analyze_patterns ──────────────────────────────────────────────────────────

def test_f23_analyze_returns_report():
    for i in range(3):
        submit_feedback(FeedbackRecord(
            query_id=f"good-{i}", rating=5,
            signals={"source_count": 3, "confidence": 0.92, "has_numbers": True},
        ))
    for i in range(3):
        submit_feedback(FeedbackRecord(
            query_id=f"bad-{i}", rating=1,
            signals={"source_count": 1, "confidence": 0.45, "has_numbers": False},
        ))
    report = analyze_patterns()
    assert isinstance(report, PatternReport)
    assert report.analyzed_count >= 6


def test_f23_analyze_high_patterns_detected():
    for i in range(3):
        submit_feedback(FeedbackRecord(
            query_id=f"q{i}", rating=5,
            signals={"source_count": 4, "confidence": 0.95},
        ))
    report = analyze_patterns()
    assert len(report.high_quality_patterns) >= 1


def test_f23_analyze_low_patterns_detected():
    for i in range(3):
        submit_feedback(FeedbackRecord(
            query_id=f"q{i}", rating=1,
            signals={"source_count": 1, "confidence": 0.40},
        ))
    report = analyze_patterns()
    assert len(report.low_quality_patterns) >= 1


def test_f23_analyze_empty_feedback():
    report = analyze_patterns()
    assert report.analyzed_count == 0


# ── get_insights ───────────────────────────────────────────────────────────────

def test_f23_insights_returns_recommendations():
    for i in range(5):
        submit_feedback(FeedbackRecord(
            query_id=f"q{i}", rating=2,
            signals={"source_count": 1, "confidence": 0.50},
        ))
    insights = get_insights()
    assert "recommendations" in insights
    assert isinstance(insights["recommendations"], list)


def test_f23_insights_has_summary():
    submit_feedback(FeedbackRecord(query_id="q1", rating=5))
    insights = get_insights()
    assert "summary"        in insights
    assert "total_feedback" in insights["summary"]


# ── A/B testing ───────────────────────────────────────────────────────────────

def test_f23_create_ab_test():
    test = create_ab_test("test-1", ["control", "treatment"])
    assert isinstance(test, ABTest)
    assert test.name == "test-1"
    assert set(test.variants.keys()) == {"control", "treatment"}


def test_f23_ab_test_stored():
    create_ab_test("test-1", ["A", "B"])
    tests = list_ab_tests()
    assert any(t.name == "test-1" for t in tests)


def test_f23_create_ab_test_duplicate_raises():
    create_ab_test("test-1", ["A", "B"])
    with pytest.raises(ValueError, match="already exists"):
        create_ab_test("test-1", ["A", "B"])


def test_f23_assign_variant_returns_valid():
    create_ab_test("test-1", ["A", "B"])
    variant = assign_variant("test-1", "user-42")
    assert variant in ["A", "B"]


def test_f23_assign_variant_deterministic():
    create_ab_test("test-1", ["A", "B"])
    v1 = assign_variant("test-1", "user-42")
    v2 = assign_variant("test-1", "user-42")
    assert v1 == v2


def test_f23_assign_variant_different_users_may_differ():
    create_ab_test("test-1", ["A", "B"])
    variants = {assign_variant("test-1", f"user-{i}") for i in range(20)}
    assert len(variants) == 2


def test_f23_record_ab_result():
    create_ab_test("test-1", ["control", "treatment"])
    record_ab_result("test-1", "control",   3.5)
    record_ab_result("test-1", "treatment", 4.5)
    results = get_ab_results("test-1")
    assert results["control"]["count"]       == 1
    assert results["treatment"]["count"]     == 1
    assert results["control"]["avg_rating"]  == 3.5
    assert results["treatment"]["avg_rating"] == 4.5


def test_f23_ab_test_winner():
    create_ab_test("test-1", ["control", "treatment"])
    for _ in range(3):
        record_ab_result("test-1", "control",   3.0)
    for _ in range(3):
        record_ab_result("test-1", "treatment", 4.5)
    results = get_ab_results("test-1")
    assert results["winner"] == "treatment"


def test_f23_ab_winner_none_with_no_results():
    create_ab_test("test-1", ["A", "B"])
    results = get_ab_results("test-1")
    assert results["winner"] is None


def test_f23_record_ab_result_unknown_test_raises():
    with pytest.raises(KeyError):
        record_ab_result("ghost", "A", 4.0)


def test_f23_record_ab_result_unknown_variant_raises():
    create_ab_test("test-1", ["A", "B"])
    with pytest.raises(ValueError):
        record_ab_result("test-1", "C", 4.0)


# ── API ───────────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f23_api_submit_feedback():
    with _client() as c:
        resp = c.post("/api/v1/feedback",
                      json={"query_id": "q1", "rating": 4}, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["rating"] == 4
    assert "id" in data


def test_f23_api_submit_invalid_rating():
    with _client() as c:
        resp = c.post("/api/v1/feedback",
                      json={"query_id": "q1", "rating": 6}, headers=AUTH)
    assert resp.status_code == 422


def test_f23_api_list_feedback():
    with _client() as c:
        c.post("/api/v1/feedback", json={"query_id": "q1", "rating": 5}, headers=AUTH)
        c.post("/api/v1/feedback", json={"query_id": "q2", "rating": 3}, headers=AUTH)
        resp = c.get("/api/v1/feedback", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2


def test_f23_api_feedback_stats():
    with _client() as c:
        c.post("/api/v1/feedback", json={"query_id": "q1", "rating": 5}, headers=AUTH)
        c.post("/api/v1/feedback", json={"query_id": "q2", "rating": 3}, headers=AUTH)
        resp = c.get("/api/v1/feedback/stats", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "average_rating"   in data
    assert "distribution"     in data
    assert "high_rated_count" in data


def test_f23_api_insights():
    with _client() as c:
        resp = c.get("/api/v1/feedback/insights", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "recommendations" in data
    assert "summary"         in data


def test_f23_api_create_ab_test():
    with _client() as c:
        resp = c.post("/api/v1/feedback/ab", json={
            "name": "search-v2", "variants": ["control", "treatment"],
        }, headers=AUTH)
    assert resp.status_code == 201
    assert resp.json()["name"] == "search-v2"


def test_f23_api_list_ab_tests():
    with _client() as c:
        c.post("/api/v1/feedback/ab",
               json={"name": "t1", "variants": ["A", "B"]}, headers=AUTH)
        resp = c.get("/api/v1/feedback/ab", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_f23_api_get_ab_results():
    with _client() as c:
        c.post("/api/v1/feedback/ab",
               json={"name": "t2", "variants": ["A", "B"]}, headers=AUTH)
        resp = c.get("/api/v1/feedback/ab/t2", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "A" in data
    assert "B" in data


def test_f23_api_assign_variant():
    with _client() as c:
        c.post("/api/v1/feedback/ab",
               json={"name": "t3", "variants": ["A", "B"]}, headers=AUTH)
        resp = c.post("/api/v1/feedback/ab/t3/assign",
                      json={"user_id": "user-42"}, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["variant"] in ["A", "B"]
    assert data["test_id"] == "t3"


def test_f23_api_record_ab_result():
    with _client() as c:
        c.post("/api/v1/feedback/ab",
               json={"name": "t4", "variants": ["A", "B"]}, headers=AUTH)
        resp = c.post("/api/v1/feedback/ab/t4/record",
                      json={"variant": "A", "rating": 4.5}, headers=AUTH)
    assert resp.status_code == 201


def test_f23_api_reset():
    with _client() as c:
        c.post("/api/v1/feedback",
               json={"query_id": "q1", "rating": 5}, headers=AUTH)
        resp = c.delete("/api/v1/feedback/reset", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_f23_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/feedback/stats")
    assert resp.status_code == 401
