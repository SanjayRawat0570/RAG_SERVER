"""Tests for F24: Cost Optimization."""
from __future__ import annotations

import pytest

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
from app.rag.cost.pricing import PRICING


@pytest.fixture(autouse=True)
def _clean():
    reset_tracker()
    reset_budgets()
    yield
    reset_tracker()
    reset_budgets()


# ── Pricing & estimation ───────────────────────────────────────────────────────

def test_f24_count_tokens_nonzero():
    assert count_tokens("hello world this is a test") > 0


def test_f24_count_tokens_empty_gives_minimum():
    assert count_tokens("") >= 1


def test_f24_estimate_cost_free_model():
    assert estimate_cost("extractive-stub", 10_000, 5_000) == 0.0


def test_f24_estimate_cost_gemini_flash_free():
    assert estimate_cost("gemini-2.5-flash", 10_000, 5_000) == 0.0


def test_f24_estimate_cost_gpt4o():
    cost = estimate_cost("gpt-4o", 1_000, 200)
    assert cost > 0.0


def test_f24_estimate_cost_unknown_model_is_free():
    assert estimate_cost("no-such-model", 5_000, 1_000) == 0.0


def test_f24_pricing_table_has_common_models():
    assert "gpt-4o"      in PRICING
    assert "gpt-4o-mini" in PRICING
    assert "extractive-stub" in PRICING


# ── Budget enforcement ─────────────────────────────────────────────────────────

def test_f24_reserve_within_budget():
    reserve("u1", 10.0, 0.5)   # should not raise


def test_f24_reserve_exceeds_budget():
    record_spend("u1", 10.0, 9.8)
    with pytest.raises(BudgetExceededError):
        reserve("u1", 10.0, 0.5)   # 9.8 + 0.5 = 10.3 > 10.0


def test_f24_record_spend_accumulates():
    record_spend("u1", 10.0, 3.0)
    record_spend("u1", 10.0, 2.5)
    stats = budget_stats()
    assert abs(stats["u1"]["spent"] - 5.5) < 1e-5


def test_f24_budget_stats_empty():
    assert budget_stats() == {}


def test_f24_budget_stats_shows_remaining():
    record_spend("u1", 10.0, 3.0)
    stats = budget_stats()
    assert stats["u1"]["spent"]     == 3.0
    assert stats["u1"]["remaining"] == 7.0
    assert stats["u1"]["limit"]     == 10.0


def test_f24_reset_budgets_all():
    record_spend("u1", 10.0, 5.0)
    record_spend("u2", 20.0, 1.0)
    reset_budgets()
    assert budget_stats() == {}


def test_f24_reset_budgets_single_key():
    record_spend("u1", 10.0, 5.0)
    record_spend("u2", 20.0, 1.0)
    reset_budgets("u1")
    stats = budget_stats()
    assert "u1" not in stats
    assert "u2" in stats


def test_f24_upsert_budget_creates_entry():
    result = upsert_budget("t1", 50.0)
    assert result["key"]   == "t1"
    assert result["limit"] == 50.0
    assert result["spent"] == 0.0


# ── Usage tracker ─────────────────────────────────────────────────────────────

def test_f24_record_usage_creates_event():
    event = record_usage(user_id="alice", model="gpt-4o-mini",
                         input_tokens=500, output_tokens=100)
    assert event.id     != ""
    assert event.cost   > 0.0
    assert event.cached is False


def test_f24_record_usage_cached_is_free():
    event = record_usage(model="gpt-4o", input_tokens=500,
                         output_tokens=200, cached=True)
    assert event.cost   == 0.0
    assert event.cached is True


def test_f24_get_usage_by_user():
    record_usage(user_id="alice", model="gpt-4o-mini")
    record_usage(user_id="bob",   model="gpt-4o")
    events = get_usage(user_id="alice")
    assert all(e.user_id == "alice" for e in events)
    assert len(events) == 1


def test_f24_get_usage_by_model():
    record_usage(model="gpt-4o-mini")
    record_usage(model="gpt-4o")
    events = get_usage(model="gpt-4o-mini")
    assert all(e.model == "gpt-4o-mini" for e in events)


def test_f24_usage_summary_total_cost():
    record_usage(model="gpt-4o", input_tokens=1000, output_tokens=200)
    summary = usage_summary()
    assert summary["total_cost"]     > 0.0
    assert summary["total_requests"] == 1


def test_f24_usage_summary_by_model():
    record_usage(model="gpt-4o",      input_tokens=1000, output_tokens=200)
    record_usage(model="gpt-4o-mini", input_tokens=500,  output_tokens=100)
    summary = usage_summary()
    assert "gpt-4o"      in summary["by_model"]
    assert "gpt-4o-mini" in summary["by_model"]


def test_f24_usage_summary_cached_requests():
    record_usage(model="gpt-4o", cached=False)
    record_usage(model="gpt-4o", cached=True)
    summary = usage_summary()
    assert summary["cached_requests"] == 1
    assert summary["cache_hit_rate"]  == 0.5


def test_f24_usage_summary_empty():
    summary = usage_summary()
    assert summary["total_cost"]     == 0.0
    assert summary["total_requests"] == 0


# ── Optimizer ─────────────────────────────────────────────────────────────────

def test_f24_recommend_model_returns_valid():
    result = recommend_model(100.0)
    assert "model"          in result
    assert "estimated_cost" in result
    assert "quality_tier"   in result


def test_f24_recommend_within_budget():
    # gpt-4o-mini costs ~$0.00027, gpt-4o costs ~$0.008 per call.
    # budget=0.0002 means only free (cost=0.0) models are affordable.
    result = recommend_model(budget_remaining=0.0002, avg_input_tokens=1000, avg_output_tokens=200)
    assert result["estimated_cost"] == 0.0


def test_f24_recommend_prefer_quality_gets_highest_tier():
    result = recommend_model(1.0, avg_input_tokens=1000, avg_output_tokens=200,
                             prefer_quality=True)
    assert result["quality_tier"] >= 3   # should pick a premium model


def test_f24_recommend_prefer_cheap_gets_low_cost():
    r1 = recommend_model(1.0, prefer_quality=False)
    r2 = recommend_model(1.0, prefer_quality=True)
    # Cheapest model should cost no more than best-quality model
    assert r1["estimated_cost"] <= r2["estimated_cost"]


def test_f24_model_comparison_sorted_by_cost():
    results = model_comparison(["gpt-4o", "gpt-4o-mini", "extractive-stub"], 1000, 200)
    costs = [r["cost"] for r in results]
    assert costs == sorted(costs)
    assert results[0]["model"] == "extractive-stub"


def test_f24_model_comparison_structure():
    results = model_comparison(["gpt-4o-mini"], 500, 100)
    assert len(results) == 1
    assert "model"         in results[0]
    assert "cost"          in results[0]
    assert "input_tokens"  in results[0]
    assert "output_tokens" in results[0]


def test_f24_cache_savings_80pct():
    result = cache_savings(1000, 0.8, "gpt-4o", 500, 200)
    assert result["savings_pct"]    == 80.0
    assert result["new_queries"]    == 200
    assert result["cached_queries"] == 800
    assert result["savings"]        > 0.0


def test_f24_cache_savings_zero_hit_rate():
    result = cache_savings(100, 0.0, "gpt-4o-mini")
    assert result["savings"]   == 0.0
    assert result["new_queries"] == 100


def test_f24_cache_savings_full_hit_rate():
    result = cache_savings(100, 1.0, "gpt-4o")
    assert result["new_queries"]    == 0
    assert result["cached_queries"] == 100
    assert result["savings"]        == result["original_cost"]


# ── API ───────────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f24_api_get_pricing():
    with _client() as c:
        resp = c.get("/api/v1/cost/pricing", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "pricing" in data
    assert "gpt-4o" in data["pricing"]


def test_f24_api_estimate_cost():
    with _client() as c:
        resp = c.post("/api/v1/cost/estimate", json={
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 200,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["estimated_cost"] > 0
    assert "price_per_1k_input"  in data


def test_f24_api_get_budgets_empty():
    with _client() as c:
        resp = c.get("/api/v1/cost/budgets", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["budgets"] == {}


def test_f24_api_create_budget():
    with _client() as c:
        resp = c.post("/api/v1/cost/budgets",
                      json={"key": "tenant-1", "limit": 50.0}, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["key"]   == "tenant-1"
    assert data["limit"] == 50.0


def test_f24_api_reserve_allowed():
    with _client() as c:
        c.post("/api/v1/cost/budgets", json={"key": "t1", "limit": 10.0}, headers=AUTH)
        resp = c.post("/api/v1/cost/budgets/reserve", json={
            "key": "t1", "limit": 10.0, "estimated_cost": 0.5,
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["allowed"] is True


def test_f24_api_reserve_denied():
    with _client() as c:
        # Spend up to near the limit, then try to reserve more
        c.post("/api/v1/cost/usage", json={
            "model": "gpt-4o", "operation": "llm",
            "input_tokens": 0, "output_tokens": 0,
            "cached": False, "tenant": "default",
        }, headers=AUTH)
        resp = c.post("/api/v1/cost/budgets/reserve", json={
            "key": "t1", "limit": 0.001, "estimated_cost": 10.0,
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["allowed"] is False


def test_f24_api_record_usage():
    with _client() as c:
        resp = c.post("/api/v1/cost/usage", json={
            "model": "gpt-4o-mini", "operation": "llm",
            "input_tokens": 500, "output_tokens": 100,
        }, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["cost"] > 0
    assert "id" in data


def test_f24_api_record_cached_usage():
    with _client() as c:
        resp = c.post("/api/v1/cost/usage", json={
            "model": "gpt-4o", "cached": True,
            "input_tokens": 1000, "output_tokens": 500,
        }, headers=AUTH)
    assert resp.status_code == 201
    assert resp.json()["cost"]   == 0.0
    assert resp.json()["cached"] is True


def test_f24_api_list_usage():
    with _client() as c:
        c.post("/api/v1/cost/usage", json={"model": "gpt-4o-mini"}, headers=AUTH)
        c.post("/api/v1/cost/usage", json={"model": "gpt-4o"},      headers=AUTH)
        resp = c.get("/api/v1/cost/usage", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2


def test_f24_api_usage_summary():
    with _client() as c:
        c.post("/api/v1/cost/usage", json={
            "model": "gpt-4o", "input_tokens": 1000, "output_tokens": 200,
        }, headers=AUTH)
        resp = c.get("/api/v1/cost/summary", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_cost"      in data
    assert "by_model"        in data
    assert "by_operation"    in data
    assert "cache_hit_rate"  in data


def test_f24_api_recommend_model():
    with _client() as c:
        resp = c.post("/api/v1/cost/recommend", json={
            "budget_remaining": 1.0, "prefer_quality": True,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "model"          in data
    assert "estimated_cost" in data
    assert "quality_tier"   in data


def test_f24_api_compare_models():
    with _client() as c:
        resp = c.post("/api/v1/cost/compare", json={
            "models": ["gpt-4o", "gpt-4o-mini", "extractive-stub"],
            "input_tokens": 1000, "output_tokens": 200,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"])   == 3
    assert data["cheapest"]      == "extractive-stub"


def test_f24_api_cache_savings():
    with _client() as c:
        resp = c.post("/api/v1/cost/savings/cache", json={
            "total_queries": 1000, "cache_hit_rate": 0.8,
            "model": "gpt-4o", "avg_input_tokens": 500, "avg_output_tokens": 200,
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["savings_pct"]    == 80.0
    assert data["savings"]        > 0.0


def test_f24_api_reset():
    with _client() as c:
        c.post("/api/v1/cost/usage", json={"model": "gpt-4o-mini"}, headers=AUTH)
        resp = c.delete("/api/v1/cost/reset", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_f24_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/cost/pricing")
    assert resp.status_code == 401
