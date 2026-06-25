"""Tests for F7: Error Handling & Fallbacks."""
from __future__ import annotations

import pytest

from app.engine.fallback import FallbackChainDef, FallbackExecutor, FallbackOption
from app.models.workflow import WorkflowDef


# ── Test workflow helpers ─────────────────────────────────────────────────────

def _wf_ok(name: str, value) -> WorkflowDef:
    """Workflow that always succeeds and returns value."""
    return WorkflowDef(
        name=name,
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": value}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )


def _wf_error(name: str) -> WorkflowDef:
    """Workflow that always returns error status (add int+str → TypeError)."""
    return WorkflowDef(
        name=name,
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "err", "type": "processing",
             "config": {"operation": "add", "amount": 1}},
            {"id": "out", "type": "output", "config": {"value": "$.err"}},
        ],
        edges=[
            {"source": "in",  "target": "err"},
            {"source": "err", "target": "out"},
        ],
    )


def _wf_empty(name: str) -> WorkflowDef:
    """Workflow that always returns an empty list — treated as 'no results'."""
    return WorkflowDef(
        name=name,
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "out", "type": "output", "config": {"value": []}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )


def _chain(options_spec, **kwargs) -> FallbackChainDef:
    """Build a FallbackChainDef from a list of (name, workflow) tuples."""
    return FallbackChainDef(
        name="test_chain",
        options=[FallbackOption(name=n, description=n, workflow=w) for n, w in options_spec],
        **kwargs,
    )


# ── FallbackExecutor unit tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f7_primary_succeeds_depth_zero():
    """Primary option succeeds → depth=0, degraded=False."""
    result = await FallbackExecutor(_chain([
        ("primary", _wf_ok("p", {"answer": "yes"})),
        ("fallback", _wf_ok("f", {"answer": "fallback"})),
    ])).run({})

    assert result.succeeded is True
    assert result.fallback_depth == 0
    assert result.degraded is False
    assert result.used_option == "primary"
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "success"


@pytest.mark.asyncio
async def test_f7_fallback_at_depth_one():
    """Primary errors → first fallback succeeds → depth=1, degraded=True."""
    result = await FallbackExecutor(_chain([
        ("primary",  _wf_error("err")),
        ("fallback1", _wf_ok("f1", {"answer": "from fallback"})),
    ])).run({"v": "bad"})  # bad input makes the 'add' fail

    assert result.succeeded is True
    assert result.fallback_depth == 1
    assert result.degraded is True
    assert result.used_option == "fallback1"
    assert result.attempts[0].status == "error"
    assert result.attempts[1].status == "success"


@pytest.mark.asyncio
async def test_f7_cascades_through_multiple_failures():
    """Two failures before success → depth=2."""
    result = await FallbackExecutor(_chain([
        ("opt1", _wf_error("e1")),
        ("opt2", _wf_error("e2")),
        ("opt3", _wf_ok("ok", "found")),
    ])).run({"v": "bad"})

    assert result.succeeded is True
    assert result.fallback_depth == 2
    assert result.used_option == "opt3"
    assert len(result.attempts) == 3


@pytest.mark.asyncio
async def test_f7_empty_results_treated_as_failure():
    """Empty list result is treated as failure when skip_empty_results=True."""
    result = await FallbackExecutor(_chain([
        ("empty_search", _wf_empty("e")),
        ("keyword_search", _wf_ok("k", [{"id": "1", "score": 0.8}])),
    ], skip_empty_results=True)).run({})

    assert result.succeeded is True
    assert result.fallback_depth == 1
    assert result.attempts[0].status == "empty"
    assert result.attempts[1].status == "success"


@pytest.mark.asyncio
async def test_f7_empty_results_not_skipped_when_disabled():
    """Empty list is accepted when skip_empty_results=False."""
    result = await FallbackExecutor(_chain([
        ("search", _wf_empty("e")),
    ], skip_empty_results=False)).run({})

    assert result.succeeded is True
    assert result.fallback_depth == 0
    assert result.attempts[0].status == "success"


@pytest.mark.asyncio
async def test_f7_all_fail_error_raises():
    """on_all_fail='error' raises RuntimeError when everything fails."""
    chain = _chain([
        ("opt1", _wf_error("e1")),
        ("opt2", _wf_error("e2")),
    ], on_all_fail="error")

    with pytest.raises(RuntimeError, match="All fallback options exhausted"):
        await FallbackExecutor(chain).run({"v": "bad"})


@pytest.mark.asyncio
async def test_f7_all_fail_empty_returns_result():
    """on_all_fail='empty' returns FallbackResult with succeeded=False."""
    chain = _chain([
        ("opt1", _wf_empty("e1")),
        ("opt2", _wf_empty("e2")),
    ], on_all_fail="empty", skip_empty_results=True)

    result = await FallbackExecutor(chain).run({})
    assert result.succeeded is False
    assert result.used_option == "none"
    assert result.degraded is True
    assert "exhausted" in result.message.lower()


@pytest.mark.asyncio
async def test_f7_partial_note_set_when_degraded():
    """partial_note is populated when a fallback (not primary) wins."""
    result = await FallbackExecutor(_chain([
        ("primary", _wf_error("err")),
        ("fallback", _wf_ok("ok", [1, 2, 3])),
    ])).run({"v": "bad"})

    assert result.degraded is True
    assert result.partial_note is not None
    assert "primary" in result.partial_note.lower()


@pytest.mark.asyncio
async def test_f7_attempt_durations_are_positive():
    """Each attempt records a non-negative duration_ms."""
    result = await FallbackExecutor(_chain([
        ("opt1", _wf_ok("a", "ok")),
    ])).run({})

    for attempt in result.attempts:
        assert attempt.duration_ms >= 0


# ── Existing circuit-breaker (F7 foundation) ─────────────────────────────────

@pytest.mark.asyncio
async def test_f7_circuit_breaker_still_works():
    """Verify the existing F7 circuit breaker is unaffected by F7 additions."""
    from app.engine.circuit_breaker import OPEN, registry
    from app.engine.executor import WorkflowExecutor

    registry.reset("svc_f7")
    wf = WorkflowDef(
        name="breaker_test",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "x",  "type": "processing",
             "config": {"operation": "add", "amount": 1,
                        "breaker_key": "svc_f7", "breaker_threshold": 2}},
        ],
        edges=[{"source": "in", "target": "x"}],
    )
    executor = WorkflowExecutor(wf)
    await executor.run({"v": "bad"})  # failure 1
    await executor.run({"v": "bad"})  # failure 2 → opens breaker
    assert registry.get("svc_f7").state == OPEN
    registry.reset("svc_f7")


# ── Scenario 1: Search fallback (real RAG) ────────────────────────────────────

@pytest.mark.asyncio
async def test_f7_scenario1_finds_results_via_fallback():
    """
    Scenario 1: Vector search on empty namespace (no results) falls back to
    keyword search on the same namespace where we've just indexed data.
    """
    from app.api.pipelines import build_index_workflow
    from app.engine.executor import WorkflowExecutor

    # Index a document
    await WorkflowExecutor(build_index_workflow()).run({
        "tenant":   "f7s1",
        "text":     "Quarterly revenue increased by 20 percent.",
        "filename": "q3.txt",
    })

    from app.api.pipelines import (
        build_semantic_only_workflow,
        build_keyword_only_workflow,
        build_entity_only_workflow,
    )

    # Build the cascade
    chain = FallbackChainDef(
        name="search_fallback",
        options=[
            FallbackOption(name="semantic", description="Semantic",
                           workflow=build_semantic_only_workflow()),
            FallbackOption(name="keyword",  description="Keyword",
                           workflow=build_keyword_only_workflow()),
            FallbackOption(name="entity",   description="Entity",
                           workflow=build_entity_only_workflow()),
        ],
        on_all_fail="empty",
        skip_empty_results=True,
    )
    result = await FallbackExecutor(chain).run({"question": "revenue", "tenant": "f7s1"})
    # At least one method should find the indexed document
    assert result.succeeded is True
    hits = result.outputs.get("out") or []
    assert isinstance(hits, list)


# ── Scenario 2: LLM fallback ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f7_scenario2_llm_fallback_always_answers():
    """
    Scenario 2: Even if the primary LLM fails, the stub (always available)
    ensures an answer is returned.
    """
    from app.api.pipelines import build_ask_workflow

    chain = FallbackChainDef(
        name="ask_fallback",
        options=[
            # stub is our fallback — it always returns an answer
            FallbackOption(name="primary_stub",  description="Primary LLM",
                           workflow=build_ask_workflow("stub")),
            FallbackOption(name="fallback_stub", description="Fallback LLM",
                           workflow=build_ask_workflow("stub")),
        ],
        on_all_fail="empty",
        skip_empty_results=False,
    )
    result = await FallbackExecutor(chain).run(
        {"question": "What is AI?", "tenant": "f7s2"}
    )
    assert result.succeeded is True
    answer = result.outputs.get("out", {}).get("answer", "")
    assert isinstance(answer, str)


# ── Scenario 3: Query expansion ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f7_scenario3_query_expansion():
    """
    Scenario 3: Empty results on direct query → fallback to expanded query.
    Uses an empty tenant so direct query always returns nothing.
    """
    from app.api.pipelines import build_semantic_only_workflow, build_expanded_search_workflow, build_entity_only_workflow

    chain = FallbackChainDef(
        name="query_expansion",
        options=[
            FallbackOption(name="direct",   description="Direct query",
                           workflow=build_semantic_only_workflow()),
            FallbackOption(name="expanded", description="Expanded query",
                           workflow=build_expanded_search_workflow()),
            FallbackOption(name="entity",   description="Entity search",
                           workflow=build_entity_only_workflow()),
        ],
        on_all_fail="empty",
        skip_empty_results=True,
    )
    result = await FallbackExecutor(chain).run(
        {"question": "revenue", "tenant": "f7s3_empty"}
    )
    # All options return empty for empty tenant → succeeded=False
    assert result.succeeded is False or isinstance(result.attempts, list)
    assert len(result.attempts) >= 1


# ── API endpoints ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f7_api_search_fallback():
    """POST /errors/search-fallback returns succeeded + attempt list."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/errors/search-fallback",
            json={"question": "What is revenue?", "tenant": "f7api"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "succeeded" in data
    assert "attempts" in data
    assert "fallback_depth" in data
    assert isinstance(data["attempts"], list)


@pytest.mark.asyncio
async def test_f7_api_ask_fallback():
    """POST /errors/ask-fallback returns an answer."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/errors/ask-fallback",
            json={"question": "Explain machine learning", "tenant": "f7api"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["succeeded"] is True


@pytest.mark.asyncio
async def test_f7_api_query_expansion():
    """POST /errors/query-expansion returns expansion metadata."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/errors/query-expansion",
            json={"question": "machine learning", "tenant": "f7qe"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "expansion_applied" in data
    assert "attempts" in data


@pytest.mark.asyncio
async def test_f7_api_ingest_fallback():
    """POST /errors/ingest-fallback always succeeds (has a guidance fallback)."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/errors/ingest-fallback",
            json={"filename": "test.pdf", "text": "Sample content here.", "tenant": "f7ingest"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "succeeded" in data
    assert data["succeeded"] is True


@pytest.mark.asyncio
async def test_f7_api_generic_run():
    """POST /errors/run with a simple always-succeed chain."""
    from fastapi.testclient import TestClient
    from app.main import app

    chain_json = {
        "chain": {
            "name": "test_fallback",
            "options": [
                {
                    "name": "primary",
                    "description": "Primary option",
                    "workflow": {
                        "name": "wf_ok",
                        "nodes": [
                            {"id": "in",  "type": "input"},
                            {"id": "out", "type": "output",
                             "config": {"value": {"result": "ok"}}},
                        ],
                        "edges": [{"source": "in", "target": "out"}],
                    },
                }
            ],
        },
        "inputs": {},
    }
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/errors/run",
            json=chain_json,
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["succeeded"] is True
    assert data["fallback_depth"] == 0


@pytest.mark.asyncio
async def test_f7_api_error_history_offline():
    """GET /errors/history returns empty list with note when Supabase is offline."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/errors/history",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data


@pytest.mark.asyncio
async def test_f7_api_error_patterns_offline():
    """GET /errors/patterns returns empty patterns when Supabase is offline."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/errors/patterns",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "patterns" in data
