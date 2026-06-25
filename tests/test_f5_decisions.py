"""Tests for F5: Conditional Logic & Decision Trees."""
from __future__ import annotations

import pytest

from app.engine.executor import WorkflowExecutor
from app.engine.context import ExecutionContext
from app.models.workflow import WorkflowDef
from app.api.pipelines import (
    build_doc_type_router_workflow,
    build_search_strategy_workflow,
    build_confidence_router_workflow,
    build_index_workflow,
    STORE, DIM,
)


# ── ClassifyNode unit tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f5_classify_matches_first_rule():
    """First matching rule wins; returns correct category + confidence."""
    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.filename",
                "rules": [
                    {"category": "pdf",   "when": {"left": "$.inputs.filename", "op": "endswith", "right": ".pdf"}, "confidence": 0.99},
                    {"category": "docx",  "when": {"left": "$.inputs.filename", "op": "endswith", "right": ".docx"}, "confidence": 0.90},
                ],
                "default": "text",
                "default_confidence": 0.5,
            }},
            {"id": "out", "type": "output", "config": {"value": "$.cls"}},
        ],
        edges=[{"source": "in", "target": "cls"}, {"source": "cls", "target": "out"}],
    )
    res = await WorkflowExecutor(wf).run({"filename": "report.pdf"})
    assert res.status == "success"
    assert res.outputs["out"]["category"] == "pdf"
    assert res.outputs["out"]["confidence"] == 0.99


@pytest.mark.asyncio
async def test_f5_classify_falls_back_to_default():
    """No matching rule → default category and confidence."""
    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.ext",
                "rules": [
                    {"category": "pdf", "when": {"left": "$.inputs.ext", "op": "==", "right": ".pdf"}, "confidence": 0.99},
                ],
                "default": "unknown",
                "default_confidence": 0.3,
            }},
            {"id": "out", "type": "output", "config": {"value": "$.cls"}},
        ],
        edges=[{"source": "in", "target": "cls"}, {"source": "cls", "target": "out"}],
    )
    res = await WorkflowExecutor(wf).run({"ext": ".xyz"})
    assert res.outputs["out"]["category"] == "unknown"
    assert res.outputs["out"]["confidence"] == 0.3
    assert res.outputs["out"]["matched_rule"] == "default"


@pytest.mark.asyncio
async def test_f5_classify_or_condition():
    """OR rule matches any of the listed values."""
    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.filename",
                "rules": [
                    {"category": "image",
                     "when": {"or": [
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".png"},
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".jpg"},
                     ]},
                     "confidence": 0.99},
                ],
                "default": "text",
            }},
            {"id": "out", "type": "output", "config": {"value": "$.cls"}},
        ],
        edges=[{"source": "in", "target": "cls"}, {"source": "cls", "target": "out"}],
    )
    for fname in ("photo.png", "photo.jpg"):
        res = await WorkflowExecutor(wf).run({"filename": fname})
        assert res.outputs["out"]["category"] == "image", f"failed for {fname}"


@pytest.mark.asyncio
async def test_f5_classify_and_condition():
    """AND rule requires all conditions to be true."""
    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.score",
                "rules": [
                    {"category": "accept",
                     "when": {"and": [
                         {"left": "$.inputs.score",   "op": ">",  "right": 0.5},
                         {"left": "$.inputs.results", "op": ">",  "right": 0},
                     ]},
                     "confidence": 0.90},
                ],
                "default": "reject",
            }},
            {"id": "out", "type": "output", "config": {"value": "$.cls"}},
        ],
        edges=[{"source": "in", "target": "cls"}, {"source": "cls", "target": "out"}],
    )
    # Both conditions met → accept
    res = await WorkflowExecutor(wf).run({"score": 0.8, "results": 3})
    assert res.outputs["out"]["category"] == "accept"

    # Score fails → reject
    res2 = await WorkflowExecutor(wf).run({"score": 0.3, "results": 3})
    assert res2.outputs["out"]["category"] == "reject"


@pytest.mark.asyncio
async def test_f5_classify_label_in_matched_rule():
    """matched_rule reflects the label when provided."""
    wf = WorkflowDef(
        name="t",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.v",
                "rules": [
                    {"category": "hit", "label": "my-label",
                     "when": {"left": "$.inputs.v", "op": "==", "right": "x"}},
                ],
                "default": "miss",
            }},
            {"id": "out", "type": "output", "config": {"value": "$.cls"}},
        ],
        edges=[{"source": "in", "target": "cls"}, {"source": "cls", "target": "out"}],
    )
    res = await WorkflowExecutor(wf).run({"v": "x"})
    assert res.outputs["out"]["matched_rule"] == "my-label"


# ── Example 1: Document Type Routing ─────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("filename,expected_parser", [
    ("report.pdf",       "pypdf"),
    ("slides.docx",      "python-docx"),
    ("photo.png",        "tesseract"),
    ("photo.jpg",        "tesseract"),
    ("page.html",        "beautifulsoup"),
    ("data.txt",         "generic"),
    ("notes.md",         "generic"),
])
async def test_f5_doc_type_router(filename, expected_parser):
    """Document Type Routing: each extension routes to the correct parser."""
    wf  = build_doc_type_router_workflow()
    res = await WorkflowExecutor(wf).run({"filename": filename})
    assert res.status == "success"
    parser_info = res.outputs.get("out") or {}
    assert parser_info.get("parser") == expected_parser, (
        f"{filename} → expected parser '{expected_parser}' got '{parser_info.get('parser')}'"
    )


@pytest.mark.asyncio
async def test_f5_doc_type_router_skips_wrong_branches():
    """Only the matching parser branch should run; others are skipped."""
    wf  = build_doc_type_router_workflow()
    res = await WorkflowExecutor(wf).run({"filename": "deck.pdf"})
    statuses = {r.node_id: r.status for r in res.results}
    assert statuses["pdf_parse"]   == "success"
    assert statuses["word_parse"]  == "skipped"
    assert statuses["image_parse"] == "skipped"


# ── Example 2: Search Strategy Selection ─────────────────────────────────────

@pytest.mark.asyncio
async def test_f5_search_strategy_question_routes_semantic():
    """A plain question should be classified as semantic search."""
    wf  = build_search_strategy_workflow()
    res = await WorkflowExecutor(wf).run({"question": "What is machine learning?", "tenant": "t"})
    assert res.status == "success"
    out = res.outputs.get("out") or {}
    assert out.get("strategy") == "semantic"


@pytest.mark.asyncio
async def test_f5_search_strategy_quoted_routes_keyword():
    """A query with a quoted phrase should be classified as keyword search."""
    wf  = build_search_strategy_workflow()
    res = await WorkflowExecutor(wf).run({"question": 'Find "quarterly revenue" in reports', "tenant": "t"})
    assert res.status == "success"
    out = res.outputs.get("out") or {}
    assert out.get("strategy") == "keyword"


@pytest.mark.asyncio
async def test_f5_search_strategy_command_routes_keyword():
    """A command verb query should route to keyword search."""
    wf  = build_search_strategy_workflow()
    res = await WorkflowExecutor(wf).run({"question": "list all documents about AI", "tenant": "t"})
    assert res.status == "success"
    out = res.outputs.get("out") or {}
    # "list" triggers command intent → keyword
    assert out.get("strategy") in ("keyword", "hybrid")


@pytest.mark.asyncio
async def test_f5_search_strategy_only_one_branch_runs():
    """Only one search-strategy branch should run; others are skipped."""
    wf  = build_search_strategy_workflow()
    res = await WorkflowExecutor(wf).run({"question": "What is AI?", "tenant": "t"})
    statuses = {r.node_id: r.status for r in res.results}
    active = [nid for nid, st in statuses.items()
              if nid.startswith("use_") and st == "success"]
    skipped = [nid for nid, st in statuses.items()
               if nid.startswith("use_") and st == "skipped"]
    assert len(active) == 1
    assert len(skipped) == 3


# ── Example 3: Response Quality / Confidence Routing ─────────────────────────

@pytest.mark.asyncio
async def test_f5_confidence_router_no_docs_cannot_answer():
    """With an empty store the search returns nothing → cannot_answer branch."""
    wf  = build_confidence_router_workflow()
    # Use a tenant with no indexed data
    res = await WorkflowExecutor(wf).run({"question": "Totally obscure query xyz123?", "tenant": "empty_tenant"})
    assert res.status == "success"
    action = res.outputs.get("out") or {}
    assert action.get("action") == "cannot_answer_confidently"


@pytest.mark.asyncio
async def test_f5_confidence_router_with_docs_routes_high_or_medium():
    """After indexing relevant content the confidence route should not be 'none'."""
    # Index a document first
    await WorkflowExecutor(build_index_workflow()).run({
        "tenant": "f5conf",
        "text":   "Machine learning is a subset of artificial intelligence.",
        "filename": "ml.txt",
    })
    wf  = build_confidence_router_workflow()
    res = await WorkflowExecutor(wf).run({"question": "What is machine learning?", "tenant": "f5conf"})
    assert res.status == "success"
    action = res.outputs.get("out") or {}
    assert action.get("action") in (
        "return_immediately", "return_with_citations", "expand_context_retry"
    ), f"unexpected action: {action}"


@pytest.mark.asyncio
async def test_f5_confidence_router_only_one_branch_runs():
    """Exactly one response-action branch should run."""
    wf  = build_confidence_router_workflow()
    res = await WorkflowExecutor(wf).run({"question": "irrelevant xyz987?", "tenant": "f5_empty_2"})
    action_nodes = {"return_direct", "return_cited", "expand_retry", "cannot_answer"}
    statuses = {r.node_id: r.status for r in res.results}
    ran = [n for n in action_nodes if statuses.get(n) == "success"]
    assert len(ran) == 1


# ── Complex AND / OR conditions (spec literal) ───────────────────────────────

@pytest.mark.asyncio
async def test_f5_complex_and_or_condition():
    """
    Spec literal:
        IF (search found results) AND (confidence > 70%)
        THEN return results
        ELSE expand and retry
    """
    wf = WorkflowDef(
        name="complex_and",
        nodes=[
            {"id": "in",  "type": "input"},
            {"id": "cls", "type": "classify", "config": {
                "input": "$.inputs.score",
                "rules": [
                    {"category": "return_results",
                     "when": {"and": [
                         {"left": "$.inputs.found_results", "op": "==",  "right": True},
                         {"left": "$.inputs.score",         "op": ">",   "right": 0.70},
                     ]},
                     "confidence": 0.90},
                ],
                "default": "expand_and_retry",
            }},
            {"id": "sw",  "type": "switch", "config": {
                "cases": [
                    {"label": "return",  "when": {"left": "$.cls.category", "op": "==", "right": "return_results"}},
                ],
                "default": "retry",
            }},
            {"id": "good", "type": "processing", "config": {"operation": "set", "value": "RETURN"}},
            {"id": "bad",  "type": "processing", "config": {"operation": "set", "value": "RETRY"}},
            {"id": "out",  "type": "output"},
        ],
        edges=[
            {"source": "in",  "target": "cls"},
            {"source": "cls", "target": "sw"},
            {"source": "sw",  "target": "good", "condition": {"left": "$.sw.case", "op": "==", "right": "return"}},
            {"source": "sw",  "target": "bad",  "condition": {"left": "$.sw.case", "op": "==", "right": "retry"}},
            {"source": "good","target": "out"},
            {"source": "bad", "target": "out"},
        ],
    )
    # Both conditions met → return
    r1 = await WorkflowExecutor(wf).run({"found_results": True, "score": 0.85})
    assert r1.outputs["out"] == "RETURN"

    # Score too low → retry
    r2 = await WorkflowExecutor(wf).run({"found_results": True, "score": 0.50})
    assert r2.outputs["out"] == "RETRY"

    # No results → retry
    r3 = await WorkflowExecutor(wf).run({"found_results": False, "score": 0.95})
    assert r3.outputs["out"] == "RETRY"


# ── Decision API endpoints ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f5_api_route_document():
    """POST /decisions/route-document returns parser info."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/route-document",
            json={"filename": "annual_report.pdf"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "pdf"
    assert data["parser"]   == "pypdf"
    assert data["confidence"] == 0.99
    assert "decision_path" in data


@pytest.mark.asyncio
async def test_f5_api_route_document_image():
    """OCR route for image files."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/route-document",
            json={"filename": "scan.jpg"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "image"
    assert data["parser"]   == "tesseract"


@pytest.mark.asyncio
async def test_f5_api_route_query():
    """POST /decisions/route-query classifies and runs search."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/route-query",
            json={"question": "What is deep learning?", "tenant": "f5api"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy"] in ("semantic", "hybrid", "keyword", "entity")
    assert "decision_path" in data
    assert isinstance(data["hits"], list)


@pytest.mark.asyncio
async def test_f5_api_route_quality():
    """POST /decisions/route-quality returns a confidence level + action."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/route-quality",
            json={"question": "Explain quantum computing", "tenant": "f5qual"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence_level"] in ("high", "medium", "low", "none")
    assert data["action"] is not None
    assert "decision_path" in data


@pytest.mark.asyncio
async def test_f5_api_run_named_tree():
    """POST /decisions/run executes a named decision tree."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/run",
            json={"tree": "doc_type_router", "inputs": {"filename": "data.json"}},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["outcome"] == "json"


@pytest.mark.asyncio
async def test_f5_api_run_unknown_tree_422():
    """POST /decisions/run with unknown tree name → 422."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/decisions/run",
            json={"tree": "no_such_tree", "inputs": {}},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_f5_api_get_tree_definition():
    """GET /decisions/tree/{name} returns nodes, edges, and decision_nodes."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/decisions/tree/doc_type_router",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "doc_type_router"
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0
    # Should expose classify and switch as decision nodes
    decision_types = {d["type"] for d in data["decision_nodes"]}
    assert "classify" in decision_types
    assert "switch" in decision_types


@pytest.mark.asyncio
async def test_f5_api_get_tree_not_found():
    """GET /decisions/tree/{unknown} → 404."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/decisions/tree/does_not_exist",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 404
