"""Tests for F3: Chaining & Sequential Execution."""
from __future__ import annotations

import pytest

from app.engine.chain import ChainDef, ChainExecutor, ChainStep
from app.engine.nodes.decompose import _decompose
from app.engine.nodes.synthesize import SynthesizeNode
from app.engine.context import ExecutionContext
from app.models.workflow import NodeDef, NodeType, WorkflowDef
from app.api.pipelines import (
    build_index_workflow,
    build_ask_workflow,
    build_decompose_workflow,
    build_synthesize_workflow,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _simple_wf(name: str, output_value) -> WorkflowDef:
    """Minimal workflow: uses 'set' operation to return a constant value."""
    return WorkflowDef(
        name=name,
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "proc", "type": "processing",
             "config": {"operation": "set", "value": output_value}},
            {"id": "out", "type": "output",
             "config": {"value": "$.proc"}},
        ],
        edges=[
            {"source": "in", "target": "proc"},
            {"source": "proc", "target": "out"},
        ],
    )


def _failing_wf(name: str) -> WorkflowDef:
    """Workflow that always errors (external node pointing to bad URL)."""
    return WorkflowDef(
        name=name,
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "bad", "type": "external",
             "config": {"url": "http://127.0.0.1:1/notexist", "method": "GET"}},
            {"id": "out", "type": "output", "config": {"value": "$.bad"}},
        ],
        edges=[
            {"source": "in", "target": "bad"},
            {"source": "bad", "target": "out"},
        ],
    )


# ── Document Upload Chain ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_document_chain_succeeds():
    """Index workflow completes and returns upserted count."""
    chain = ChainDef(
        name="doc_upload",
        steps=[ChainStep(name="index", workflow=build_index_workflow())],
    )
    result = await ChainExecutor(chain).run(
        {"tenant": "test", "text": "Hello world from F3.", "filename": "test.txt"}
    )
    assert result.status == "success"
    assert len(result.steps) == 1
    assert result.steps[0].step == "index"
    assert result.steps[0].status == "success"
    assert result.final_output.get("upserted", 0) >= 1


# ── Query Processing Chain ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_ask_chain_returns_answer():
    """Full RAG ask chain produces an answer field."""
    chain = ChainDef(
        name="rag_ask",
        steps=[ChainStep(name="ask", workflow=build_ask_workflow("stub"))],
    )
    result = await ChainExecutor(chain).run(
        {"tenant": "test", "question": "What is machine learning?"}
    )
    assert result.status in ("success", "partial")
    assert "answer" in result.final_output


# ── Sequential output forwarding ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_output_forwarded_between_steps():
    """Each step can read output produced by the previous step."""
    wf1 = _simple_wf("step1", {"greeting": "hello"})
    wf2 = WorkflowDef(
        name="step2",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "out", "type": "output",
             "config": {"value": "$.inputs.greeting"}},
        ],
        edges=[{"source": "in", "target": "out"}],
    )
    chain = ChainDef(
        name="forward_test",
        steps=[
            ChainStep(name="s1", workflow=wf1),
            ChainStep(name="s2", workflow=wf2),
        ],
    )
    result = await ChainExecutor(chain).run({})
    assert result.status == "success"
    assert result.final_output.get("greeting") == "hello"


# ── Error handling: fail_fast ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_fail_fast_stops_on_first_error():
    """With on_error=fail_fast, chain stops after first failing step."""
    chain = ChainDef(
        name="fail_fast_test",
        on_error="fail_fast",
        steps=[
            ChainStep(name="bad", workflow=_failing_wf("bad")),
            ChainStep(name="good", workflow=_simple_wf("good", "ok")),
        ],
    )
    result = await ChainExecutor(chain).run({})
    assert result.status == "error"
    # Second step must NOT have run
    assert len(result.steps) == 1
    assert result.steps[0].step == "bad"


# ── Error handling: continue-on-error ────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_continue_on_error_keeps_going():
    """With on_error=continue, chain runs all steps even when one fails."""
    chain = ChainDef(
        name="continue_test",
        on_error="continue",
        steps=[
            ChainStep(name="bad", workflow=_failing_wf("bad")),
            ChainStep(name="good", workflow=_simple_wf("good", "ok")),
        ],
    )
    result = await ChainExecutor(chain).run({})
    assert result.status == "partial"
    assert len(result.steps) == 2
    statuses = {s.step: s.status for s in result.steps}
    assert statuses["bad"] == "error"
    assert statuses["good"] == "success"


# ── Decompose node ────────────────────────────────────────────────────────────

def test_f3_decompose_temporal_comparison():
    """Temporal comparison yields per-year + difference sub-questions."""
    subs = _decompose("How did the budget change from 2022 to 2023?", 5)
    assert len(subs) >= 2
    assert any("2022" in q for q in subs)
    assert any("2023" in q for q in subs)
    assert any("difference" in q.lower() for q in subs)


def test_f3_decompose_multi_part():
    """Multi-part question split on '?' boundaries."""
    subs = _decompose("What is AI? What is ML?", 5)
    assert len(subs) == 2


def test_f3_decompose_and_conjunction():
    """'and' conjunction splits into two sub-questions."""
    subs = _decompose("What are the revenue and expenses?", 5)
    assert len(subs) == 2


def test_f3_decompose_single_question_unchanged():
    """Single simple question is returned as-is."""
    subs = _decompose("What is machine learning?", 5)
    assert len(subs) == 1
    assert "machine learning" in subs[0].lower()


# ── Synthesize node ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_synthesize_merges_answers():
    """Synthesize node deduplicates and joins multiple answers."""
    node = SynthesizeNode(NodeDef(id="synth", type=NodeType.SYNTHESIZE))
    ctx = ExecutionContext(run_id="test", inputs={
        "sub_answers": [
            {"question": "q1", "answer": "The budget in 2022 was $1M."},
            {"question": "q2", "answer": "The budget in 2023 was $1.5M."},
            {"question": "q3", "answer": "The difference is $0.5M."},
        ],
        "question": "How did the budget change from 2022 to 2023?",
    })
    node.config = {
        "answers": "$.inputs.sub_answers",
        "original_question": "$.inputs.question",
    }
    output = await node.run(ctx, {})
    assert output["synthesized"] is True
    assert output["source_count"] == 3
    assert "1M" in output["answer"] or "budget" in output["answer"].lower()


@pytest.mark.asyncio
async def test_f3_synthesize_empty_answers():
    """Synthesize node handles empty sub-answers gracefully."""
    node = SynthesizeNode(NodeDef(id="synth", type=NodeType.SYNTHESIZE))
    ctx = ExecutionContext(run_id="test", inputs={"sub_answers": [], "question": "Q?"})
    node.config = {
        "answers": "$.inputs.sub_answers",
        "original_question": "$.inputs.question",
    }
    output = await node.run(ctx, {})
    assert output["source_count"] == 0
    assert "could not find" in output["answer"].lower()


# ── Decompose + Synthesize workflows ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_f3_decompose_workflow():
    """build_decompose_workflow returns sub_questions list."""
    from app.engine.executor import WorkflowExecutor
    wf = build_decompose_workflow()
    result = await WorkflowExecutor(wf).run(
        {"question": "What was the revenue in 2021 and 2022?"}
    )
    assert result.status == "success"
    out = result.outputs.get("out", {})
    assert "sub_questions" in out
    assert len(out["sub_questions"]) >= 1


@pytest.mark.asyncio
async def test_f3_synthesize_workflow():
    """build_synthesize_workflow merges sub-answers."""
    from app.engine.executor import WorkflowExecutor
    wf = build_synthesize_workflow()
    result = await WorkflowExecutor(wf).run({
        "sub_answers": [
            {"question": "q1", "answer": "Revenue was $1M in 2021."},
            {"question": "q2", "answer": "Revenue was $2M in 2022."},
        ],
        "question": "What was revenue in 2021 and 2022?",
    })
    assert result.status == "success"
    out = result.outputs.get("out", {})
    assert out.get("synthesized") is True
