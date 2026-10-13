"""Structured output: JSON answers validated by Pydantic, retry, fallback, streaming."""
from __future__ import annotations

import json

import pytest

from app.rag.cost import BudgetExceededError, get_usage, reset_budgets, reset_tracker, upsert_budget
from app.rag.llm import registry as llm_registry
from app.rag.llm.base import LLMResponse
from app.rag.llm.gateway import LLMUnavailableError, call_llm
from app.rag.llm.gemini import build_payload
from app.rag.llm.structured import (
    ANSWER_SCHEMA,
    AnswerStreamExtractor,
    StructuredOutputError,
    generate_structured,
    parse_structured,
)

DOCS = [
    {"marker": "[1]", "id": "d1", "text": "Full-time employees receive 25 days of paid annual leave."},
    {"marker": "[2]", "id": "d2", "text": "All customer data is encrypted at rest with AES-256."},
]
REQUEST = {
    "messages": [{"role": "system", "content": "Answer from context."},
                 {"role": "user", "content": "How many days of annual leave do full-time employees get?"}],
    "query": "How many days of annual leave do full-time employees get?",
    "documents": DOCS,
}
VALID = json.dumps({"answer": "25 days [1].", "citations": [1], "confidence": "high", "answerable": True})


class ScriptedLLM:
    """Provider that returns queued replies and records every request it sees."""

    def __init__(self, name: str, replies: list[str], fail: bool = False) -> None:
        self.name = name
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.configs: list[dict] = []
        self.fail = fail

    async def generate(self, request, config):
        self.requests.append(request)
        self.configs.append(config)
        if self.fail:
            raise ConnectionError("provider down")
        return LLMResponse(text=self.replies.pop(0), provider=self.name, model="scripted")

    async def generate_stream(self, request, config):
        response = await self.generate(request, config)
        for i in range(0, len(response.text), 7):
            yield response.text[i:i + 7]


@pytest.fixture
def scripted():
    created: list[str] = []

    def make(replies: list[str], fail: bool = False, name: str = "scripted") -> ScriptedLLM:
        llm = ScriptedLLM(name, replies, fail)
        llm_registry.register_provider(name, lambda: llm)
        llm_registry._instances.pop(name, None)
        created.append(name)
        return llm

    yield make
    for name in created:
        llm_registry._FACTORIES.pop(name, None)
        llm_registry._instances.pop(name, None)


# ── Parsing ───────────────────────────────────────────────────────────────────

def test_valid_json_parses():
    answer = parse_structured(VALID)
    assert answer.answer == "25 days [1]."
    assert answer.citations == [1]
    assert answer.confidence == "high"
    assert answer.answerable is True


def test_fenced_json_parses():
    assert parse_structured(f"```json\n{VALID}\n```").citations == [1]


@pytest.mark.parametrize("bad", [
    "not json at all",
    '{"answer": "x", "citations": [1], "confidence": "very", "answerable": true}',
    '{"answer": "x", "citations": [1]}',
    '{"answer": "", "citations": [], "confidence": "low", "answerable": false}',
])
def test_invalid_replies_raise(bad):
    with pytest.raises(StructuredOutputError):
        parse_structured(bad)


# ── Retry and fallback ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_valid_first_reply_needs_one_attempt(scripted):
    llm = scripted([VALID])
    out = await generate_structured("scripted", REQUEST, {})
    assert out["structured"] == {"attempts": 1, "fallback": False, "errors": []}
    assert out["cited_markers"] == ["[1]"]
    # JSON mode is requested from the provider and the contract is in the system prompt.
    assert llm.configs[0]["response_schema"] == ANSWER_SCHEMA
    assert '"answerable"' in llm.requests[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_malformed_json_retries_with_validation_error(scripted):
    llm = scripted(["{answer: oops", VALID])
    out = await generate_structured("scripted", REQUEST, {})
    assert out["structured"]["attempts"] == 2
    assert out["structured"]["fallback"] is False
    assert out["answer"] == "25 days [1]."
    # The retry shows the model its bad reply and the validation error.
    retry_messages = llm.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": "{answer: oops"}
    assert "invalid JSON" in retry_messages[-1]["content"]


@pytest.mark.asyncio
async def test_malformed_twice_falls_back_to_extractive(scripted):
    scripted(["garbage", '{"answer": "x", "confidence": "maybe"}'])
    out = await generate_structured("scripted", REQUEST, {})
    assert out["structured"]["attempts"] == 2
    assert out["structured"]["fallback"] is True
    assert len(out["structured"]["errors"]) == 2
    assert "25 days" in out["answer"]           # extractive answer from DOCS[0]
    assert out["cited_markers"] == ["[1]"]
    assert out["confidence"] in ("high", "medium", "low")


@pytest.mark.asyncio
async def test_hallucinated_citations_are_dropped(scripted):
    reply = json.dumps({"answer": "a", "citations": [1, 7, 1, 0], "confidence": "low", "answerable": True})
    scripted([reply])
    out = await generate_structured("scripted", REQUEST, {})
    assert out["citations"] == [1]


@pytest.mark.asyncio
async def test_unconfigured_provider_degrades_to_extractive():
    out = await generate_structured("gemini", REQUEST, {})   # no GEMINI_API_KEY in tests
    assert out["structured"]["fallback"] is True
    assert "not configured" in out["structured"]["errors"][0]
    assert "25 days" in out["answer"]


@pytest.mark.asyncio
async def test_stub_reports_unanswerable_as_idk():
    request = {**REQUEST, "query": "What is the parking fee at the Berlin office?"}
    out = await generate_structured("stub", request, {})
    assert out["answerable"] is False
    assert out["citations"] == []
    assert "don't know" in out["answer"]


# ── Gateway guards ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gateway_opens_breaker_after_failures(scripted):
    scripted([], fail=True)
    for _ in range(5):
        with pytest.raises(ConnectionError):
            await call_llm("scripted", REQUEST, {})
    with pytest.raises(LLMUnavailableError, match="circuit"):
        await call_llm("scripted", REQUEST, {})
    from app.engine.circuit_breaker import registry
    registry.reset("llm:scripted")


@pytest.mark.asyncio
async def test_gateway_enforces_tenant_budget_and_logs_cost(scripted):
    reset_budgets()
    reset_tracker()
    scripted([VALID], name="gpt-4o-scripted")
    await call_llm("gpt-4o-scripted", REQUEST, {"model": "gpt-4o"}, tenant="acme-budget")
    events = get_usage(tenant="acme-budget")
    assert events and events[0].operation == "llm:generate"
    assert events[0].cost > 0

    upsert_budget("acme-budget", 0.0000001)
    with pytest.raises(BudgetExceededError):
        await call_llm("gpt-4o-scripted", REQUEST, {"model": "gpt-4o"}, tenant="acme-budget")
    reset_budgets()


# ── Gemini JSON mode ──────────────────────────────────────────────────────────

def test_gemini_payload_uses_json_schema_mode_and_multi_turn():
    request = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "bad"},
        {"role": "user", "content": "fix it"},
    ]}
    payload = build_payload(request, {"response_schema": ANSWER_SCHEMA})
    gen = payload["generationConfig"]
    assert gen["responseMimeType"] == "application/json"
    assert gen["responseSchema"]["propertyOrdering"][0] == "answer"
    assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]
    assert payload["system_instruction"]["parts"][0]["text"] == "sys"


def test_gemini_payload_plain_mode_unchanged():
    payload = build_payload({"messages": [{"role": "user", "content": "q"}]}, {})
    assert "responseMimeType" not in payload["generationConfig"]


# ── Streaming ─────────────────────────────────────────────────────────────────

def test_stream_extractor_decodes_answer_incrementally():
    text = json.dumps({"answer": 'Say "hi"\nthen café [1]', "citations": [1],
                       "confidence": "low", "answerable": True}, ensure_ascii=True)
    ex = AnswerStreamExtractor()
    pieces = [ex.feed(ch) for ch in text]          # worst case: one char at a time
    assert "".join(pieces) == 'Say "hi"\nthen café [1]'
    assert ex.done


@pytest.mark.asyncio
async def test_sse_pipeline_streams_answer_then_sources_event():
    from app.api.stream import _rag_pipeline_events

    events = [ev async for ev in _rag_pipeline_events("What is AI?", "so-stream", "stub")]
    kinds = [e["event"] for e in events]
    assert "sources" in kinds
    assert kinds.index("sources") < kinds.index("complete")
    sources = next(e for e in events if e["event"] == "sources")
    assert sources["confidence"] in ("high", "medium", "low")
    assert isinstance(sources["answerable"], bool)
    tokens = "".join(e["token"] for e in events if e["event"] == "token")
    if not sources["replaced"]:
        assert tokens == sources["answer"]


@pytest.mark.asyncio
async def test_sse_invalid_stream_is_repaired_in_sources_event(scripted):
    from app.api.pipelines import build_index_workflow
    from app.api.stream import _rag_pipeline_events
    from app.engine.executor import WorkflowExecutor

    await WorkflowExecutor(build_index_workflow()).run({
        "tenant": "so-repair", "filename": "hr.md",
        "text": "# Leave\nFull-time employees receive 25 days of paid annual leave.",
    })
    scripted(['{"answer": "25 days [1]", "citations": "oops"}', VALID])
    events = [ev async for ev in _rag_pipeline_events(
        "How many days of annual leave do full-time employees get?", "so-repair", "scripted")]
    sources = next(e for e in events if e["event"] == "sources")
    assert sources["structured"]["attempts"] == 2
    assert sources["replaced"] is True
    assert sources["answer"] == "25 days [1]."
    assert sources["cited_markers"] == ["[1]"]


def test_rag_ask_returns_structured_fields():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        client.post("/api/v1/rag/index", json={
            "tenant": "so-api", "filename": "sec.md",
            "text": "# Encryption\nAll customer data is encrypted at rest with AES-256.",
        })
        out = client.post("/api/v1/rag/ask", json={
            "tenant": "so-api", "question": "How is customer data encrypted at rest?",
        }).json()
    assert out["answerable"] is True
    assert out["confidence"] in ("high", "medium")
    assert out["cited_markers"] == ["[1]"]
    assert out["structured"]["fallback"] is False
