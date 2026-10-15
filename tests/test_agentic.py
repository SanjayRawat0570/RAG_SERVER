"""Agentic retrieval: tools, tenant isolation, hard limits, multi-hop answers."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.pipelines import build_agentic_workflow, build_index_workflow
from app.engine.executor import WorkflowExecutor
from app.rag.agent.tools import TOOL_SPECS, ToolContext, run_tool
from app.rag.graph import reset_graph
from app.rag.llm import registry as llm_registry
from app.rag.llm.base import LLMResponse
from app.rag.llm.gemini import build_payload

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"
AUTH = {"Authorization": "Bearer dev"}


async def _index(tenant: str, filename: str, text: str) -> None:
    result = await WorkflowExecutor(build_index_workflow()).run(
        {"tenant": tenant, "filename": filename, "text": text})
    assert result.status == "success"


async def _index_corpus(tenant: str, *names: str) -> None:
    for name in names:
        await _index(tenant, name, (CORPUS / name).read_text(encoding="utf-8"))


async def _agentic(tenant: str, question: str, provider: str = "stub", **limits):
    result = await WorkflowExecutor(build_agentic_workflow(provider, **limits)).run(
        {"tenant": tenant, "question": question, "few_shot": []})
    assert result.status == "success", [r.error for r in result.results if r.error]
    return result.outputs["out"]


class ToolBot:
    """A tool-capable fake model: replays scripted tool calls, records requests."""

    supports_tools = True

    def __init__(self, name, calls_per_turn, usage=None, fail=False):
        self.name = name
        self.calls_per_turn = calls_per_turn   # callable(turn) -> list of {"name","args"}
        self.usage = usage or {"input_tokens": 50, "output_tokens": 10}
        self.fail = fail
        self.turns = 0

    async def generate(self, request, config):
        if self.fail:
            raise ConnectionError("model unavailable")
        if "tools" not in config:  # final answer step (structured, no tools)
            return LLMResponse(text='{"answer": "done", "citations": [], "confidence": "low", '
                                    '"answerable": false}', provider=self.name, model="toolbot")
        self.turns += 1
        calls = self.calls_per_turn(self.turns)
        return LLMResponse(text="" if calls else "DONE", provider=self.name, model="toolbot",
                           usage=dict(self.usage), tool_calls=calls,
                           parts=[{"functionCall": c} for c in calls])


@pytest.fixture
def toolbot():
    made: list[str] = []

    def make(name="toolbot", **kw) -> ToolBot:
        bot = ToolBot(name, **kw)
        llm_registry.register_provider(name, lambda: bot)
        llm_registry._instances.pop(name, None)
        made.append(name)
        return bot

    yield make
    for name in made:
        llm_registry._FACTORIES.pop(name, None)
        llm_registry._instances.pop(name, None)


# ── Tools & tenant isolation ──────────────────────────────────────────────────

def test_tool_specs_never_expose_a_tenant_parameter():
    names = {s["name"] for s in TOOL_SPECS}
    assert names == {"search_docs", "get_document", "lookup_entity"}
    for spec in TOOL_SPECS:
        params = set(spec["parameters"]["properties"])
        assert not params & {"tenant", "tenant_id", "namespace"}, spec["name"]


@pytest.mark.asyncio
async def test_tools_cannot_reach_another_tenants_data():
    reset_graph()
    await _index("iso-a", "secret.md", "# Nightjar\nProject Nightjar has a budget of 9 million. "
                                       "Project Nightjar is owned by the Falcon Unit.")
    await _index("iso-b", "public.md", "# Roadmap\nThe public roadmap is published every quarter.")

    as_b = ToolContext(tenant="iso-b")
    # The model tries to smuggle tenant A in through the arguments.
    res = await run_tool("search_docs", {"query": "Nightjar budget", "tenant": "iso-a",
                                         "namespace": "iso-a"}, as_b)
    assert res["dropped_args"] == ["namespace", "tenant"]
    assert res["error"] is None
    assert all("Nightjar" not in h["metadata"]["text"] for h in res["hits"])
    assert all(h["metadata"]["source"] == "public.md" for h in res["hits"])

    doc = await run_tool("get_document", {"doc_id": "secret.md"}, as_b)
    assert doc["summary"]["found"] is False and doc["hits"] == []

    ent_b = await run_tool("lookup_entity", {"name": "Project Nightjar"}, as_b)
    assert ent_b["summary"]["entities"] == []
    ent_a = await run_tool("lookup_entity", {"name": "Project Nightjar"}, ToolContext(tenant="iso-a"))
    assert ent_a["summary"]["entities"], "tenant A must still see its own graph"


@pytest.mark.asyncio
async def test_hostile_model_cannot_pull_other_tenant_into_evidence(toolbot):
    await _index("host-a", "secret.md", "# Nightjar\nProject Nightjar has a budget of 9 million.")
    await _index("host-b", "public.md", "# Roadmap\nThe public roadmap is published every quarter.")
    toolbot(calls_per_turn=lambda t: [
        {"name": "search_docs", "args": {"query": "Nightjar budget", "tenant": "host-a"}},
        {"name": "get_document", "args": {"doc_id": "secret.md", "tenant_id": "host-a"}},
    ] if t == 1 else [])

    out = await _agentic("host-b", "What is the Nightjar budget?", provider="toolbot")
    calls = [s for s in out["steps"] if s["type"] == "tool_call"]
    assert {tuple(s["dropped_args"]) for s in calls} == {("tenant",), ("tenant_id",)}
    assert all(c.get("source") != "secret.md" for c in out["citations"] or [])
    assert "9 million" not in out["answer"]


@pytest.mark.asyncio
async def test_get_document_returns_whole_document_in_order():
    await _index_corpus("doc-t", "hr_leave_policy.md")
    res = await run_tool("get_document", {"doc_id": "hr_leave_policy.md"}, ToolContext(tenant="doc-t"))
    assert res["summary"]["found"] and res["summary"]["chunks"] >= 3
    text = res["summary"]["text"]
    assert text.index("Annual Leave") < text.index("Sick Leave") < text.index("Parental Leave")


# ── Hard limits ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_step_limit_is_enforced(toolbot):
    await _index_corpus("lim-t", "teams.md")
    bot = toolbot(calls_per_turn=lambda t: [{"name": "search_docs", "args": {"query": f"team q{t}"}}])
    # The corpus cannot answer this, so only the round limit can stop the agent.
    out = await _agentic("lim-t", "What is the Mobile team's travel budget?", provider="toolbot",
                         max_rounds=3)
    rounds = {s["round"] for s in out["steps"] if s["type"] == "tool_call"}
    assert out["agent"]["rounds"] == 3 and len(rounds) == 3
    assert out["agent"]["stop_reason"] == "max_rounds"
    assert bot.turns == 3                       # never asked to plan a 4th round


@pytest.mark.asyncio
async def test_token_budget_is_enforced(toolbot):
    await _index_corpus("tok-t", "teams.md")
    toolbot(calls_per_turn=lambda t: [{"name": "search_docs", "args": {"query": f"q{t}"}}],
            usage={"input_tokens": 5000, "output_tokens": 2000})
    out = await _agentic("tok-t", "What is the Mobile team's travel budget?", provider="toolbot",
                         token_budget=6000)
    assert out["agent"]["stop_reason"] == "token_budget"
    assert out["agent"]["rounds"] == 1


@pytest.mark.asyncio
async def test_timeout_is_enforced():
    out = await _agentic("to-t", "Who leads the Mobile team?", timeout_s=1e-9)
    assert out["agent"]["stop_reason"] == "timeout"
    assert out["agent"]["rounds"] == 0


@pytest.mark.asyncio
async def test_planner_falls_back_offline_when_model_fails(toolbot):
    await _index_corpus("fb-t", "teams.md")
    toolbot(calls_per_turn=lambda t: [], fail=True)
    out = await _agentic("fb-t", "Who leads the Mobile team?", provider="toolbot")
    assert any(s["type"] == "planner_fallback" for s in out["steps"])
    assert "Tom Becker" in out["answer"]


# ── Multi-hop ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected,bridge", [
    ("Who leads the team that owns Project Atlas?", "Priya Shah", "Data Platform"),
    ("Who leads the team responsible for Project Cedar?", "Amara Okafor", "Payments"),
])
async def test_multi_hop_question_answered_in_agentic_mode(question, expected, bridge):
    tenant = f"mh-{bridge.lower().replace(' ', '-')}"
    await _index_corpus(tenant, "projects.md", "teams.md")
    out = await _agentic(tenant, question)

    assert expected in out["answer"]
    kinds = [s["type"] for s in out["steps"]]
    assert kinds[0] == "plan" and kinds[-1] == "answer"
    assert "lookup_entity" in {s.get("tool") for s in out["steps"]}
    rewrite = next(s for s in out["steps"] if s["type"] == "rewrite")
    assert bridge in rewrite["query"]
    assert out["rewritten_query"] == rewrite["query"]
    assert out["agent"]["rounds"] == 2
    assert set(out["cited_sources"]) == {"teams.md"}


@pytest.mark.asyncio
async def test_fast_mode_cannot_answer_the_same_multi_hop_question():
    from app.api.pipelines import build_ask_workflow

    await _index_corpus("mh-fast", "projects.md", "teams.md")
    result = await WorkflowExecutor(build_ask_workflow("stub")).run(
        {"tenant": "mh-fast", "question": "Who leads the team that owns Project Atlas?", "few_shot": []})
    assert "Priya Shah" not in result.outputs["out"]["answer"]


# ── API ───────────────────────────────────────────────────────────────────────

def test_mode_is_selectable_per_request_and_fast_is_default():
    from app.main import app

    text = (CORPUS / "projects.md").read_text(encoding="utf-8") + "\n\n" + \
        (CORPUS / "teams.md").read_text(encoding="utf-8")
    with TestClient(app) as client:
        auth = {"Authorization": "Bearer dev"}
        client.post("/api/v1/rag/index", headers=auth,
                    json={"tenant": "mode-t", "text": text, "filename": "org.md"})
        q = "Who leads the team that owns Project Beacon?"
        fast = client.post("/api/v1/rag/ask", headers=auth, json={"tenant": "mode-t", "question": q}).json()
        agentic = client.post("/api/v1/rag/ask", headers=auth, json={"tenant": "mode-t", "question": q,
                                                                     "mode": "agentic"}).json()
        bad = client.post("/api/v1/rag/ask", headers=auth,
                          json={"tenant": "mode-t", "question": q, "mode": "x"})

    assert fast["mode"] == "fast" and not fast.get("steps")
    assert agentic["mode"] == "agentic" and agentic["steps"]
    assert "Tom Becker" in agentic["answer"]
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_sse_agentic_mode_emits_agent_steps():
    from app.api.stream import _rag_pipeline_events

    await _index_corpus("sse-ag", "projects.md", "teams.md")
    events = [e async for e in _rag_pipeline_events(
        "Who leads the team that owns Project Atlas?", "sse-ag", "stub", mode="agentic")]
    kinds = [e["event"] for e in events]
    assert "agent_steps" in kinds and "sources" in kinds and kinds[-1] == "complete"
    steps = next(e for e in events if e["event"] == "agent_steps")
    assert any(s["type"] == "rewrite" for s in steps["steps"])
    complete = events[-1]
    assert "Priya Shah" in complete["answer"]


def test_gemini_payload_declares_tools():
    payload = build_payload({"messages": [{"role": "user", "content": "q"}]}, {"tools": TOOL_SPECS})
    decls = payload["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in decls] == ["search_docs", "get_document", "lookup_entity"]
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"


@pytest.mark.asyncio
async def test_lookup_entity_matches_names_with_extra_words():
    reset_graph("ent-t")
    await _index_corpus("ent-t", "projects.md")
    res = await run_tool("lookup_entity", {"name": "Data Platform team"}, ToolContext(tenant="ent-t"))
    names = [e["name"] for e in res["summary"]["entities"]]
    assert "Data Platform" in names
    assert any("Project Atlas" in f for f in res["facts"])
