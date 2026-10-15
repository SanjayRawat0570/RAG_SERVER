"""Conversation memory: follow-up rewriting, short/long-term memory, clear, GDPR."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.rag.llm import registry as llm_registry
from app.rag.llm.base import LLMResponse
from app.rag.memory import store
from app.rag.memory.rewriter import heuristic_rewrite, is_follow_up, rewrite_follow_up
from app.rag.memory.store import Turn
from app.rag.personalization.store import infer_interests, reset_store
from app.rag.security.gdpr import delete_user_data, export_user_data

CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"
AUTH = {"Authorization": "Bearer dev"}           # the dev user ("dev")


@pytest.fixture(autouse=True)
def _clean():
    store.reset()
    reset_store()
    yield
    store.reset()
    reset_store()


def _turn(q: str) -> list[Turn]:
    return [Turn(question=q, standalone_question=q, answer="...")]


# ── Rewriting ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("previous,follow_up,expected", [
    ("What is the leave policy for full-time staff?", "what about part-time staff?",
     "What is the leave policy for part-time staff?"),
    ("How many days of annual leave do full-time employees get?", "And what about part-time employees?",
     "How many days of annual leave do part-time employees get?"),
    ("Which team owns Project Atlas?", "what about Project Cedar?", "Which team owns Project Cedar?"),
    ("Who leads the Mobile team?", "and the Payments team?", "Who leads the Payments team?"),
])
def test_follow_up_is_rewritten_into_standalone_question(previous, follow_up, expected):
    turns = _turn(previous)
    assert is_follow_up(follow_up, turns)
    assert heuristic_rewrite(follow_up, turns) == expected


def test_pronoun_follow_up_carries_previous_question_as_context():
    rewritten = heuristic_rewrite("How often are they rotated?", _turn("How are encryption keys stored?"))
    assert rewritten.startswith("How often are they rotated?")
    assert "How are encryption keys stored?" in rewritten


@pytest.mark.parametrize("question", [
    "What is the minimum password length?",
    "Who leads the Data Platform team?",
])
def test_standalone_questions_are_not_rewritten(question):
    assert not is_follow_up(question, _turn("How is data encrypted?"))
    assert not is_follow_up("what about part-time staff?", [])     # no history → nothing to resolve


@pytest.mark.asyncio
async def test_llm_rewrite_used_when_a_model_is_available():
    class Rewriter:
        name = "rewriter"
        seen: list = []

        async def generate(self, request, config):
            Rewriter.seen.append(request)
            return LLMResponse(text="What is the leave policy for part-time staff?",
                               provider="rewriter", model="r")

    llm_registry.register_provider("rewriter", Rewriter)
    llm_registry._instances.pop("rewriter", None)
    try:
        out = await rewrite_follow_up("what about part-time staff?",
                                      _turn("What is the leave policy for full-time staff?"),
                                      provider="rewriter")
    finally:
        llm_registry._FACTORIES.pop("rewriter", None)
        llm_registry._instances.pop("rewriter", None)
    assert out == {"question": "What is the leave policy for part-time staff?",
                   "rewritten": True, "method": "llm"}
    assert "full-time staff" in Rewriter.seen[0]["messages"][1]["content"]


# ── Short-term store ──────────────────────────────────────────────────────────

def test_only_last_five_turns_are_kept_per_conversation():
    conv = store.create("t", "u")
    for i in range(8):
        store.add_turn(conv, Turn(question=f"q{i}", standalone_question=f"q{i}", answer="a"))
    assert [t.question for t in store.history(conv)] == ["q3", "q4", "q5", "q6", "q7"]
    assert conv.title == "q0"


def test_conversations_are_scoped_to_tenant_and_user():
    conv = store.create("tenant-a", "alice")
    assert store.get("tenant-a", "alice", conv.id) is conv
    assert store.get("tenant-b", "alice", conv.id) is None
    assert store.get("tenant-a", "bob", conv.id) is None
    # Someone else's id never resumes their chat — a new one is started instead.
    assert store.get_or_create("tenant-a", "bob", conv.id).id != conv.id


# ── API end-to-end ────────────────────────────────────────────────────────────

def _index(client: TestClient, tenant: str) -> None:
    for name in ("hr_leave_policy.md", "teams.md"):
        client.post("/api/v1/rag/index", headers=AUTH, json={
            "tenant": tenant, "filename": name, "text": (CORPUS / name).read_text(encoding="utf-8")})


def test_follow_up_through_api_shows_rewritten_query_in_trace():
    from app.main import app

    with TestClient(app) as client:
        _index(client, "mem-api")
        first = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-api", "question": "How is annual leave calculated for full-time employees?",
            "new_conversation": True}).json()
        conv_id = first["conversation_id"]
        assert conv_id and first["standalone_question"] == first["original_question"]

        follow = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-api", "question": "what about part-time staff?",
            "conversation_id": conv_id}).json()

    assert follow["conversation_id"] == conv_id
    assert follow["standalone_question"] == "How is annual leave calculated for part-time staff?"
    step = follow["steps"][0]
    assert step["type"] == "memory_rewrite" and step["query"] == follow["standalone_question"]
    assert "pro rata" in follow["answer"]


def test_agentic_mode_keeps_the_memory_step_before_agent_steps():
    from app.main import app

    with TestClient(app) as client:
        _index(client, "mem-ag")
        first = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-ag", "question": "Who leads the Mobile team?", "new_conversation": True}).json()
        follow = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-ag", "question": "and the Payments team?", "mode": "agentic",
            "conversation_id": first["conversation_id"]}).json()
    assert follow["steps"][0]["type"] == "memory_rewrite"
    assert any(s["type"] == "tool_call" for s in follow["steps"])
    assert "Amara Okafor" in follow["answer"]


def test_callers_without_credentials_are_refused():
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/rag/ask", json={"tenant": "mem-anon", "question": "What is AI?"})
    assert r.status_code == 401


def test_anonymous_demo_callers_get_no_memory(monkeypatch):
    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "demo_mode", True)
    with TestClient(app) as client:
        out = client.post("/api/v1/rag/ask", json={"tenant": "mem-anon", "question": "What is AI?"}).json()
    assert out["conversation_id"] is None


def test_sidebar_endpoints_list_show_and_clear():
    from app.main import app

    with TestClient(app) as client:
        _index(client, "mem-ui")
        new = client.post("/api/v1/conversations?tenant=mem-ui", headers=AUTH).json()
        client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-ui", "question": "Who leads the Mobile team?", "conversation_id": new["id"]})
        listed = client.get("/api/v1/conversations?tenant=mem-ui", headers=AUTH).json()
        assert listed["conversations"][0]["id"] == new["id"]
        assert listed["conversations"][0]["title"] == "Who leads the Mobile team?"

        chat = client.get(f"/api/v1/conversations/{new['id']}?tenant=mem-ui", headers=AUTH).json()
        assert chat["messages"][0]["question"] == "Who leads the Mobile team?"
        assert "Tom Becker" in chat["messages"][0]["answer"]

        assert client.get(f"/api/v1/conversations/{new['id']}?tenant=other", headers=AUTH).status_code == 404
        assert client.delete(f"/api/v1/conversations/{new['id']}?tenant=mem-ui", headers=AUTH).status_code == 200
        assert client.get(f"/api/v1/conversations/{new['id']}?tenant=mem-ui", headers=AUTH).status_code == 404


def test_sse_stream_uses_memory():
    import json as _json
    from app.main import app

    with TestClient(app) as client:
        _index(client, "mem-sse")
        first = client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-sse", "question": "What is the leave policy for full-time staff?",
            "new_conversation": True}).json()
        with client.stream("POST", "/api/v1/rag/stream/ask", headers=AUTH, json={
                "tenant": "mem-sse", "question": "what about part-time staff?",
                "conversation_id": first["conversation_id"]}) as resp:
            events = [_json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]
    memory_ev = events[0]
    assert memory_ev["event"] == "memory"
    assert memory_ev["standalone_question"] == "What is the leave policy for part-time staff?"
    complete = events[-1]
    assert complete["event"] == "complete" and complete["conversation_id"] == first["conversation_id"]
    assert len(store.history(store.get("mem-sse", "dev", first["conversation_id"]))) == 2


# ── Long-term memory → personalization ────────────────────────────────────────

def test_cited_topics_become_interests_and_boost_ranking():
    from app.main import app
    from app.rag.memory.service import effective_profile

    with TestClient(app) as client:
        _index(client, "mem-lt")
        for q in ("How many days of paid sick leave do employees receive?",
                  "How long is parental leave for primary caregivers?"):
            client.post("/api/v1/rag/ask", headers=AUTH,
                        json={"tenant": "mem-lt", "question": q, "new_conversation": True})

    assert infer_interests("dev")[0]["topic"] == "HR"   # leave policy text → HR topic
    assert "HR" in effective_profile("dev").content.interests


@pytest.mark.asyncio
async def test_personalize_node_reranks_by_long_term_interests():
    from app.engine.context import ExecutionContext
    from app.engine.nodes import get_node
    from app.models.workflow import NodeDef, NodeType
    from app.rag.personalization.profile import QueryRecord
    from app.rag.personalization.store import record_query

    hits = [
        {"id": "sec", "score": 0.50, "metadata": {"text": "Encryption keys are stored in the cloud."}},
        {"id": "hr", "score": 0.45, "metadata": {"text": "Every employee receives paid benefits."}},
    ]
    node = get_node(NodeDef(id="p", type=NodeType.PERSONALIZE, config={"user_id": "$.inputs.user_id"}))
    unchanged = await node.run(ExecutionContext("r", {"user_id": "nobody"}), {"rerank": hits})
    assert [h["id"] for h in unchanged] == ["sec", "hr"]

    record_query("lt-user", QueryRecord(query="q", topics=["HR"]))
    boosted = await node.run(ExecutionContext("r", {"user_id": "lt-user"}), {"rerank": hits})
    assert [h["id"] for h in boosted] == ["hr", "sec"]


# ── GDPR ──────────────────────────────────────────────────────────────────────

def test_gdpr_export_includes_memory_and_erase_removes_it():
    conv = store.create("acme", "erin")
    store.add_turn(conv, Turn(question="q", standalone_question="q", answer="a", cited_sources=["hr.md"]))
    other = store.create("acme", "frank")

    from app.rag.personalization.profile import QueryRecord
    from app.rag.personalization.store import record_query
    record_query("erin", QueryRecord(query="q", topics=["HR"]))

    export = export_user_data("erin")
    assert export["conversations"][0]["messages"][0]["question"] == "q"
    assert export["inferred_interests"][0]["topic"] == "HR"

    result = delete_user_data("erin")
    assert result["deleted"]["conversations"] == 1
    assert store.list_conversations("acme", "erin") == []
    assert infer_interests("erin") == []
    assert store.get("acme", "frank", other.id) is other          # other users untouched
    assert export_user_data("erin")["conversations"] == []


def test_gdpr_erase_endpoint_clears_memory():
    from app.main import app

    with TestClient(app) as client:
        _index(client, "mem-gdpr")
        client.post("/api/v1/rag/ask", headers=AUTH, json={
            "tenant": "mem-gdpr", "question": "Who leads the Mobile team?", "new_conversation": True})
        assert client.get("/api/v1/conversations?tenant=mem-gdpr", headers=AUTH).json()["total"] == 1
        client.delete("/api/v1/security/gdpr/data", headers=AUTH)
        assert client.get("/api/v1/conversations?tenant=mem-gdpr", headers=AUTH).json()["total"] == 0
