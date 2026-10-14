"""MCP server (app/mcp_server.py) + tenant-scoped API keys and RBAC.

The MCP server talks to the real FastAPI app through an in-process ASGI
transport, and the MCP client talks to the server through the SDK's in-memory
session — so these tests exercise the full path a Claude Desktop call takes.
"""
from __future__ import annotations

import json
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.shared.memory import create_connected_server_and_client_session

from app.main import app
from app.mcp_server import RAGClient, build_server
from app.rag.security import apikeys
from app.rag.security.apikeys import create_api_key, reset_api_keys, revoke_api_key
from app.rag.tenants import add_member, create_organization, delete_organization

DOC_A = ("# Project Falcon\n\nProject Falcon has a budget of 7 million euros. "
         "The Falcon steering committee meets every Tuesday.")
DOC_B = ("# Project Falcon\n\nProject Falcon has a budget of 99 million dollars, codeword "
         "BLUEHERON. The Falcon steering committee meets every Friday.")


@pytest.fixture(autouse=True)
def _clean():
    reset_api_keys()
    yield
    reset_api_keys()


@pytest.fixture
def tenants():
    """Two tenants holding a near-identical document with different facts."""
    a, b = f"mcp-a-{uuid.uuid4().hex[:6]}", f"mcp-b-{uuid.uuid4().hex[:6]}"
    with TestClient(app) as client:
        for tenant, text in ((a, DOC_A), (b, DOC_B)):
            r = client.post("/api/v1/rag/index", json={"tenant": tenant, "text": text,
                                                       "filename": f"{tenant}-falcon.md"})
            assert r.status_code == 200, r.text
    return a, b


def _client(key: str) -> RAGClient:
    return RAGClient(key, "http://test/api/v1", transport=httpx.ASGITransport(app=app))


def _payload(result) -> dict:
    assert not result.isError, result.content
    return json.loads(result.content[0].text)


# ── MCP tools ──────────────────────────────────────────────────────────────────

async def test_tool_listing():
    _, key = create_api_key("alice", "desktop", tenant="acme")
    async with create_connected_server_and_client_session(build_server(_client(key))) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    assert set(tools) == {"search_knowledge_base", "ask"}
    assert set(tools["search_knowledge_base"].inputSchema["properties"]) == {"query", "top_k"}
    assert "question" in tools["ask"].inputSchema["properties"]
    # The tenant is never something the model can choose.
    for tool in tools.values():
        assert "tenant" not in json.dumps(tool.inputSchema).lower()


async def test_tenant_a_key_never_returns_tenant_b_chunks(tenants):
    a, b = tenants
    _, key_a = create_api_key("alice", "desktop", tenant=a, role="viewer")
    async with create_connected_server_and_client_session(build_server(_client(key_a))) as session:
        found = _payload(await session.call_tool("search_knowledge_base",
                                                 {"query": "Project Falcon budget codeword", "top_k": 10}))
        answer = _payload(await session.call_tool("ask", {"question": "What is the budget of Project Falcon?"}))

    assert found["results"], "tenant A's own document must be found"
    assert all(r["source"] == f"{a}-falcon.md" for r in found["results"])
    blob = json.dumps(found) + json.dumps(answer)
    assert "BLUEHERON" not in blob and "99 million" not in blob and b not in blob
    assert "7 million" in answer["answer"]
    assert answer["citations"] and all(c["source"] == f"{a}-falcon.md" for c in answer["citations"])


async def test_ask_supports_follow_up_conversations(tenants):
    a, _ = tenants
    _, key = create_api_key("alice", "desktop", tenant=a)
    async with create_connected_server_and_client_session(build_server(_client(key))) as session:
        first = _payload(await session.call_tool("ask", {"question": "What is the budget of Project Falcon?"}))
        assert first["conversation_id"]
        second = _payload(await session.call_tool(
            "ask", {"question": "when does its steering committee meet?",
                    "conversation_id": first["conversation_id"]}))
    assert second["conversation_id"] == first["conversation_id"]
    assert "Tuesday" in second["answer"]


async def test_unscoped_key_is_refused_with_a_clear_error():
    _, key = create_api_key("alice", "legacy")  # no tenant
    async with create_connected_server_and_client_session(build_server(_client(key))) as session:
        result = await session.call_tool("search_knowledge_base", {"query": "anything"})
    assert result.isError
    assert "not scoped to a tenant" in result.content[0].text


async def test_revoked_key_is_rejected(tenants):
    a, _ = tenants
    record, key = create_api_key("alice", "desktop", tenant=a)
    revoke_api_key(record.id, "alice")
    async with create_connected_server_and_client_session(build_server(_client(key))) as session:
        result = await session.call_tool("ask", {"question": "What is the budget of Project Falcon?"})
    assert result.isError and "401" in result.content[0].text


# ── Same RBAC on the REST API ──────────────────────────────────────────────────

def test_key_scoped_to_a_cannot_reach_b_over_rest(tenants):
    a, b = tenants
    _, key_a = create_api_key("alice", "k", tenant=a, role="owner")
    with TestClient(app) as client:
        h = {"X-API-Key": key_a}
        assert client.post("/api/v1/rag/search", json={"tenant": a, "query": "Falcon"}, headers=h).status_code == 200
        for path, body in (("/api/v1/rag/search", {"tenant": b, "query": "Falcon"}),
                           ("/api/v1/rag/ask", {"tenant": b, "question": "Falcon budget?"}),
                           ("/api/v1/rag/index", {"tenant": b, "text": "x", "filename": "x.md"})):
            r = client.post(path, json=body, headers=h)
            assert r.status_code == 403, (path, r.text)
        assert client.get(f"/api/v1/conversations?tenant={b}", headers=h).status_code == 403
        assert client.get(f"/api/v1/tenant-settings/{b}", headers=h).status_code == 403


def test_websocket_applies_the_same_tenant_checks(tenants):
    a, b = tenants
    _, key_a = create_api_key("alice", "k", tenant=a, role="viewer")
    with TestClient(app) as client:
        with client.websocket_connect(f"/api/v1/rag/stream/ws?api_key={key_a}") as ws:
            ws.send_json({"type": "ask", "question": "What is the Falcon budget?", "tenant": b})
            denied = ws.receive_json()
            assert denied["event"] == "error" and denied["status"] == 403
            ws.send_json({"type": "ask", "question": "What is the Falcon budget?", "tenant": a})
            events = []
            while not events or events[-1]["event"] not in ("complete", "error"):
                events.append(ws.receive_json())
        assert events[-1]["event"] == "complete"
        assert "7 million" in events[-1]["answer"] and "BLUEHERON" not in json.dumps(events)

        with client.websocket_connect("/api/v1/rag/stream/ws?api_key=sk-bogus") as ws:
            assert ws.receive_json()["status"] == 401

        # Anonymous callers can't reach a registered organisation.
        slug = f"org-{uuid.uuid4().hex[:6]}"
        create_organization(slug, "Org", owner_id="someone-else")
        try:
            with client.websocket_connect("/api/v1/rag/stream/ws") as ws:
                ws.send_json({"type": "ask", "question": "x", "tenant": slug})
                assert ws.receive_json()["status"] == 403
        finally:
            delete_organization(slug)


def test_bearer_api_key_and_invalid_keys():
    _, key = create_api_key("alice", "k", tenant="acme")
    with TestClient(app) as client:
        me = client.get("/api/v1/security/whoami", headers={"Authorization": f"Bearer {key}"}).json()
        assert me == {"user_id": "alice", "auth": "api_key", "tenant": "acme",
                      "role": "member", "key_name": "k"}
        # A wrong key is a 401 even on endpoints that allow anonymous callers.
        r = client.post("/api/v1/rag/search", json={"tenant": "acme", "query": "x"},
                        headers={"X-API-Key": "sk-not-a-real-key"})
        assert r.status_code == 401


def test_role_permissions():
    _, viewer = create_api_key("alice", "v", tenant="acme", role="viewer")
    _, member = create_api_key("alice", "m", tenant="acme", role="member")
    _, admin = create_api_key("alice", "a", tenant="acme", role="admin")
    doc = {"tenant": "acme", "text": "# Note\n\nRBAC test document.", "filename": "rbac.md"}
    with TestClient(app) as client:
        assert client.post("/api/v1/rag/index", json=doc, headers={"X-API-Key": viewer}).status_code == 403
        assert client.post("/api/v1/rag/index", json=doc, headers={"X-API-Key": member}).status_code == 200
        put = {"few_shot_enabled": True}
        assert client.put("/api/v1/tenant-settings/acme", json=put, headers={"X-API-Key": member}).status_code == 403
        assert client.put("/api/v1/tenant-settings/acme", json=put, headers={"X-API-Key": admin}).status_code == 200


def test_registered_organisation_requires_membership():
    slug = f"org-{uuid.uuid4().hex[:6]}"
    create_organization(slug, "Org", owner_id="someone-else")
    try:
        with TestClient(app) as client:
            h = {"Authorization": "Bearer dev"}
            body = {"tenant": slug, "query": "x"}
            assert client.post("/api/v1/rag/search", json=body, headers=h).status_code == 403
            add_member("dev", slug, role="viewer")
            assert client.post("/api/v1/rag/search", json=body, headers=h).status_code == 200
            # A key never holds more rights than its creator.
            r = client.post("/api/v1/security/api-keys",
                            json={"name": "k", "tenant": slug, "role": "admin"}, headers=h)
            assert r.status_code == 403
            r = client.post("/api/v1/security/api-keys",
                            json={"name": "k", "tenant": slug, "role": "viewer"}, headers=h)
            assert r.status_code == 201 and r.json()["tenant"] == slug
    finally:
        delete_organization(slug)


def test_api_key_cannot_create_api_keys():
    _, key = create_api_key("alice", "k", tenant="acme", role="owner")
    with TestClient(app) as client:
        r = client.post("/api/v1/security/api-keys", json={"name": "x"}, headers={"X-API-Key": key})
    assert r.status_code == 403


def test_keys_survive_a_process_restart():
    """Keys are mirrored to disk so the API and other processes share them."""
    _, key = create_api_key("alice", "k", tenant="acme")
    apikeys._store.clear()
    apikeys._mtime = None   # as if a fresh process started
    record = apikeys.verify_api_key(key)
    assert record is not None and record.tenant == "acme"
