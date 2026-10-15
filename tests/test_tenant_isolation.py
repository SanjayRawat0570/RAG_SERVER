"""Tenant isolation across the whole API (P0 security fix).

The route sweep below finds every HTTP route that accepts a tenant or namespace
(body field, query parameter or path parameter) by introspecting the app, and
asserts that an API key for tenant A gets 403 when it names tenant B. A new
route that forgets the check fails this test without anyone listing it here.
"""
from __future__ import annotations

import types
import typing
from typing import Any, Literal, get_args, get_origin

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.main import app
from app.rag.security.apikeys import create_api_key
from app.rag.tenants import add_member, create_organization, get_organization

client = TestClient(app)

A, B = "tenant-a", "tenant-b"
TENANT_FIELDS = {"tenant", "namespace"}
TENANT_PATH = {"tenant", "slug"}


def _key(tenant: str, role: str = "owner") -> dict[str, str]:
    _, plaintext = create_api_key("owner-of-both", f"key-{tenant}", tenant=tenant, role=role)
    return {"X-API-Key": plaintext}


@pytest.fixture(scope="module")
def key_a() -> dict[str, str]:
    # The key's creator belongs to both organisations, so only the key's scope can refuse.
    for slug in (A, B):
        if get_organization(slug) is None:
            create_organization(slug=slug, name=slug, owner_id="owner-of-both")
            add_member(user_id="owner-of-both", org_slug=slug, role="owner")
    return _key(A)


# ── Dummy request construction ─────────────────────────────────────────────────

def _dummy(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        return _dummy(args[0]) if args else None
    if origin is typing.Annotated:
        return _dummy(get_args(annotation)[0])
    if origin is Literal:
        return get_args(annotation)[0]
    if origin in (list, set, tuple):
        args = get_args(annotation)
        return [_dummy(args[0]) if args else "x"]
    if origin is dict or annotation is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _body(annotation)
    return {str: "query text", int: 1, float: 0.5, bool: True}.get(annotation, "x")


def _body(model: type[BaseModel]) -> dict[str, Any]:
    body = {}
    for name, field in model.model_fields.items():
        if name in TENANT_FIELDS:
            body[name] = B
        elif field.is_required():
            body[name] = _dummy(field.annotation)
    return body


def tenant_routes() -> list[tuple[str, str, APIRoute]]:
    found = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        d = route.dependant
        names = {p.name for p in d.query_params} | {p.name for p in d.path_params if p.name in TENANT_PATH}
        for p in d.body_params:
            names |= set(getattr(p.type_, "model_fields", {}) or {}) & TENANT_FIELDS
            names |= {p.name} & TENANT_FIELDS  # form fields (multipart uploads)
        if names & (TENANT_FIELDS | TENANT_PATH):
            for method in sorted(route.methods):
                found.append((method, route.path, route))
    return found


def _request(method: str, route: APIRoute, headers: dict[str, str]):
    d = route.dependant
    path = route.path
    for p in d.path_params:
        path = path.replace("{" + p.name + "}", B if p.name in TENANT_PATH else "x").replace(
            "{" + p.name + ":path}", "x")
    params = {p.name: (B if p.name in TENANT_FIELDS else _dummy(p.field_info.annotation))
              for p in d.query_params if p.name in TENANT_FIELDS or p.required}
    body = None
    if d.body_params:
        p = d.body_params[0]
        model = p.type_
        body = _body(model) if isinstance(model, type) and issubclass(model, BaseModel) else {}
        if len(d.body_params) > 1 or getattr(p.field_info, "embed", False):
            body = {p.name: body}
    if any(type(p.field_info).__name__ in ("File", "Form") for p in d.body_params):
        return client.request(method, path, params=params, headers=headers, data={"tenant": B},
                              files={"file": ("doc.md", b"# Doc\n\nHello.", "text/markdown")})
    return client.request(method, path, params=params, headers=headers, json=body)


ROUTES = tenant_routes()


def test_sweep_finds_the_known_tenant_routes():
    paths = {path for _, path, _ in ROUTES}
    for expected in ("/api/v1/search/keyword", "/api/v1/search/semantic", "/api/v1/search/hybrid",
                     "/api/v1/vectors/search", "/api/v1/vectors/records", "/api/v1/rag/answer",
                     "/api/v1/rag/ask", "/api/v1/graph/stats", "/api/v1/graph/entities",
                     "/api/v1/documents/upload", "/api/v1/conversations", "/api/v1/cost/usage"):
        assert expected in paths
    assert len(ROUTES) >= 60


@pytest.mark.parametrize("method,path,route", ROUTES, ids=[f"{m} {p}" for m, p, _ in ROUTES])
def test_key_for_tenant_a_is_refused_on_tenant_b(method, path, route, key_a):
    r = _request(method, route, key_a)
    assert r.status_code == 403, f"{method} {path} -> {r.status_code}: {r.text[:200]}"


# ── Data actually stays put ────────────────────────────────────────────────────

def _index(tenant: str, text: str, filename: str) -> None:
    from app.rag.security.scope import set_principal

    set_principal(None)
    r = client.post("/api/v1/rag/index", headers=_key(tenant),
                    json={"tenant": tenant, "text": text, "filename": filename})
    assert r.status_code == 200, r.text


def test_secret_in_b_never_reaches_a(key_a):
    _index(B, "# Falcon\n\nThe Falcon code word is NIGHTJAR.", "falcon.md")
    attacks = [
        ("POST", "/api/v1/search/keyword", {"query": "Falcon code word", "store": "kb", "namespace": B}),
        ("POST", "/api/v1/rag/search", {"query": "Falcon code word", "tenant": B}),
        ("POST", "/api/v1/rag/ask", {"question": "What is the Falcon code word?", "tenant": B}),
    ]
    for method, path, body in attacks:
        r = client.request(method, path, headers=key_a, json=body)
        assert r.status_code == 403 and "NIGHTJAR" not in r.text, (path, r.status_code)


def test_user_supplied_workflow_cannot_read_another_tenant(key_a):
    """/workflows/run executes arbitrary nodes — the data layer refuses B's namespace."""
    _index(B, "# Falcon\n\nThe Falcon code word is NIGHTJAR.", "falcon.md")
    wf = {"workflow": {"name": "steal", "nodes": [
        {"id": "in", "type": "input"},
        {"id": "kw", "type": "keyword_search",
         "config": {"store": "kb", "namespace": B, "query": "Falcon code word"}},
        {"id": "out", "type": "output", "config": {"value": "$.kw"}}],
        "edges": [{"source": "in", "target": "kw"}, {"source": "kw", "target": "out"}]}, "inputs": {}}
    r = client.post("/api/v1/workflows/run", headers=key_a, json=wf)
    assert r.status_code == 403 and "NIGHTJAR" not in r.text


def test_key_still_works_on_its_own_tenant(key_a):
    _index(A, "# Parking\n\nThe garage opens at 7am.", "parking.md")
    r = client.post("/api/v1/search/keyword", headers=key_a,
                    json={"query": "garage opens", "store": "kb", "namespace": A})
    assert r.status_code == 200 and r.json()["hits"]


def test_listings_hide_other_tenants(key_a):
    _index(B, "# Falcon\n\nSecret.", "falcon2.md")
    stores = client.get("/api/v1/vectors/stores", headers=key_a).json()["stores"]
    assert all(B not in s["namespaces"] for s in stores)
    ns = client.get("/api/v1/vectors/stores/kb/ns", headers=key_a).json()["namespaces"]
    assert B not in {n["namespace"] for n in ns}
    assert B not in client.get("/api/v1/budgets", headers=key_a).json()["budgets"]


def test_viewer_cannot_grant_itself_owner():
    viewer = _key(A, role="viewer")
    r = client.post(f"/api/v1/tenants/{A}/members", headers=viewer,
                    json={"user_id": "someone", "role": "owner"})
    assert r.status_code == 403


# ── No credentials (P0: anonymous access removed) ──────────────────────────────

@pytest.mark.parametrize("method,path,route", ROUTES, ids=[f"{m} {p}" for m, p, _ in ROUTES])
def test_no_credentials_is_401_on_every_tenant_route(method, path, route):
    r = _request(method, route, {})
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}: {r.text[:200]}"


def test_anonymous_write_and_read_are_refused():
    planted = {"tenant": "acme", "filename": "planted.md", "text": "# Beacon\n\nOwned by the attacker."}
    assert client.post("/api/v1/rag/index", json=planted).status_code == 401
    assert client.post("/api/v1/rag/ask", json={"tenant": "acme", "question": "Who owns Beacon?"}).status_code == 401


def test_demo_mode_is_opt_in_and_read_only(monkeypatch, key_a):
    from app.config import settings

    monkeypatch.setattr(settings, "demo_mode", True)
    assert client.post("/api/v1/rag/ask", json={"tenant": "demo-ws", "question": "hi"}).status_code == 200
    planted = {"tenant": "demo-ws", "filename": "x.md", "text": "# X\n\nPlanted."}
    assert client.post("/api/v1/rag/index", json=planted).status_code == 403
    # Registered organisations stay members-only, even in demo mode.
    assert client.post("/api/v1/rag/ask", json={"tenant": B, "question": "hi"}).status_code == 403


def test_any_bearer_is_not_a_user_outside_development(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "app_env", "production")
    r = client.post("/api/v1/rag/ask", headers={"Authorization": "Bearer whatever"},
                    json={"tenant": "acme", "question": "hi"})
    assert r.status_code == 401


# ── Organisation registry survives a restart (P0) ──────────────────────────────

def test_registered_org_stays_closed_after_restart():
    import os
    import subprocess
    import sys
    import uuid

    from app.rag import tenants
    from app.rag.tenants import registry

    slug = f"persist-{uuid.uuid4().hex[:6]}"
    dev = {"Authorization": "Bearer dev"}
    assert client.post("/api/v1/tenants", headers=dev, json={"slug": slug, "name": "P"}).status_code == 201
    outsider = _key("someone-else-tenant")

    # Restart in-process: drop the in-memory registry and load it from disk.
    registry._orgs.clear()
    registry._teams.clear()
    registry._memberships.clear()
    assert tenants.get_organization(slug) is None
    tenants.load_registry()
    assert tenants.get_organization(slug) is not None
    assert tenants.is_member("dev", slug)
    assert client.post("/api/v1/rag/ask", headers=outsider, json={"tenant": slug, "question": "x"}).status_code == 403

    # A fresh interpreter (a real restart) sees the same organisation and owner.
    code = ("from app.rag.tenants import get_organization, is_member;"
            f"o = get_organization('{slug}'); print(o is not None and is_member('dev', '{slug}'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy(),
                         cwd=os.path.dirname(os.path.dirname(__file__)), timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "True", out.stderr[-500:]
