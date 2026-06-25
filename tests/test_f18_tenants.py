"""Tests for F18: Multi-Tenancy & User Isolation."""
from __future__ import annotations

import pytest

from app.rag.tenants import (
    add_member, create_organization, create_team,
    delete_organization, delete_team,
    get_organization, get_team,
    get_usage, get_user_orgs, is_member,
    list_members, list_organizations, list_teams,
    org_namespace, record_usage, remove_member,
    reset_registry, reset_usage, resolve_namespace,
    team_namespace, user_namespace,
)
from app.rag.tenants.models import Organization, Team


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean():
    reset_registry()
    yield
    reset_registry()


# ── Organisation CRUD ──────────────────────────────────────────────────────────

def test_f18_create_org():
    org = create_organization("acme", "Acme Corp", "user-1")
    assert isinstance(org, Organization)
    assert org.slug == "acme"
    assert org.owner_id == "user-1"


def test_f18_create_org_invalid_slug():
    with pytest.raises(Exception):
        create_organization("UPPER", "Bad", "user-1")


def test_f18_create_org_duplicate_slug():
    create_organization("acme", "Acme", "user-1")
    with pytest.raises(ValueError, match="already exists"):
        create_organization("acme", "Acme 2", "user-2")


def test_f18_get_org_returns_none_when_missing():
    assert get_organization("nope") is None


def test_f18_list_orgs_returns_all():
    create_organization("org-a", "A", "u1")
    create_organization("org-b", "B", "u2")
    orgs = list_organizations()
    slugs = [o.slug for o in orgs]
    assert "org-a" in slugs
    assert "org-b" in slugs


def test_f18_delete_org_removes_it():
    create_organization("to-del", "Del Me", "u1")
    assert delete_organization("to-del") is True
    assert get_organization("to-del") is None


def test_f18_delete_org_also_removes_teams_and_members():
    create_organization("myco", "MyCo", "u1")
    create_team("myco", "eng", "Engineering")
    add_member("u2", "myco", "eng")
    delete_organization("myco")
    assert list_teams("myco") == []
    assert list_members("myco") == []


def test_f18_delete_org_unknown_returns_false():
    assert delete_organization("ghost") is False


# ── Team CRUD ──────────────────────────────────────────────────────────────────

def test_f18_create_team():
    create_organization("co", "Co", "u1")
    team = create_team("co", "eng", "Engineering")
    assert isinstance(team, Team)
    assert team.org_slug == "co"
    assert team.slug == "eng"


def test_f18_create_team_unknown_org():
    with pytest.raises(ValueError, match="does not exist"):
        create_team("ghost", "eng", "Engineering")


def test_f18_create_team_duplicate():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    with pytest.raises(ValueError, match="already exists"):
        create_team("co", "eng", "Engineering 2")


def test_f18_list_teams():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    create_team("co", "product", "Product")
    teams = list_teams("co")
    slugs = [t.slug for t in teams]
    assert "eng" in slugs
    assert "product" in slugs


def test_f18_get_team():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    t = get_team("co", "eng")
    assert t is not None
    assert t.name == "Engineering"


def test_f18_get_team_missing():
    create_organization("co", "Co", "u1")
    assert get_team("co", "ghost") is None


def test_f18_delete_team():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    add_member("u2", "co", "eng")
    assert delete_team("co", "eng") is True
    assert get_team("co", "eng") is None
    # Membership also removed.
    assert list_members("co", team_slug="eng") == []


def test_f18_delete_team_unknown():
    create_organization("co", "Co", "u1")
    assert delete_team("co", "ghost") is False


# ── Membership ─────────────────────────────────────────────────────────────────

def test_f18_add_member():
    create_organization("co", "Co", "u1")
    m = add_member("u2", "co", role="member")
    assert m.user_id == "u2"
    assert m.org_slug == "co"
    assert m.role == "member"


def test_f18_is_member_true():
    create_organization("co", "Co", "u1")
    add_member("u2", "co")
    assert is_member("u2", "co") is True


def test_f18_is_member_false():
    create_organization("co", "Co", "u1")
    assert is_member("stranger", "co") is False


def test_f18_list_members_all():
    create_organization("co", "Co", "u1")
    add_member("u2", "co")
    add_member("u3", "co")
    members = list_members("co")
    uids = [m.user_id for m in members]
    assert "u2" in uids
    assert "u3" in uids


def test_f18_list_members_by_team():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    add_member("u2", "co", "eng")
    add_member("u3", "co")  # org-level only
    eng_members = list_members("co", team_slug="eng")
    assert len(eng_members) == 1
    assert eng_members[0].user_id == "u2"


def test_f18_remove_member():
    create_organization("co", "Co", "u1")
    add_member("u2", "co")
    assert remove_member("u2", "co") is True
    assert is_member("u2", "co") is False


def test_f18_remove_member_not_present():
    create_organization("co", "Co", "u1")
    assert remove_member("ghost", "co") is False


def test_f18_get_user_orgs():
    create_organization("org-a", "A", "u1")
    create_organization("org-b", "B", "u2")
    add_member("u1", "org-a")
    add_member("u1", "org-b")
    memberships = get_user_orgs("u1")
    slugs = [m.org_slug for m in memberships]
    assert "org-a" in slugs
    assert "org-b" in slugs


# ── Namespace helpers ──────────────────────────────────────────────────────────

def test_f18_user_namespace():
    assert user_namespace("alice") == "user:alice"


def test_f18_team_namespace():
    assert team_namespace("acme", "eng") == "acme/eng"


def test_f18_org_namespace():
    assert org_namespace("acme") == "acme"


def test_f18_resolve_namespace_personal():
    ns = resolve_namespace("alice")
    assert ns == "user:alice"


def test_f18_resolve_namespace_org_level():
    create_organization("co", "Co", "u1")
    add_member("alice", "co")
    ns = resolve_namespace("alice", org_slug="co")
    assert ns == "co"


def test_f18_resolve_namespace_team_level():
    create_organization("co", "Co", "u1")
    create_team("co", "eng", "Engineering")
    add_member("alice", "co", "eng")
    ns = resolve_namespace("alice", org_slug="co", team_slug="eng")
    assert ns == "co/eng"


def test_f18_resolve_namespace_not_member_fallback():
    create_organization("co", "Co", "u1")
    # alice is NOT a member → falls back to personal namespace.
    ns = resolve_namespace("alice", org_slug="co")
    assert ns == "user:alice"


# ── Usage tracking ─────────────────────────────────────────────────────────────

def test_f18_record_usage_increments_query_count():
    create_organization("co", "Co", "u1")
    record_usage("co", "u1", tokens=500, cost_usd=0.001)
    stats = get_usage("co")
    assert len(stats) == 1
    assert stats[0].total_queries == 1
    assert stats[0].total_tokens  == 500


def test_f18_record_usage_accumulates():
    create_organization("co", "Co", "u1")
    record_usage("co", "u1", tokens=100, cost_usd=0.01)
    record_usage("co", "u1", tokens=200, cost_usd=0.02)
    stats = get_usage("co")
    assert stats[0].total_queries == 2
    assert stats[0].total_tokens  == 300


def test_f18_record_usage_cache_hit():
    create_organization("co", "Co", "u1")
    record_usage("co", "u1", cache_hit=True)
    stats = get_usage("co")
    assert stats[0].cache_hits == 1


def test_f18_reset_usage():
    create_organization("co", "Co", "u1")
    record_usage("co", "u1", tokens=100)
    reset_usage("co")
    assert get_usage("co") == []


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f18_api_create_org():
    with _client() as c:
        resp = c.post("/api/v1/tenants", json={"slug": "test-org", "name": "Test Org"},
                      headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["slug"] == "test-org"
    assert data["owner_id"] == "dev"
    assert data["member_count"] == 1  # owner auto-added


def test_f18_api_create_org_duplicate():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "dup-org", "name": "Dup"},
               headers=AUTH)
        resp = c.post("/api/v1/tenants", json={"slug": "dup-org", "name": "Dup"},
                      headers=AUTH)
    assert resp.status_code == 409


def test_f18_api_list_orgs():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "list-org", "name": "List Org"},
               headers=AUTH)
        resp = c.get("/api/v1/tenants", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 1


def test_f18_api_get_org():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "detail-org", "name": "Detail"},
               headers=AUTH)
        resp = c.get("/api/v1/tenants/detail-org", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["slug"] == "detail-org"


def test_f18_api_get_org_not_found():
    with _client() as c:
        resp = c.get("/api/v1/tenants/ghost", headers=AUTH)
    assert resp.status_code == 404


def test_f18_api_delete_org():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "del-org", "name": "Del"},
               headers=AUTH)
        resp = c.delete("/api/v1/tenants/del-org", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "del-org"


def test_f18_api_create_team():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "team-org", "name": "TeamOrg"},
               headers=AUTH)
        resp = c.post("/api/v1/tenants/team-org/teams",
                      json={"slug": "eng", "name": "Engineering"}, headers=AUTH)
    assert resp.status_code == 201
    assert resp.json()["slug"] == "eng"


def test_f18_api_list_teams():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "tl-org", "name": "TL"}, headers=AUTH)
        c.post("/api/v1/tenants/tl-org/teams",
               json={"slug": "eng", "name": "Engineering"}, headers=AUTH)
        c.post("/api/v1/tenants/tl-org/teams",
               json={"slug": "product", "name": "Product"}, headers=AUTH)
        resp = c.get("/api/v1/tenants/tl-org/teams", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] == 2


def test_f18_api_add_member():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "mb-org", "name": "MB"}, headers=AUTH)
        resp = c.post("/api/v1/tenants/mb-org/members",
                      json={"user_id": "alice", "role": "member"}, headers=AUTH)
    assert resp.status_code == 201
    assert resp.json()["user_id"] == "alice"


def test_f18_api_list_members():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "lm-org", "name": "LM"}, headers=AUTH)
        c.post("/api/v1/tenants/lm-org/members",
               json={"user_id": "bob"}, headers=AUTH)
        resp = c.get("/api/v1/tenants/lm-org/members", headers=AUTH)
    assert resp.status_code == 200
    # dev (owner) + bob
    assert resp.json()["total"] >= 2


def test_f18_api_usage():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "usage-org", "name": "U"}, headers=AUTH)
        resp = c.get("/api/v1/tenants/usage-org/usage", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries"  in data
    assert "total_tokens"   in data
    assert "cache_hit_rate" in data


def test_f18_api_reset_usage():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "reset-org", "name": "R"}, headers=AUTH)
        resp = c.post("/api/v1/tenants/reset-org/reset", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_f18_api_namespace_personal():
    with _client() as c:
        resp = c.get("/api/v1/tenants/me/namespace", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace"] == "user:dev"
    assert data["personal_namespace"] == "user:dev"


def test_f18_api_namespace_org_level():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "ns-org", "name": "NS"}, headers=AUTH)
        resp = c.get("/api/v1/tenants/me/namespace?org_slug=ns-org", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["namespace"] == "ns-org"


def test_f18_api_namespace_team_level():
    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "ns2-org", "name": "NS2"}, headers=AUTH)
        c.post("/api/v1/tenants/ns2-org/teams",
               json={"slug": "devs", "name": "Devs"}, headers=AUTH)
        c.post("/api/v1/tenants/ns2-org/members",
               json={"user_id": "dev", "team_slug": "devs"}, headers=AUTH)
        resp = c.get("/api/v1/tenants/me/namespace?org_slug=ns2-org&team_slug=devs",
                     headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["namespace"] == "ns2-org/devs"


def test_f18_api_rag_answer_uses_tenant_namespace():
    """Queries with org_slug get routed to the org's namespace (F18 integration)."""
    from app.rag.embeddings import embed_texts
    from app.rag.vectorstore import VectorRecord, get_store, reset_stores

    reset_stores()
    store = get_store("tenant-store", 256)
    vec   = embed_texts(["Organisation revenue details."], "local-hash", 256)[0]
    store.upsert([VectorRecord(id="d1", vector=vec,
                               metadata={"text": "Organisation revenue details."})],
                 namespace="tenant-org")

    with _client() as c:
        c.post("/api/v1/tenants", json={"slug": "tenant-org", "name": "TenantOrg"},
               headers=AUTH)
        resp = c.post("/api/v1/rag/answer", json={
            "query":    "revenue",
            "store":    "tenant-store",
            "provider": "stub",
            "org_slug": "tenant-org",
        }, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data

    reset_stores()


def test_f18_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/tenants")
    assert resp.status_code == 401
