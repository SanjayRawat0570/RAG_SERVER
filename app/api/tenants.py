"""Multi-Tenancy API (F18).

Hierarchy:
    Organization  (company / project)
      └── Team    (department / sub-group)
            └── User membership (role: owner | admin | member | viewer)

Namespace isolation:
    Personal data  → "user:{user_id}"
    Team data      → "{org_slug}/{team_slug}"
    Org-wide data  → "{org_slug}"

All vector store searches, caches, and document uploads respect this namespace,
so each organisation/team/user sees only their own data.

Endpoints
---------
POST   /tenants                         Create organisation
GET    /tenants                         List organisations
GET    /tenants/{slug}                  Organisation detail + member count
DELETE /tenants/{slug}                  Delete org + all data
POST   /tenants/{slug}/teams            Create team inside org
GET    /tenants/{slug}/teams            List teams in org
DELETE /tenants/{slug}/teams/{team}     Delete team
POST   /tenants/{slug}/members          Add user to org (and optionally a team)
GET    /tenants/{slug}/members          List members (optionally filter by team)
DELETE /tenants/{slug}/members/{uid}    Remove user from org
GET    /tenants/{slug}/usage            Aggregated usage stats for the org
POST   /tenants/{slug}/reset            Reset usage counters (not data)
GET    /tenants/me/namespace            Resolve caller's effective namespace
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.tenants import (
    Organization, Team,
    add_member, create_organization, create_team,
    delete_organization, delete_team,
    get_organization, get_team,
    get_usage, get_user_orgs, is_member,
    list_members, list_organizations, list_teams,
    record_usage, remove_member, reset_usage, resolve_namespace,
    user_namespace,
)

router = APIRouter(prefix="/tenants", tags=["tenants"])


# ── Request / Response models ──────────────────────────────────────────────────

class CreateOrgRequest(BaseModel):
    slug:     str = Field(..., min_length=2, max_length=63)
    name:     str = Field(..., min_length=1, max_length=255)
    settings: dict[str, Any] = Field(default_factory=dict)


class CreateTeamRequest(BaseModel):
    slug: str = Field(..., min_length=2, max_length=63)
    name: str = Field(..., min_length=1, max_length=255)


class AddMemberRequest(BaseModel):
    user_id:   str = Field(..., min_length=1)
    team_slug: str | None = None
    role:      str = "member"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_org(slug: str) -> Organization:
    org = get_organization(slug)
    if org is None:
        raise HTTPException(404, f"Organisation '{slug}' not found")
    return org


def _require_member(user_id: str, org_slug: str) -> None:
    if not is_member(user_id, org_slug):
        raise HTTPException(403, "You are not a member of this organisation")


def _org_response(org: Organization) -> dict[str, Any]:
    members = list_members(org.slug)
    teams   = list_teams(org.slug)
    return {
        **org.model_dump(),
        "member_count": len(members),
        "team_count":   len(teams),
    }


# ── Organisation endpoints ─────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_org(req: CreateOrgRequest, user: CurrentUser) -> dict[str, Any]:
    """Create a new organisation owned by the calling user."""
    try:
        org = create_organization(slug=req.slug, name=req.name,
                                  owner_id=user["id"], settings=req.settings)
        # Owner becomes the first member.
        add_member(user_id=user["id"], org_slug=req.slug, role="owner")
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _org_response(org)


@router.get("")
async def list_orgs(user: CurrentUser) -> dict[str, Any]:
    """List all organisations the calling user belongs to."""
    my_orgs = {m.org_slug for m in get_user_orgs(user["id"])}
    all_orgs = list_organizations()
    visible  = [o for o in all_orgs if o.slug in my_orgs]
    return {
        "total":         len(visible),
        "organizations": [_org_response(o) for o in visible],
    }


@router.get("/me/namespace")
async def my_namespace(
    user:      CurrentUser,
    org_slug:  str | None = Query(None),
    team_slug: str | None = Query(None),
) -> dict[str, str]:
    """Resolve the effective namespace for this user.

    Pass org_slug / team_slug to get the most specific namespace you belong to.
    Omit both to get your personal namespace.
    """
    ns = resolve_namespace(user["id"], org_slug=org_slug, team_slug=team_slug)
    return {
        "user_id":        user["id"],
        "org_slug":       org_slug  or "",
        "team_slug":      team_slug or "",
        "namespace":      ns,
        "personal_namespace": user_namespace(user["id"]),
    }


@router.get("/{slug}")
async def get_org(slug: str, user: CurrentUser) -> dict[str, Any]:
    org = _require_org(slug)
    _require_member(user["id"], slug)
    return _org_response(org)


@router.delete("/{slug}")
async def remove_org(slug: str, user: CurrentUser) -> dict[str, Any]:
    org = _require_org(slug)
    # Only the owner can delete the org.
    if org.owner_id != user["id"]:
        raise HTTPException(403, "Only the owner can delete an organisation")
    delete_organization(slug)
    return {"deleted": slug, "status": "ok"}


# ── Team endpoints ─────────────────────────────────────────────────────────────

@router.post("/{slug}/teams", status_code=201)
async def create_org_team(slug: str, req: CreateTeamRequest,
                          user: CurrentUser) -> dict[str, Any]:
    _require_org(slug)
    _require_member(user["id"], slug)
    try:
        team = create_team(org_slug=slug, slug=req.slug, name=req.name)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return team.model_dump()


@router.get("/{slug}/teams")
async def list_org_teams(slug: str, user: CurrentUser) -> dict[str, Any]:
    _require_org(slug)
    _require_member(user["id"], slug)
    teams = list_teams(slug)
    return {"total": len(teams), "teams": [t.model_dump() for t in teams]}


@router.delete("/{slug}/teams/{team_slug}")
async def remove_org_team(slug: str, team_slug: str, user: CurrentUser) -> dict[str, Any]:
    org = _require_org(slug)
    _require_member(user["id"], slug)
    if org.owner_id != user["id"]:
        raise HTTPException(403, "Only the owner can delete teams")
    if not delete_team(slug, team_slug):
        raise HTTPException(404, f"Team '{team_slug}' not found in '{slug}'")
    return {"deleted": team_slug, "org": slug, "status": "ok"}


# ── Member endpoints ───────────────────────────────────────────────────────────

@router.post("/{slug}/members", status_code=201)
async def add_org_member(slug: str, req: AddMemberRequest,
                         user: CurrentUser) -> dict[str, Any]:
    org = _require_org(slug)
    _require_member(user["id"], slug)
    if req.team_slug and not get_team(slug, req.team_slug):
        raise HTTPException(404, f"Team '{req.team_slug}' not found")
    m = add_member(user_id=req.user_id, org_slug=slug,
                   team_slug=req.team_slug, role=req.role)
    return m.model_dump()


@router.get("/{slug}/members")
async def list_org_members(slug: str, user: CurrentUser,
                           team: str | None = Query(None)) -> dict[str, Any]:
    _require_org(slug)
    _require_member(user["id"], slug)
    members = list_members(slug, team_slug=team)
    return {
        "total":   len(members),
        "members": [m.model_dump() for m in members],
    }


@router.delete("/{slug}/members/{uid}")
async def remove_org_member(slug: str, uid: str, user: CurrentUser) -> dict[str, Any]:
    org = _require_org(slug)
    _require_member(user["id"], slug)
    # Only owner/admin can remove others; any user can remove themselves.
    if uid != user["id"] and org.owner_id != user["id"]:
        raise HTTPException(403, "Only the owner can remove other members")
    removed = remove_member(user_id=uid, org_slug=slug)
    if not removed:
        raise HTTPException(404, f"User '{uid}' is not a member of '{slug}'")
    return {"removed": uid, "org": slug, "status": "ok"}


# ── Usage endpoints ────────────────────────────────────────────────────────────

@router.get("/{slug}/usage")
async def get_org_usage(slug: str, user: CurrentUser) -> dict[str, Any]:
    _require_org(slug)
    _require_member(user["id"], slug)
    stats = get_usage(slug)
    total_queries  = sum(s.total_queries  for s in stats)
    total_tokens   = sum(s.total_tokens   for s in stats)
    total_cost     = sum(s.total_cost_usd for s in stats)
    total_cache    = sum(s.cache_hits     for s in stats)
    return {
        "org_slug":        slug,
        "total_queries":   total_queries,
        "total_tokens":    total_tokens,
        "total_cost_usd":  round(total_cost, 6),
        "cache_hits":      total_cache,
        "cache_hit_rate":  round(total_cache / total_queries, 4) if total_queries else 0.0,
        "breakdown":       [s.model_dump() for s in stats],
    }


@router.post("/{slug}/reset")
async def reset_org_usage(slug: str, user: CurrentUser) -> dict[str, Any]:
    """Reset usage counters for the organisation (does not delete documents)."""
    org = _require_org(slug)
    if org.owner_id != user["id"]:
        raise HTTPException(403, "Only the owner can reset usage")
    reset_usage(slug)
    return {"reset": slug, "status": "ok"}
