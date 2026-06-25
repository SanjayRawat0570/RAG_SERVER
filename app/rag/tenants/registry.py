"""In-memory tenant registry (F18).

Stores Organizations, Teams, and Memberships.  In a production deployment
these would live in Supabase tables with RLS policies; here we use dicts so
the system stays offline-first with no external dependencies.

Key design:
- Namespace for a user's personal data:  user_id
- Namespace for a team's shared data:    {org_slug}/{team_slug}
- Namespace for an org-wide pool:        {org_slug}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.rag.tenants.models import Membership, Organization, Team, UsageStat

# ── In-memory stores ────────────────────────────────────────────────────────────

_orgs:        dict[str, Organization]     = {}   # slug → org
_teams:       dict[str, dict[str, Team]]  = {}   # org_slug → (team_slug → team)
_memberships: list[Membership]            = []
_usage:       dict[str, UsageStat]        = {}   # "{org_slug}:{team_slug}:{user_id}" → stat

_DEFAULT_ORG  = "dev-org"
_DEFAULT_TEAM = "dev-team"


def _seed_dev() -> None:
    """Seed a default dev organisation so offline tests have a valid context."""
    if _DEFAULT_ORG not in _orgs:
        create_organization(
            slug=_DEFAULT_ORG, name="Dev Organisation",
            owner_id="dev", settings={"allowed_providers": ["stub", "gemini"]},
        )
        create_team(_DEFAULT_ORG, slug=_DEFAULT_TEAM, name="Dev Team")
        add_member(user_id="dev", org_slug=_DEFAULT_ORG,
                   team_slug=_DEFAULT_TEAM, role="owner")


# ── Organisation CRUD ───────────────────────────────────────────────────────────

def create_organization(slug: str, name: str, owner_id: str,
                        settings: dict[str, Any] | None = None) -> Organization:
    if slug in _orgs:
        raise ValueError(f"Organisation '{slug}' already exists")
    org = Organization(
        id=str(uuid.uuid4()), slug=slug, name=name,
        owner_id=owner_id, settings=settings or {},
    )
    _orgs[slug] = org
    _teams.setdefault(slug, {})
    return org


def get_organization(slug: str) -> Organization | None:
    return _orgs.get(slug)


def list_organizations() -> list[Organization]:
    return list(_orgs.values())


def delete_organization(slug: str) -> bool:
    """Remove org, its teams, and all memberships."""
    global _memberships
    if slug not in _orgs:
        return False
    del _orgs[slug]
    _teams.pop(slug, None)
    _memberships = [m for m in _memberships if m.org_slug != slug]
    _usage_keys = [k for k in _usage if k.startswith(f"{slug}:")]
    for k in _usage_keys:
        del _usage[k]
    return True


# ── Team CRUD ───────────────────────────────────────────────────────────────────

def create_team(org_slug: str, slug: str, name: str) -> Team:
    if org_slug not in _orgs:
        raise ValueError(f"Organisation '{org_slug}' does not exist")
    if slug in _teams.get(org_slug, {}):
        raise ValueError(f"Team '{slug}' already exists in '{org_slug}'")
    team = Team(id=str(uuid.uuid4()), org_slug=org_slug, slug=slug, name=name)
    _teams.setdefault(org_slug, {})[slug] = team
    return team


def get_team(org_slug: str, team_slug: str) -> Team | None:
    return _teams.get(org_slug, {}).get(team_slug)


def list_teams(org_slug: str) -> list[Team]:
    return list(_teams.get(org_slug, {}).values())


def delete_team(org_slug: str, team_slug: str) -> bool:
    global _memberships
    teams = _teams.get(org_slug, {})
    if team_slug not in teams:
        return False
    del teams[team_slug]
    _memberships = [m for m in _memberships
                    if not (m.org_slug == org_slug and m.team_slug == team_slug)]
    return True


# ── Membership ──────────────────────────────────────────────────────────────────

def add_member(user_id: str, org_slug: str, team_slug: str | None = None,
               role: str = "member") -> Membership:
    m = Membership(user_id=user_id, org_slug=org_slug,
                   team_slug=team_slug, role=role)
    _memberships.append(m)
    return m


def remove_member(user_id: str, org_slug: str, team_slug: str | None = None) -> bool:
    global _memberships
    before = len(_memberships)
    _memberships = [
        m for m in _memberships
        if not (m.user_id == user_id and m.org_slug == org_slug
                and (team_slug is None or m.team_slug == team_slug))
    ]
    return len(_memberships) < before


def list_members(org_slug: str, team_slug: str | None = None) -> list[Membership]:
    return [
        m for m in _memberships
        if m.org_slug == org_slug
        and (team_slug is None or m.team_slug == team_slug)
    ]


def get_user_orgs(user_id: str) -> list[Membership]:
    return [m for m in _memberships if m.user_id == user_id]


def is_member(user_id: str, org_slug: str) -> bool:
    return any(m.user_id == user_id and m.org_slug == org_slug for m in _memberships)


# ── Namespace helpers ───────────────────────────────────────────────────────────

def user_namespace(user_id: str) -> str:
    """Personal data namespace — completely private to this user."""
    return f"user:{user_id}"


def team_namespace(org_slug: str, team_slug: str) -> str:
    """Shared namespace for a team within an org."""
    return f"{org_slug}/{team_slug}"


def org_namespace(org_slug: str) -> str:
    """Org-wide shared namespace."""
    return org_slug


def resolve_namespace(user_id: str, org_slug: str | None = None,
                      team_slug: str | None = None) -> str:
    """Pick the most specific namespace the user belongs to."""
    if org_slug and team_slug and is_member(user_id, org_slug):
        if get_team(org_slug, team_slug):
            return team_namespace(org_slug, team_slug)
    if org_slug and is_member(user_id, org_slug):
        return org_namespace(org_slug)
    return user_namespace(user_id)


# ── Usage tracking ──────────────────────────────────────────────────────────────

def _usage_key(org_slug: str, team_slug: str | None, user_id: str | None) -> str:
    return f"{org_slug}:{team_slug or ''}:{user_id or ''}"


def record_usage(org_slug: str, user_id: str, *, tokens: int = 0,
                 cost_usd: float = 0.0, cache_hit: bool = False,
                 team_slug: str | None = None) -> None:
    key = _usage_key(org_slug, team_slug, user_id)
    if key not in _usage:
        _usage[key] = UsageStat(org_slug=org_slug, team_slug=team_slug, user_id=user_id)
    stat = _usage[key]
    stat.total_queries  += 1
    stat.total_tokens   += tokens
    stat.total_cost_usd  = round(stat.total_cost_usd + cost_usd, 8)
    if cache_hit:
        stat.cache_hits += 1
    stat.last_active = datetime.now(timezone.utc)


def get_usage(org_slug: str) -> list[UsageStat]:
    prefix = f"{org_slug}:"
    return [v for k, v in _usage.items() if k.startswith(prefix)]


def reset_usage(org_slug: str) -> None:
    keys = [k for k in _usage if k.startswith(f"{org_slug}:")]
    for k in keys:
        del _usage[k]


# ── Registry reset (for tests) ──────────────────────────────────────────────────

def reset_registry() -> None:
    global _memberships
    _orgs.clear()
    _teams.clear()
    _memberships = []
    _usage.clear()
