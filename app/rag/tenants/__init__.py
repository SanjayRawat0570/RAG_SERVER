"""Tenant / multi-tenancy module (F18)."""
from app.rag.tenants.models import Membership, Organization, Team, UsageStat
from app.rag.tenants.registry import (
    add_member,
    create_organization,
    create_team,
    delete_organization,
    delete_team,
    get_organization,
    get_team,
    get_usage,
    get_user_orgs,
    is_member,
    list_members,
    list_organizations,
    list_teams,
    org_namespace,
    record_usage,
    remove_member,
    reset_registry,
    reset_usage,
    resolve_namespace,
    team_namespace,
    user_namespace,
)

__all__ = [
    "Organization", "Team", "Membership", "UsageStat",
    "create_organization", "get_organization", "list_organizations", "delete_organization",
    "create_team", "get_team", "list_teams", "delete_team",
    "add_member", "remove_member", "list_members", "get_user_orgs", "is_member",
    "user_namespace", "team_namespace", "org_namespace", "resolve_namespace",
    "record_usage", "get_usage", "reset_usage",
    "reset_registry",
]
