"""Role-based access control for tenant data (F18/F25).

One check — :func:`authorize` — guards every tenant-scoped operation in the
REST API, and therefore the MCP server too (which calls the REST API with an
API key).

Who may act on tenant ``T``:

* **API key** — only if the key is scoped to ``T``; its role decides what it may do.
* **Signed-in user, ``T`` is a registered organisation** — only members of ``T``;
  their membership role decides.
* **Signed-in user, ``T`` is a free-form workspace name** (not an organisation) —
  allowed, as before roles existed. Register the workspace as an organisation
  (``POST /tenants``) to enforce membership.
* **Anonymous demo user** (only with ``DEMO_MODE`` on) — ``query`` on free-form
  workspaces, nothing else. Without demo mode there is no anonymous access.

Roles and permissions::

    viewer  query
    member  query, ingest
    admin   query, ingest, manage
    owner   query, ingest, manage
"""
from __future__ import annotations

ROLES = ("viewer", "member", "admin", "owner")
PERMISSIONS: dict[str, set[str]] = {
    "viewer": {"query"},
    "member": {"query", "ingest"},
    "admin":  {"query", "ingest", "manage"},
    "owner":  {"query", "ingest", "manage"},
}


class AccessDenied(Exception):
    """The caller may not perform this operation on this tenant."""


def role_rank(role: str) -> int:
    return ROLES.index(role) if role in ROLES else -1


def tenant_role(user: dict, tenant: str) -> str | None:
    """The caller's role in *tenant*, or None when the tenant is not an organisation."""
    from app.rag.tenants import get_organization, get_user_orgs

    if get_organization(tenant) is None:
        return None
    roles = [m.role for m in get_user_orgs(user.get("id", "")) if m.org_slug == tenant]
    if not roles:
        raise AccessDenied(f"not a member of tenant '{tenant}'")
    return max(roles, key=role_rank)


def authorize(user: dict, tenant: str, permission: str) -> str:
    """Raise :class:`AccessDenied` unless *user* may *permission* on *tenant*; return the role."""
    if user.get("auth") == "demo":
        if permission != "query":
            raise AccessDenied("anonymous demo access is read-only")
        if tenant_role(user, tenant) is not None:  # raises: not a member
            raise AccessDenied(f"not a member of tenant '{tenant}'")
        return "viewer"
    if user.get("auth") == "api_key":
        if not user.get("tenant"):
            raise AccessDenied("this API key is not scoped to a tenant")
        if user["tenant"] != tenant:
            raise AccessDenied(f"this API key is scoped to tenant '{user['tenant']}', not '{tenant}'")
        role = user.get("role", "viewer")
    else:
        role = tenant_role(user, tenant)
        if role is None:
            return "member"   # free-form workspace: no membership model to enforce
    if permission not in PERMISSIONS.get(role, set()):
        raise AccessDenied(f"role '{role}' may not '{permission}' in tenant '{tenant}'")
    return role
