"""Request-scoped tenant authorization for data access (F18/F25).

The API's auth dependency records the caller (the *principal*) for the current
request. Every place that reads or writes tenant data then checks the
namespace it touches against that principal:

* route handlers call :func:`authorize_namespace` (via ``deps.require_tenant``)
  before doing anything, so a forbidden request fails fast with 403;
* the vector stores call :func:`guard` on every read and write, so a namespace
  that reaches them some other way — a user-supplied workflow on
  ``/workflows/run``, an agent tool, a legacy endpoint — is still refused.

Code that runs without a principal (tests, eval scripts, startup) is trusted.

Namespaces map to tenants as the tenant registry builds them: ``org`` and
``org/team`` belong to tenant ``org``; ``user:<id>`` is private to that user.
Stores whose name starts with ``__`` are internal (e.g. the semantic cache's
``__semcache__…``) and are not tenant data.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from app.rag.security.rbac import AccessDenied, authorize

_principal: ContextVar[dict[str, Any] | None] = ContextVar("principal", default=None)

USER_PREFIX = "user:"


def set_principal(user: dict[str, Any] | None) -> None:
    _principal.set(user)


def current_principal() -> dict[str, Any] | None:
    return _principal.get()


def namespace_tenant(namespace: str) -> str:
    """The tenant that owns *namespace* (``org/team`` → ``org``)."""
    return (namespace or "default").split("/", 1)[0]


def authorize_namespace(user: dict[str, Any], namespace: str, permission: str = "query") -> str:
    """Raise :class:`AccessDenied` unless *user* may *permission* on *namespace*; return the role."""
    namespace = namespace or "default"
    if namespace.startswith(USER_PREFIX):
        # Personal namespace: only its owner, and never through a tenant-scoped API key.
        if user.get("auth") == "api_key" or namespace[len(USER_PREFIX):] != str(user.get("id")):
            raise AccessDenied(f"namespace '{namespace}' is private to another user")
        return "owner"
    return authorize(user, namespace_tenant(namespace), permission)


def guard(namespace: str, permission: str = "query", store: str = "") -> None:
    """Data-layer check: refuse *namespace* unless the current principal may use it."""
    if store.startswith("__"):
        return
    user = _principal.get()
    if user is not None:
        authorize_namespace(user, namespace, permission)
