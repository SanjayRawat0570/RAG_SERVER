"""Shared FastAPI dependencies — reusable across all routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import settings

_sb = None
API_KEY_PREFIX = "sk-"


def _client():
    global _sb
    if _sb is not None:
        return _sb
    if settings.supabase_url and settings.supabase_key:
        from supabase import create_client  # type: ignore[import]
        _sb = create_client(settings.supabase_url, settings.supabase_key)
    return _sb


def _api_key_user(key: str) -> dict:
    """Resolve an API key to a caller scoped to the key's tenant and role."""
    from app.rag.security.apikeys import verify_api_key

    record = verify_api_key(key)
    if record is None:
        raise HTTPException(401, "Invalid, revoked or expired API key")
    return {
        "id":         record.user_id,
        "email":      "",
        "username":   f"api-key:{record.name}",
        "auth":       "api_key",
        "api_key_id": record.id,
        "tenant":     record.tenant,
        "role":       record.role,
    }


async def get_current_user(
    authorization: Annotated[str, Header(alias="authorization")] = "",
    x_api_key: Annotated[str, Header(alias="x-api-key")] = "",
) -> dict:
    """Verify the caller and return the user dict.

    Accepts a Supabase JWT (``Authorization: Bearer <jwt>``) or an API key
    (``Authorization: Bearer sk-…`` or ``X-API-Key: sk-…``). API keys are always
    verified, even in development.

    Raises 401 if credentials are missing or invalid.
    Falls back to a dev user when:
      - Supabase is not configured, OR
      - APP_ENV is development/test and the token is literally "dev"
    """
    if x_api_key:
        return _api_key_user(x_api_key.strip())
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")

    token = authorization[7:].strip()
    if token.startswith(API_KEY_PREFIX):
        return _api_key_user(token)

    # Dev bypass: accept literal token "dev" in non-production environments.
    if token == "dev" and settings.app_env in ("development", "test"):
        return {"id": "dev", "email": "dev@local", "username": "Dev User"}

    sb = _client()
    if not sb:
        # No Supabase configured — accept any bearer token as dev user.
        return {"id": "dev", "email": "dev@local", "username": "Dev User"}

    try:
        resp = sb.auth.get_user(token)
        u    = resp.user
        return {
            "id":       str(u.id),
            "email":    u.email,
            "username": (u.user_metadata or {}).get("username", u.email),
        }
    except Exception:
        raise HTTPException(401, "Invalid or expired token")


# Convenience type alias — import this in route files.
CurrentUser = Annotated[dict, Depends(get_current_user)]


async def get_optional_user(
    authorization: Annotated[str, Header(alias="authorization")] = "",
    x_api_key: Annotated[str, Header(alias="x-api-key")] = "",
) -> dict:
    """Like get_current_user but anonymous when no credentials are sent.

    A presented-but-invalid API key is still rejected (401) rather than silently
    downgraded to anonymous.
    """
    try:
        return await get_current_user(authorization, x_api_key)
    except HTTPException:
        if x_api_key or authorization[7:].strip().startswith(API_KEY_PREFIX):
            raise
        return {"id": "anonymous", "email": "", "username": "anonymous"}


def require_tenant(user: dict, tenant: str, permission: str = "query") -> str:
    """RBAC gate for tenant data: 403 unless *user* may *permission* on *tenant*."""
    from app.rag.security.rbac import AccessDenied, authorize

    try:
        return authorize(user, tenant, permission)
    except AccessDenied as exc:
        raise HTTPException(403, str(exc)) from exc
