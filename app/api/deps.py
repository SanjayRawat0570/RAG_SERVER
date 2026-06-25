"""Shared FastAPI dependencies — reusable across all routes."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import settings

_sb = None


def _client():
    global _sb
    if _sb is not None:
        return _sb
    if settings.supabase_url and settings.supabase_key:
        from supabase import create_client  # type: ignore[import]
        _sb = create_client(settings.supabase_url, settings.supabase_key)
    return _sb


async def get_current_user(
    authorization: Annotated[str, Header(alias="authorization")] = "",
) -> dict:
    """Verify the Supabase JWT and return the user dict.

    Raises 401 if the token is missing or invalid.
    Falls back to a dev user when:
      - Supabase is not configured, OR
      - APP_ENV is development/test and the token is literally "dev"
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid authorization header")

    token = authorization[7:].strip()

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
) -> dict:
    """Like get_current_user but never raises — anonymous fallback on any error."""
    try:
        return await get_current_user(authorization)
    except HTTPException:
        return {"id": "anonymous", "email": "", "username": "anonymous"}
