"""Auth endpoints — thin wrapper around Supabase Auth.

Registration and login are handled directly by the Supabase JS client in the
browser.  The backend only exposes /me to let the frontend validate its JWT.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import CurrentUser

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
async def me(user: CurrentUser) -> dict:
    """Return the authenticated user's profile."""
    return user
