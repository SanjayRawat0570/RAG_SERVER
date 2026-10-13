"""Conversation memory API — the chat sidebar.

GET    /conversations?tenant=acme            The caller's chats in a tenant (newest first)
POST   /conversations?tenant=acme            Start a new chat
GET    /conversations/{id}?tenant=acme       One chat with its stored turns (last 5)
DELETE /conversations/{id}?tenant=acme       Clear (delete) one chat
DELETE /conversations?tenant=acme            Clear all of the caller's chats in a tenant

Chats are scoped to the authenticated user and tenant: another user's or
another tenant's conversation id simply returns 404.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.api.deps import CurrentUser
from app.rag.memory import store

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(user: CurrentUser, tenant: str = Query("default")) -> dict[str, Any]:
    convs = store.list_conversations(tenant, user["id"])
    return {"tenant": tenant, "total": len(convs), "conversations": [c.summary() for c in convs]}


@router.post("", status_code=201)
async def new_conversation(user: CurrentUser, tenant: str = Query("default")) -> dict[str, Any]:
    return store.create(tenant, user["id"]).summary()


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str, user: CurrentUser,
                           tenant: str = Query("default")) -> dict[str, Any]:
    conv = store.get(tenant, user["id"], conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return conv.to_dict()


@router.delete("/{conversation_id}")
async def clear_conversation(conversation_id: str, user: CurrentUser,
                             tenant: str = Query("default")) -> dict[str, Any]:
    if not store.delete(tenant, user["id"], conversation_id):
        raise HTTPException(404, "Conversation not found")
    return {"status": "deleted", "conversation_id": conversation_id}


@router.delete("")
async def clear_all(user: CurrentUser, tenant: str = Query("default")) -> dict[str, Any]:
    convs = store.list_conversations(tenant, user["id"])
    for c in convs:
        store.delete(tenant, user["id"], c.id)
    return {"status": "deleted", "count": len(convs)}
