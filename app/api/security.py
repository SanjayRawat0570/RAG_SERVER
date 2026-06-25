"""Security, Privacy & Compliance API (F25).

Endpoints
---------
POST   /security/audit/events          Log an audit event
GET    /security/audit/events          List audit events (filterable)
POST   /security/encrypt               Encrypt a string with a 256-bit key
POST   /security/decrypt               Decrypt a token produced by /encrypt
GET    /security/compliance            Compliance status report
GET    /security/gdpr/data             Data inventory for current user
GET    /security/gdpr/export           Export all user data (GDPR portability)
DELETE /security/gdpr/data             Erase all user data (right to erasure)
POST   /security/api-keys              Create API key (plaintext returned once)
GET    /security/api-keys              List current user's API keys
DELETE /security/api-keys/{key_id}     Revoke an API key
POST   /security/api-keys/verify       Verify a plaintext API key
DELETE /security/reset                 Wipe audit log + API keys (test helper)
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query as QParam
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.security import (
    AuditAction, compliance_report,
    create_api_key, data_inventory, decrypt, delete_user_data,
    encrypt, export_user_data, generate_key, get_events,
    list_api_keys, log_event, reset_api_keys, reset_audit,
    revoke_api_key, verify_api_key,
)

router = APIRouter(prefix="/security", tags=["security"])


# ── Request models ─────────────────────────────────────────────────────────────

class LogEventRequest(BaseModel):
    action:     str = Field(..., min_length=1)
    resource:   str = ""
    ip_address: str = ""
    tenant:     str = "default"
    metadata:   dict[str, Any] = Field(default_factory=dict)


class EncryptRequest(BaseModel):
    plaintext: str = Field(..., min_length=1)
    key:       str | None = None   # auto-generate if omitted


class DecryptRequest(BaseModel):
    ciphertext: str = Field(..., min_length=1)
    key:        str = Field(..., min_length=1)


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1)


class VerifyKeyRequest(BaseModel):
    key: str = Field(..., min_length=1)


# ── Audit log ──────────────────────────────────────────────────────────────────

@router.post("/audit/events", status_code=201)
async def audit_log_event(req: LogEventRequest, user: CurrentUser) -> dict[str, Any]:
    event = log_event(
        user_id=user["id"],
        action=req.action,
        resource=req.resource,
        ip_address=req.ip_address,
        tenant=req.tenant,
        metadata=req.metadata,
    )
    return {
        "id":        event.id,
        "user_id":   event.user_id,
        "action":    event.action,
        "resource":  event.resource,
        "timestamp": event.timestamp.isoformat(),
    }


@router.get("/audit/events")
async def list_audit_events(
    _:       CurrentUser,
    user_id: str | None = QParam(None),
    action:  str | None = QParam(None),
    tenant:  str | None = QParam(None),
    limit:   int        = QParam(50, ge=1, le=500),
) -> dict[str, Any]:
    events = get_events(user_id=user_id, action=action, tenant=tenant, limit=limit)
    return {
        "total": len(events),
        "events": [
            {
                "id":         e.id,
                "user_id":    e.user_id,
                "action":     e.action,
                "resource":   e.resource,
                "ip_address": e.ip_address,
                "tenant":     e.tenant,
                "timestamp":  e.timestamp.isoformat(),
            }
            for e in events
        ],
    }


# ── Encryption ─────────────────────────────────────────────────────────────────

@router.post("/encrypt")
async def encrypt_data(req: EncryptRequest, user: CurrentUser) -> dict[str, Any]:
    key = req.key or generate_key()
    token = encrypt(req.plaintext, key)
    log_event(user["id"], AuditAction.ENCRYPT, resource="plaintext")
    return {"ciphertext": token, "key": key, "key_bits": 256}


@router.post("/decrypt")
async def decrypt_data(req: DecryptRequest, user: CurrentUser) -> dict[str, Any]:
    try:
        plaintext = decrypt(req.ciphertext, req.key)
    except (ValueError, Exception) as exc:
        raise HTTPException(400, f"Decryption failed: {exc}") from exc
    log_event(user["id"], AuditAction.DECRYPT, resource="ciphertext")
    return {"plaintext": plaintext}


# ── Compliance ─────────────────────────────────────────────────────────────────

@router.get("/compliance")
async def get_compliance(_: CurrentUser) -> dict[str, Any]:
    return compliance_report()


# ── GDPR ───────────────────────────────────────────────────────────────────────

@router.get("/gdpr/data")
async def gdpr_inventory(user: CurrentUser) -> dict[str, Any]:
    log_event(user["id"], AuditAction.PROFILE_VIEW, resource="data_inventory")
    return data_inventory(user["id"])


@router.get("/gdpr/export")
async def gdpr_export(user: CurrentUser) -> dict[str, Any]:
    log_event(user["id"], AuditAction.GDPR_EXPORT, resource="all_data")
    return export_user_data(user["id"])


@router.delete("/gdpr/data")
async def gdpr_delete(user: CurrentUser) -> dict[str, Any]:
    log_event(user["id"], AuditAction.GDPR_DELETE, resource="all_data")
    return delete_user_data(user["id"])


# ── API keys ───────────────────────────────────────────────────────────────────

@router.post("/api-keys", status_code=201)
async def create_key(req: CreateKeyRequest, user: CurrentUser) -> dict[str, Any]:
    key_record, plaintext = create_api_key(user["id"], req.name)
    log_event(user["id"], AuditAction.API_KEY_CREATE, resource=key_record.id)
    return {
        "id":         key_record.id,
        "name":       key_record.name,
        "prefix":     key_record.prefix,
        "key":        plaintext,           # shown exactly once
        "created_at": key_record.created_at.isoformat(),
        "warning":    "Store this key now — it won't be shown again.",
    }


@router.get("/api-keys")
async def list_keys(user: CurrentUser) -> dict[str, Any]:
    keys = list_api_keys(user["id"])
    return {
        "total": len(keys),
        "keys": [
            {
                "id":          k.id,
                "name":        k.name,
                "prefix":      k.prefix,
                "revoked":     k.revoked,
                "usage_count": k.usage_count,
                "created_at":  k.created_at.isoformat(),
                "last_used":   k.last_used.isoformat() if k.last_used else None,
            }
            for k in keys
        ],
    }


@router.delete("/api-keys/{key_id}")
async def revoke_key(key_id: str, user: CurrentUser) -> dict[str, Any]:
    success = revoke_api_key(key_id, user["id"])
    if not success:
        raise HTTPException(404, f"API key '{key_id}' not found or not owned by you")
    log_event(user["id"], AuditAction.API_KEY_REVOKE, resource=key_id)
    return {"status": "revoked", "key_id": key_id}


@router.post("/api-keys/verify")
async def verify_key(req: VerifyKeyRequest, _: CurrentUser) -> dict[str, Any]:
    key = verify_api_key(req.key)
    if key is None:
        return {"valid": False}
    return {
        "valid":       True,
        "key_id":      key.id,
        "user_id":     key.user_id,
        "name":        key.name,
        "usage_count": key.usage_count,
    }


# ── Reset (test helper) ────────────────────────────────────────────────────────

@router.delete("/reset")
async def reset(_: CurrentUser) -> dict[str, Any]:
    reset_audit()
    reset_api_keys()
    return {"status": "ok"}
