"""Audit logging (F25) — WHO / WHAT / WHEN / WHERE / WHY."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

_MAX_EVENTS = 100_000
_events: deque[AuditEvent] = deque(maxlen=_MAX_EVENTS)


class AuditAction:
    UPLOAD          = "upload"
    QUERY           = "query"
    DELETE          = "delete"
    EXPORT          = "export"
    LOGIN           = "login"
    LOGOUT          = "logout"
    API_KEY_CREATE  = "api_key_create"
    API_KEY_REVOKE  = "api_key_revoke"
    PROFILE_VIEW    = "profile_view"
    ADMIN_VIEW      = "admin_view"
    DATA_ACCESS     = "data_access"
    ENCRYPT         = "encrypt"
    DECRYPT         = "decrypt"
    GDPR_EXPORT     = "gdpr_export"
    GDPR_DELETE     = "gdpr_delete"


@dataclass
class AuditEvent:
    id:         str
    user_id:    str
    action:     str
    resource:   str
    ip_address: str  = ""
    tenant:     str  = "default"
    metadata:   dict = field(default_factory=dict)
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def log_event(
    user_id:    str,
    action:     str,
    resource:   str  = "",
    ip_address: str  = "",
    tenant:     str  = "default",
    metadata:   dict | None = None,
) -> AuditEvent:
    event = AuditEvent(
        id=str(uuid4()),
        user_id=user_id,
        action=action,
        resource=resource,
        ip_address=ip_address,
        tenant=tenant,
        metadata=metadata or {},
    )
    _events.appendleft(event)
    return event


def get_events(
    user_id: str | None = None,
    action:  str | None = None,
    tenant:  str | None = None,
    limit:   int = 100,
) -> list[AuditEvent]:
    results: list[AuditEvent] = list(_events)
    if user_id is not None: results = [e for e in results if e.user_id == user_id]
    if action  is not None: results = [e for e in results if e.action  == action]
    if tenant  is not None: results = [e for e in results if e.tenant  == tenant]
    return results[:limit]


def reset_audit() -> None:
    _events.clear()
