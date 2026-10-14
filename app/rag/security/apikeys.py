"""API key management (F25).

Keys are stored hashed (SHA-256); plaintext is returned exactly once at
creation and never persisted.  Keys can be revoked and optionally expire.

A key may be scoped to a **tenant** with a **role** (viewer | member | admin |
owner). A scoped key can only act on its own tenant, with that role's
permissions — the same RBAC the REST API applies (see ``rbac.py``). The MCP
server authenticates with such a key.

The (hashed) keys are mirrored to ``<data_dir>/api_keys.json`` so a key
created in the web UI also works for the separately started MCP server
process. The file is re-read when another process changes it.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.rag.persist import data_file, load_json, save_json

_FILE = "api_keys.json"
_store: dict[str, APIKey] = {}   # id → APIKey
_mtime: float | None = None


@dataclass
class APIKey:
    id:          str
    user_id:     str
    name:        str
    key_hash:    str            # SHA-256(plaintext) — never show this
    prefix:      str            # first 8 chars of plaintext key for identification
    created_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at:  datetime | None = None
    revoked:     bool = False
    last_used:   datetime | None = None
    usage_count: int  = 0
    tenant:      str | None = None   # None → not scoped to any tenant's data
    role:        str = "member"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _from_json(raw: dict[str, Any]) -> APIKey:
    for f in ("created_at", "expires_at", "last_used"):
        if raw.get(f):
            raw[f] = datetime.fromisoformat(raw[f])
    return APIKey(**raw)


def _sync() -> None:
    """Reload from disk when another process wrote the file."""
    global _store, _mtime
    path = data_file(_FILE)
    mtime = path.stat().st_mtime if path.exists() else None
    if mtime != _mtime:
        _store = {k["id"]: _from_json(dict(k)) for k in load_json(_FILE, [])}
        _mtime = mtime


def _save() -> None:
    global _mtime
    save_json(_FILE, [
        {**asdict(k), **{f: getattr(k, f).isoformat() if getattr(k, f) else None
                         for f in ("created_at", "expires_at", "last_used")}}
        for k in _store.values()
    ])
    _mtime = data_file(_FILE).stat().st_mtime


def create_api_key(
    user_id:    str,
    name:       str,
    expires_at: datetime | None = None,
    *,
    tenant:     str | None = None,
    role:       str = "member",
) -> tuple[APIKey, str]:
    """Create an API key.  Returns *(record, plaintext)* — plaintext shown once."""
    _sync()
    plaintext = "sk-" + secrets.token_hex(24)
    key = APIKey(
        id=secrets.token_hex(8),
        user_id=user_id,
        name=name,
        key_hash=_hash(plaintext),
        prefix=plaintext[:8],
        expires_at=expires_at,
        tenant=tenant,
        role=role,
    )
    _store[key.id] = key
    _save()
    return key, plaintext


def verify_api_key(plaintext_key: str) -> APIKey | None:
    """Return the key record if *plaintext_key* is valid, else None."""
    _sync()
    h = _hash(plaintext_key)
    for key in _store.values():
        if not secrets.compare_digest(key.key_hash, h):
            continue
        if key.revoked:
            return None
        if key.expires_at and datetime.now(timezone.utc) > key.expires_at:
            return None
        key.last_used   = datetime.now(timezone.utc)
        key.usage_count += 1
        _save()
        return key
    return None


def revoke_api_key(key_id: str, user_id: str) -> bool:
    _sync()
    key = _store.get(key_id)
    if key is None or key.user_id != user_id:
        return False
    key.revoked = True
    _save()
    return True


def list_api_keys(user_id: str) -> list[APIKey]:
    _sync()
    return [k for k in _store.values() if k.user_id == user_id]


def delete_api_key(key_id: str) -> bool:
    _sync()
    removed = _store.pop(key_id, None) is not None
    if removed:
        _save()
    return removed


def reset_api_keys() -> None:
    _sync()
    _store.clear()
    _save()
