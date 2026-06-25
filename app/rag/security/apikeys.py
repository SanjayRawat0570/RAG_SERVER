"""API key management (F25).

Keys are stored hashed (SHA-256); plaintext is returned exactly once at
creation and never persisted.  Keys can be revoked and optionally expire.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_store: dict[str, APIKey] = {}   # id → APIKey


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


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def create_api_key(
    user_id:    str,
    name:       str,
    expires_at: datetime | None = None,
) -> tuple[APIKey, str]:
    """Create an API key.  Returns *(record, plaintext)* — plaintext shown once."""
    plaintext = "sk-" + secrets.token_hex(24)
    key = APIKey(
        id=secrets.token_hex(8),
        user_id=user_id,
        name=name,
        key_hash=_hash(plaintext),
        prefix=plaintext[:8],
        expires_at=expires_at,
    )
    _store[key.id] = key
    return key, plaintext


def verify_api_key(plaintext_key: str) -> APIKey | None:
    """Return the key record if *plaintext_key* is valid, else None."""
    h = _hash(plaintext_key)
    for key in _store.values():
        if key.key_hash != h:
            continue
        if key.revoked:
            return None
        if key.expires_at and datetime.now(timezone.utc) > key.expires_at:
            return None
        key.last_used   = datetime.now(timezone.utc)
        key.usage_count += 1
        return key
    return None


def revoke_api_key(key_id: str, user_id: str) -> bool:
    key = _store.get(key_id)
    if key is None or key.user_id != user_id:
        return False
    key.revoked = True
    return True


def list_api_keys(user_id: str) -> list[APIKey]:
    return [k for k in _store.values() if k.user_id == user_id]


def delete_api_key(key_id: str) -> bool:
    return _store.pop(key_id, None) is not None


def reset_api_keys() -> None:
    _store.clear()
