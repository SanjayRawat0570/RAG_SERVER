"""At-rest encryption (F25).

Uses a SHA-256-CTR stream cipher (offline, no external library required):
  - 256-bit random key (hex-encoded)
  - 16-byte random nonce per message — so same plaintext → different ciphertext
  - 8-byte HMAC appended for integrity; wrong key raises ValueError

This is the deterministic offline implementation. When the `cryptography`
library is available, swap to AES-256-GCM for industry-standard security.
"""
from __future__ import annotations

import base64
import hashlib
import secrets


def generate_key() -> str:
    """Return a 256-bit (32-byte) random key as a 64-char hex string."""
    return secrets.token_hex(32)


def _to_key_bytes(key: str) -> bytes:
    if len(key) == 64:                          # hex-encoded 32-byte key
        return bytes.fromhex(key)
    return hashlib.sha256(key.encode()).digest()  # hash arbitrary string to 32 bytes


def _keystream(key_bytes: bytes, nonce: bytes, length: int) -> bytes:
    """SHA-256-CTR keystream: deterministic, key+nonce-dependent."""
    stream = b""
    block  = 0
    while len(stream) < length:
        h = hashlib.sha256(key_bytes + nonce + block.to_bytes(8, "big")).digest()
        stream += h
        block  += 1
    return stream[:length]


def encrypt(plaintext: str, key: str) -> str:
    """Encrypt *plaintext* and return a base64-encoded token (nonce+ct+mac)."""
    data      = plaintext.encode("utf-8")
    nonce     = secrets.token_bytes(16)
    key_bytes = _to_key_bytes(key)
    ks        = _keystream(key_bytes, nonce, len(data))
    ct        = bytes(a ^ b for a, b in zip(data, ks))
    mac       = hashlib.sha256(key_bytes + nonce + ct).digest()[:8]
    return base64.b64encode(nonce + ct + mac).decode()


def decrypt(ciphertext_b64: str, key: str) -> str:
    """Decrypt a token produced by :func:`encrypt`.

    Raises :class:`ValueError` if the key is wrong or data is corrupted.
    """
    raw           = base64.b64decode(ciphertext_b64)
    nonce         = raw[:16]
    ct            = raw[16:-8]
    mac_stored    = raw[-8:]
    key_bytes     = _to_key_bytes(key)
    mac_expected  = hashlib.sha256(key_bytes + nonce + ct).digest()[:8]
    if mac_stored != mac_expected:
        raise ValueError("Decryption failed: invalid key or corrupted data")
    ks = _keystream(key_bytes, nonce, len(ct))
    return bytes(a ^ b for a, b in zip(ct, ks)).decode("utf-8")
