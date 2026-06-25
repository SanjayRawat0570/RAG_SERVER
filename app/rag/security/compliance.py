"""Compliance status report (F25)."""
from __future__ import annotations

from typing import Any


def compliance_report() -> dict[str, Any]:
    """Return a structured overview of the system's compliance posture."""
    return {
        "gdpr": {
            "status":   "enabled",
            "features": [
                "data_inventory",
                "data_export",
                "data_deletion",
                "audit_log",
                "consent_tracking",
            ],
            "right_to_access":      True,
            "right_to_erasure":     True,
            "right_to_portability": True,
            "right_to_rectification": True,
        },
        "encryption": {
            "at_rest":   "SHA-256-CTR stream cipher (256-bit key, HMAC integrity)",
            "in_transit": "HTTPS/TLS enforced by deployment layer",
            "key_management": "Per-request random nonce; keys never persisted with data",
        },
        "authentication": {
            "provider":       "Supabase (JWT, MFA-ready)",
            "session_tokens": "Short-lived, auto-expiry",
            "password_storage": "Hashed (Supabase-managed, bcrypt)",
        },
        "api_keys": {
            "hashing":   "SHA-256 — plaintext never stored",
            "revocable": True,
            "expiry":    "Supported",
            "audit":     "Usage tracked in audit log",
        },
        "audit_log": {
            "enabled":   True,
            "captures":  ["who", "what", "when", "where"],
            "retention": "100,000 events (rolling in-memory; swap to DB for production)",
        },
        "access_control": {
            "model": "User-scoped data isolation (row-level security via Supabase)",
            "roles": ["user", "admin", "support"],
        },
        "standards": {
            "gdpr":    "Core requirements met (export, deletion, audit, consent)",
            "soc2":    "Partial (encryption, access control, audit log)",
            "hipaa":   "Not certified — add BAA, additional audit controls",
            "pci_dss": "Not applicable (no payment processing)",
        },
    }
