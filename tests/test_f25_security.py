"""Tests for F25: Security, Privacy & Compliance."""
from __future__ import annotations

import pytest

from app.rag.security import (
    AuditAction, AuditEvent,
    compliance_report,
    create_api_key, data_inventory, decrypt, delete_user_data,
    encrypt, export_user_data, generate_key, get_events,
    list_api_keys, log_event, reset_api_keys, reset_audit,
    revoke_api_key, verify_api_key,
)
from app.rag.personalization.store import reset_store


@pytest.fixture(autouse=True)
def _clean():
    reset_audit()
    reset_api_keys()
    reset_store()
    yield
    reset_audit()
    reset_api_keys()
    reset_store()


# ── Audit logging ─────────────────────────────────────────────────────────────

def test_f25_audit_log_event():
    event = log_event("alice", AuditAction.UPLOAD, resource="doc.pdf")
    assert isinstance(event, AuditEvent)
    assert event.user_id  == "alice"
    assert event.action   == AuditAction.UPLOAD
    assert event.resource == "doc.pdf"
    assert event.id       != ""


def test_f25_audit_get_by_user():
    log_event("alice", AuditAction.QUERY,  resource="q1")
    log_event("bob",   AuditAction.UPLOAD, resource="f.pdf")
    events = get_events(user_id="alice")
    assert all(e.user_id == "alice" for e in events)
    assert len(events) == 1


def test_f25_audit_get_by_action():
    log_event("alice", AuditAction.QUERY,  resource="q1")
    log_event("alice", AuditAction.UPLOAD, resource="f.pdf")
    events = get_events(action=AuditAction.QUERY)
    assert all(e.action == AuditAction.QUERY for e in events)


def test_f25_audit_most_recent_first():
    log_event("alice", AuditAction.QUERY,  resource="first")
    log_event("alice", AuditAction.UPLOAD, resource="second")
    events = get_events(user_id="alice")
    assert events[0].resource == "second"


def test_f25_audit_empty():
    assert get_events() == []


def test_f25_audit_metadata_stored():
    log_event("alice", AuditAction.DATA_ACCESS, metadata={"reason": "support"})
    events = get_events(user_id="alice")
    assert events[0].metadata["reason"] == "support"


# ── Encryption ────────────────────────────────────────────────────────────────

def test_f25_generate_key_256_bits():
    key = generate_key()
    assert len(key) == 64        # 32 bytes = 64 hex chars
    assert all(c in "0123456789abcdef" for c in key)


def test_f25_generate_key_unique():
    k1, k2 = generate_key(), generate_key()
    assert k1 != k2


def test_f25_encrypt_differs_from_plaintext():
    key = generate_key()
    ct  = encrypt("hello world", key)
    assert ct != "hello world"


def test_f25_decrypt_recovers_plaintext():
    key = generate_key()
    ct  = encrypt("secret message", key)
    assert decrypt(ct, key) == "secret message"


def test_f25_encrypt_same_text_different_ciphertext():
    key = generate_key()
    ct1 = encrypt("repeated", key)
    ct2 = encrypt("repeated", key)
    assert ct1 != ct2   # different nonce each time


def test_f25_wrong_key_raises_decryption_error():
    key       = generate_key()
    wrong_key = generate_key()
    ct        = encrypt("secret", key)
    with pytest.raises(ValueError, match="Decryption failed"):
        decrypt(ct, wrong_key)


def test_f25_encrypt_empty_string():
    key = generate_key()
    ct  = encrypt("", key)
    assert decrypt(ct, key) == ""


def test_f25_encrypt_unicode():
    key  = generate_key()
    text = "Héllo wörld — 中文 🔒"
    assert decrypt(encrypt(text, key), key) == text


# ── API key management ────────────────────────────────────────────────────────

def test_f25_create_api_key_returns_plaintext():
    record, plaintext = create_api_key("alice", "my-key")
    assert plaintext.startswith("sk-")
    assert len(plaintext) > 10


def test_f25_api_key_hash_not_equal_to_plaintext():
    record, plaintext = create_api_key("alice", "k1")
    assert record.key_hash != plaintext
    assert len(record.key_hash) == 64   # SHA-256 hex


def test_f25_verify_api_key_valid():
    record, plaintext = create_api_key("alice", "k1")
    found = verify_api_key(plaintext)
    assert found is not None
    assert found.id == record.id


def test_f25_verify_api_key_invalid():
    assert verify_api_key("sk-thisisnotavalidkey") is None


def test_f25_verify_api_key_increments_usage():
    _, plaintext = create_api_key("alice", "k1")
    verify_api_key(plaintext)
    verify_api_key(plaintext)
    found = verify_api_key(plaintext)
    assert found.usage_count == 3


def test_f25_revoke_api_key():
    record, plaintext = create_api_key("alice", "k1")
    assert revoke_api_key(record.id, "alice") is True
    assert verify_api_key(plaintext) is None


def test_f25_revoke_wrong_user_returns_false():
    record, _ = create_api_key("alice", "k1")
    assert revoke_api_key(record.id, "bob") is False


def test_f25_list_api_keys_for_user():
    create_api_key("alice", "key-a")
    create_api_key("alice", "key-b")
    create_api_key("bob",   "key-c")
    alice_keys = list_api_keys("alice")
    assert len(alice_keys) == 2
    assert all(k.user_id == "alice" for k in alice_keys)


# ── GDPR ─────────────────────────────────────────────────────────────────────

def test_f25_data_inventory_structure():
    inv = data_inventory("alice")
    assert inv["user_id"]        == "alice"
    assert "data_categories"     in inv
    assert "retention_policy"    in inv
    assert "legal_basis"         in inv


def test_f25_export_user_data_sections():
    export = export_user_data("alice")
    assert export["user_id"]      == "alice"
    assert "exported_at"          in export
    assert "profile"              in export
    assert "query_history"        in export
    assert "feedback"             in export
    assert "audit_log"            in export
    assert "api_keys"             in export


def test_f25_delete_user_data_returns_summary():
    create_api_key("alice", "k1")
    result = delete_user_data("alice")
    assert result["status"]  == "deleted"
    assert result["user_id"] == "alice"
    assert result["deleted"]["api_keys"] == 1


def test_f25_delete_user_data_removes_api_keys():
    create_api_key("alice", "k1")
    delete_user_data("alice")
    assert list_api_keys("alice") == []


# ── Compliance report ─────────────────────────────────────────────────────────

def test_f25_compliance_report_structure():
    report = compliance_report()
    assert "gdpr"            in report
    assert "encryption"      in report
    assert "authentication"  in report
    assert "audit_log"       in report
    assert "api_keys"        in report
    assert "standards"       in report


def test_f25_compliance_gdpr_features():
    report = compliance_report()
    assert report["gdpr"]["right_to_access"]      is True
    assert report["gdpr"]["right_to_erasure"]     is True
    assert report["gdpr"]["right_to_portability"] is True


# ── API ───────────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f25_api_log_audit_event():
    with _client() as c:
        resp = c.post("/api/v1/security/audit/events", json={
            "action": "upload", "resource": "doc.pdf",
        }, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["action"]  == "upload"
    assert "id"            in data
    assert "timestamp"     in data


def test_f25_api_list_audit_events():
    with _client() as c:
        c.post("/api/v1/security/audit/events",
               json={"action": "query", "resource": "q1"}, headers=AUTH)
        c.post("/api/v1/security/audit/events",
               json={"action": "upload", "resource": "f.pdf"}, headers=AUTH)
        resp = c.get("/api/v1/security/audit/events", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2


def test_f25_api_encrypt_generates_key():
    with _client() as c:
        resp = c.post("/api/v1/security/encrypt",
                      json={"plaintext": "hello"}, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "ciphertext" in data
    assert "key"        in data
    assert data["key_bits"] == 256


def test_f25_api_decrypt_roundtrip():
    with _client() as c:
        enc = c.post("/api/v1/security/encrypt",
                     json={"plaintext": "round trip test"}, headers=AUTH).json()
        resp = c.post("/api/v1/security/decrypt", json={
            "ciphertext": enc["ciphertext"], "key": enc["key"],
        }, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["plaintext"] == "round trip test"


def test_f25_api_decrypt_wrong_key_returns_400():
    with _client() as c:
        enc = c.post("/api/v1/security/encrypt",
                     json={"plaintext": "secret"}, headers=AUTH).json()
        wrong = generate_key()
        resp = c.post("/api/v1/security/decrypt",
                      json={"ciphertext": enc["ciphertext"], "key": wrong}, headers=AUTH)
    assert resp.status_code == 400


def test_f25_api_compliance():
    with _client() as c:
        resp = c.get("/api/v1/security/compliance", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "gdpr"       in data
    assert "encryption" in data
    assert "standards"  in data


def test_f25_api_gdpr_data_inventory():
    with _client() as c:
        resp = c.get("/api/v1/security/gdpr/data", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "data_categories" in data
    assert "legal_basis"     in data


def test_f25_api_gdpr_export():
    with _client() as c:
        resp = c.get("/api/v1/security/gdpr/export", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "profile"       in data
    assert "query_history" in data
    assert "exported_at"   in data


def test_f25_api_gdpr_delete():
    with _client() as c:
        resp = c.delete("/api/v1/security/gdpr/data", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_f25_api_create_api_key():
    with _client() as c:
        resp = c.post("/api/v1/security/api-keys",
                      json={"name": "my-key"}, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["key"].startswith("sk-")
    assert "warning" in data


def test_f25_api_list_api_keys():
    with _client() as c:
        c.post("/api/v1/security/api-keys", json={"name": "k1"}, headers=AUTH)
        c.post("/api/v1/security/api-keys", json={"name": "k2"}, headers=AUTH)
        resp = c.get("/api/v1/security/api-keys", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] >= 2


def test_f25_api_revoke_api_key():
    with _client() as c:
        key_data = c.post("/api/v1/security/api-keys",
                          json={"name": "k1"}, headers=AUTH).json()
        resp = c.delete(f"/api/v1/security/api-keys/{key_data['id']}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "revoked"


def test_f25_api_verify_key_valid():
    with _client() as c:
        key_data = c.post("/api/v1/security/api-keys",
                          json={"name": "k1"}, headers=AUTH).json()
        resp = c.post("/api/v1/security/api-keys/verify",
                      json={"key": key_data["key"]}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["valid"] is True


def test_f25_api_verify_key_invalid():
    with _client() as c:
        resp = c.post("/api/v1/security/api-keys/verify",
                      json={"key": "sk-notavalidkey"}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["valid"] is False


def test_f25_api_reset():
    with _client() as c:
        c.post("/api/v1/security/audit/events",
               json={"action": "test"}, headers=AUTH)
        resp = c.delete("/api/v1/security/reset", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_f25_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/security/compliance")
    assert resp.status_code == 401
