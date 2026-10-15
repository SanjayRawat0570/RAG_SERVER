"""Cross-tenant drill against a live API. usage: python cross_tenant_drill.py BASE_URL

Prints one JSON row per attack with the expected status and PASS/FAIL, then a
summary line. Expected: 403 for an API key used on another tenant, 401 for no
credentials, and the secret never in a response body.
"""
import json
import sys

import httpx

BASE = sys.argv[1]
c = httpx.Client(base_url=BASE + "/api/v1", timeout=60)
DEV = {"Authorization": "Bearer dev"}
SECRET = "The Falcon acquisition budget is 42 million euros, code word NIGHTJAR."

# Victim: registered organisation "secure" (membership enforced), owned by dev.
r = c.post("/tenants", headers=DEV, json={"slug": "secure", "name": "Secure Corp"})
print("create org secure:", r.status_code)
r = c.post("/rag/index", headers=DEV, json={"tenant": "secure", "filename": "falcon.md",
                                            "text": "# Falcon\n\n" + SECRET})
print("index secret into secure:", r.status_code, r.json().get("upserted"))
dim = c.get("/vectorstores", headers=DEV).json()["stores"]["kb"]["dimension"]

# Attacker: an API key scoped to tenant "acme", role viewer.
r = c.post("/security/api-keys", headers=DEV, json={"name": "attacker", "tenant": "acme", "role": "viewer"})
body = r.json()
key = body.get("key") or body.get("api_key") or body.get("secret")
print("create acme viewer key:", r.status_code, "key returned:", bool(key))
ATT = {"X-API-Key": key}

rows = []


def attempt(name, method, path, headers, payload=None, expect=403, params=None):
    r = c.request(method, path, headers=headers, json=payload, params=params)
    leaked = "NIGHTJAR" in r.text or "42 million" in r.text
    ok = r.status_code == expect and not leaked
    rows.append(ok)
    print(json.dumps({"attack": name, "status": r.status_code, "expected": expect,
                      "secret_leaked": leaked, "result": "PASS" if ok else "FAIL",
                      "body": r.text[:120]}))


q = "What is the Falcon acquisition budget and code word?"
kw = {"query": "Falcon budget code word", "store": "kb", "namespace": "secure"}
attempt("acme key -> /rag/ask tenant=secure", "POST", "/rag/ask", ATT, {"tenant": "secure", "question": q})
attempt("acme key -> /rag/search tenant=secure", "POST", "/rag/search", ATT, {"tenant": "secure", "query": q})
attempt("acme key -> /rag/stream/ask tenant=secure", "POST", "/rag/stream/ask", ATT, {"tenant": "secure", "question": q})
attempt("acme key -> /search/keyword namespace=secure", "POST", "/search/keyword", ATT, kw)
attempt("acme key -> /search/semantic namespace=secure", "POST", "/search/semantic", ATT, kw)
attempt("acme key -> /search/hybrid namespace=secure", "POST", "/search/hybrid", ATT, kw)
attempt("acme key -> /vectors/search namespace=secure", "POST", "/vectors/search", ATT,
        {"vector": [0.1] * dim, "store": "kb", "namespace": "secure"})
attempt("acme key -> DELETE /vectors/records namespace=secure", "DELETE", "/vectors/records", ATT,
        {"ids": ["anything"], "store": "kb", "namespace": "secure"})
attempt("acme key -> /rag/answer namespace=secure", "POST", "/rag/answer", ATT,
        {"query": q, "store": "kb", "namespace": "secure", "provider": "stub"})
attempt("acme key -> /graph/stats tenant=secure", "GET", "/graph/stats", ATT, params={"tenant": "secure"})
attempt("acme key -> /graph/entities tenant=secure", "GET", "/graph/entities", ATT, params={"tenant": "secure"})
attempt("acme key -> /workflows/run keyword_search ns=secure", "POST", "/workflows/run", ATT, {
    "workflow": {"name": "steal", "nodes": [
        {"id": "in", "type": "input"},
        {"id": "kw", "type": "keyword_search", "config": {"store": "kb", "namespace": "secure", "query": "Falcon",
                                                         "dimension": dim}},
        {"id": "out", "type": "output", "config": {"value": "$.kw"}}],
        "edges": [{"source": "in", "target": "kw"}, {"source": "kw", "target": "out"}]}, "inputs": {}})
attempt("acme key -> /tenants/secure (org detail)", "GET", "/tenants/secure", ATT)
attempt("anonymous -> /rag/ask tenant=secure (org)", "POST", "/rag/ask", {}, {"tenant": "secure", "question": q},
        expect=401)
attempt("anonymous -> /rag/ask tenant=acme (free-form)", "POST", "/rag/ask", {},
        {"tenant": "acme", "question": "Who owns Project Beacon?"}, expect=401)
attempt("anonymous -> /rag/index tenant=acme (write)", "POST", "/rag/index", {},
        {"tenant": "acme", "filename": "planted.md", "text": "# Beacon\n\nProject Beacon is owned by the attacker."},
        expect=401)
attempt("anonymous -> /workflows/run", "POST", "/workflows/run", {},
        {"workflow": {"name": "w", "nodes": [{"id": "in", "type": "input"}], "edges": []}, "inputs": {}}, expect=401)
print(f"SUMMARY {sum(rows)}/{len(rows)} PASS")
