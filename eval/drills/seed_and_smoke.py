"""Seed a tenant over HTTP with eval/corpus, then run smoke checks against the live API.

usage: python seed_and_smoke.py BASE_URL TENANT [--no-seed]
"""
import json
import pathlib
import sys
import time

import httpx

BASE, TENANT = sys.argv[1], sys.argv[2]
CORPUS = pathlib.Path(r"C:\Users\ftt\Desktop\RAG\backend\eval\corpus")
H = {"Authorization": "Bearer dev"}

c = httpx.Client(base_url=BASE, timeout=120)
if "--no-seed" not in sys.argv:
    t0, chunks = time.perf_counter(), 0
    docs = sorted(CORPUS.rglob("*.md"))
    for d in docs:
        r = c.post("/api/v1/rag/index", headers=H,
                   json={"tenant": TENANT, "text": d.read_text(encoding="utf-8"), "filename": d.name})
        r.raise_for_status()
        chunks += r.json().get("upserted", 0)
    print(f"seeded {len(docs)} docs / {chunks} chunks into {TENANT!r} in {time.perf_counter()-t0:.1f}s")

def ask(q, tenant=TENANT, headers=H, **kw):
    t0 = time.perf_counter()
    r = c.post("/api/v1/rag/ask", headers=headers, json={"tenant": tenant, "question": q, **kw})
    ms = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, ms, body

for q in ["What is the minimum password length?", "Who owns Project Beacon?",
          "What is the CEO's favourite colour?"]:
    st, ms, b = ask(q)
    if isinstance(b, dict):
        print(json.dumps({"q": q, "status": st, "ms": round(ms), "provider": b.get("provider"),
                          "answerable": b.get("answerable"), "confidence": b.get("confidence"),
                          "cache_hit": b.get("cache_hit"), "answer": str(b.get("answer"))[:200],
                          "sources": [x.get("source") for x in (b.get("citations") or [])][:3]}))
    else:
        print(st, round(ms), str(b)[:300])
