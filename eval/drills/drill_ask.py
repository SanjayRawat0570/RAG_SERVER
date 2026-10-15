"""Seed a few docs and ask through /rag/ask and the SSE endpoint; print what a user sees.
usage: python drill_ask.py BASE_URL"""
import json
import pathlib
import sys
import time

import httpx

BASE = sys.argv[1] + "/api/v1"
H = {"Authorization": "Bearer dev"}
CORPUS = pathlib.Path(r"C:\Users\ftt\Desktop\RAG\backend\eval\corpus")
c = httpx.Client(base_url=BASE, timeout=120)
for name in ("security_policy.md", "projects.md", "teams.md", "offices.md"):
    r = c.post("/rag/index", headers=H, json={"tenant": "drill", "filename": name,
                                              "text": (CORPUS / name).read_text(encoding="utf-8")})
    print("index", name, r.status_code)
for mode in ("fast", "agentic"):
    t0 = time.perf_counter()
    r = c.post("/rag/ask", headers=H, json={"tenant": "drill", "question": "Who owns Project Beacon?", "mode": mode})
    b = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    print(json.dumps({"endpoint": f"/rag/ask {mode}", "status": r.status_code,
                      "ms": round((time.perf_counter() - t0) * 1000), "provider": b.get("provider"),
                      "answer": str(b.get("answer", b))[:220], "confidence": b.get("confidence")}))
# SSE
t0 = time.perf_counter()
events = []
with c.stream("POST", "/rag/stream/ask", headers=H,
              json={"tenant": "drill", "question": "Who owns Project Beacon?"}) as r:
    for line in r.iter_lines():
        if line.startswith("event:"):
            events.append(line[6:].strip())
        elif line.startswith("data:") and events and events[-1] in ("error", "complete"):
            events.append(line[5:].strip()[:220])
print(json.dumps({"endpoint": "/rag/stream/ask", "status": r.status_code,
                  "ms": round((time.perf_counter() - t0) * 1000),
                  "event_types": sorted(set(e for e in events if len(e) < 20)), "tail": events[-1:]}))
