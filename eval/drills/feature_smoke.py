"""Live smoke checks of feature endpoints. usage: python feature_smoke.py BASE_URL"""
import json
import pathlib
import sys

import httpx

B = sys.argv[1]
c = httpx.Client(base_url=B + "/api/v1", timeout=120, headers={"Authorization": "Bearer dev"})
EX = pathlib.Path(r"C:\Users\ftt\Desktop\RAG\backend\examples")


def show(name, r, keys=None):
    try:
        b = r.json()
    except Exception:
        b = r.text
    if keys and isinstance(b, dict):
        b = {k: b.get(k) for k in keys}
    print(f"{name:<44} {r.status_code}  {json.dumps(b, default=str)[:230]}")


for f in ("f1_linear.json", "f2_branching.json", "f4_parallel_merge.json"):
    show(f"workflows/run {f}", c.post("/workflows/run", json=json.loads((EX / f).read_text())), ["status", "outputs"])
r = c.post("/workflows/stream", json=json.loads((EX / "f4_parallel_merge.json").read_text()))
show("workflows/stream (SSE event count)", r) if r.status_code != 200 else print(
    f"{'workflows/stream (SSE events)':<44} 200  {r.text.count('event:')} events")
m = httpx.get(B + "/api/v1/metrics").text
print(f"{'metrics (prometheus lines)':<44} {len(m.splitlines())} lines; rag_ metrics: "
      f"{len({l.split('{')[0].split(' ')[0] for l in m.splitlines() if l and not l.startswith('#')})} series names")
show("monitoring/traces", c.get("/monitoring/traces"))

# F9 upload + F19 re-upload of a changed document (same filename)
t = "fresh"
v1 = "# Parking\n\nThe parking garage opens at 7am. Each employee gets one parking badge."
v2 = "# Parking\n\nThe parking garage opens at 6am. Each employee gets two parking badges."
for i, txt in enumerate((v1, v1, v2), 1):
    r = c.post("/documents/upload", data={"tenant": t}, files={"file": ("parking.md", txt.encode(), "text/markdown")})
    eid = r.json()["execution_id"]
    import time
    for _ in range(60):
        st = c.get(f"/documents/executions/{eid}").json()
        if st.get("status") in ("success", "completed", "failed", "error", "done"):
            break
        time.sleep(0.5)
    print(f"{f'documents/upload parking.md #{i}':<44} {json.dumps(st, default=str)[:230]}")
ns = c.get("/vectors/stores/kb/ns").json()
print(f"{'kb namespace counts after 3 uploads':<44} {json.dumps(ns)[:300]}")
r = c.post("/rag/ask", json={"tenant": t, "question": "When does the parking garage open?"})
show("ask after doc change", r, ["answer", "citations"])

show("graph/stats", c.get("/graph/stats"))
show("graph/entities?limit", c.get("/graph/entities"))
show("feedback POST", c.post("/feedback", json={"query": "q", "answer": "a", "rating": 1, "user_id": "dev"}))
show("feedback/insights", c.get("/feedback/insights"))
show("personalization/profile", c.get("/personalization/profile"))
show("cost/summary", c.get("/cost/summary"))
show("budgets", c.get("/budgets"))
show("caches", c.get("/caches"))
show("indexing/submit", c.post("/indexing/submit", json={"text": "# Q\n\nQueued doc about quokkas.", "filename": "q.md", "tenant": "fresh"}))
show("indexing/queue/stats", c.get("/indexing/queue/stats"))
show("search/hybrid kb default embed", c.post("/search/hybrid", json={"query": "parking", "store": "kb", "namespace": t}))
show("search/hybrid kb bge embed", c.post("/search/hybrid", json={"query": "parking", "store": "kb", "namespace": t, "embed_model": "bge-small-en-v1.5"}))
show("rag/answer (legacy) kb", c.post("/rag/answer", json={"query": "parking", "store": "kb", "namespace": t}))
show("security/compliance", c.get("/security/compliance"))
show("security/gdpr/export", c.get("/security/gdpr/export"))
show("security/audit/events", c.get("/security/audit/events"))
show("tenant-settings/fresh", c.get("/tenant-settings/fresh"))
# memory follow-up
r1 = c.post("/rag/ask", json={"tenant": "acme", "question": "Who owns Project Beacon?"}).json()
conv = r1.get("conversation_id")
r2 = c.post("/rag/ask", json={"tenant": "acme", "question": "Where is that team based?", "conversation_id": conv})
show("ask follow-up (memory)", r2, ["standalone_question", "rewritten_query", "answer"])
show("conversations list", c.get("/conversations", params={"tenant": "acme"}))
show("ask agentic multihop", c.post("/rag/ask", json={"tenant": "acme", "mode": "agentic",
     "question": "In which office is the team that owns Project Beacon based?"}), ["answer", "agent"])
