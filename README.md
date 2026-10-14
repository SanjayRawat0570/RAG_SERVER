# Enterprise RAG System — Orchestration Engine

Production-grade workflow & orchestration engine. This phase implements the
core of the system, **F1–F4**, on which the RAG pipeline (F9+) and observability
(F8) will be layered.

| Feature | Status | What it does |
|---------|--------|--------------|
| **F1** Basic Workflow Execution | ✅ | JSON workflow → validated DAG, typed nodes, execution context, retry + fallback, **nested sub-workflows + loops** |
| **F2** Graph Branching | ✅ | Edge conditions with AND/OR/NOT, numeric/string/array operators |
| **F3** Chaining & Parallel Execution | ✅ | Topological execution, context passing, independent nodes run concurrently |
| **F4** Merging Results | ✅ | `concat`, `voting`, `ranking`, `weighted`, `dedup`, `consensus` |
| **F5** Conditional Logic & Decision Trees | ✅ | `switch` node for multi-way routing (first-match case → label) |
| **F6** Streaming & Real-time | ✅ | `/workflows/stream` emits Server-Sent Events per node as it completes |
| **F7** Error Handling & Fallbacks | ✅ | Per-node retry w/ backoff, static fallback, **circuit breaker** shared across runs |
| **F8** Observability & Tracing | ✅ | OpenTelemetry spans (HTTP→workflow→node) to Jaeger, Prometheus metrics, trace-id in JSON logs, Grafana dashboard |
| **F9** Document Ingestion | ✅ | Format detection + parsers (text/markdown/html/json/pdf/docx) behind lazy adapters, cleaning, metadata |
| **F10** Advanced Chunking | ✅ | `fixed` (overlap), `recursive`, `semantic`, `structure` (markdown headings); chunks carry provenance + token estimates |
| **F11** Multi-Model Embeddings | ✅ | Pluggable embedders (default: offline deterministic hash embedder), embedding cache; sentence-transformers/API as adapters |
| **F12** Vector DB Ops | ✅ | In-process FLAT cosine index: upsert (update-or-insert), namespaces (multi-tenant), metadata filtering, delete, stats |
| **F13** Query Processing & Search | ✅ | Query understanding (intent/entities/expansion), BM25 sparse search, hybrid dense+sparse via RRF fusion |
| **F14** Reranking | ✅ | `cross_encoder` (semantic⊕lexical), `mmr` diversity, `recency` decay, `authority` weighting, `multi_factor` blend |
| **F15** Context Augmentation | ✅ | Token-budgeted context with citations, prompt templates, chain-of-thought, chat `messages` output |
| **F16** LLM Generation | ✅ | Pluggable providers: offline extractive stub (default) + Google Gemini adapter (free tier, opt-in via `GEMINI_API_KEY`) |
| **F17** Caching & Performance | ✅ | TTL/LRU response cache + semantic answer cache (near-duplicate queries reuse answers), cache hit/miss metrics |
| **F24** Cost & Token Counting | ✅ | Token counting, per-model pricing/cost estimation, per-tenant budgets with pre-spend enforcement |

## Architecture

```
app/
├── main.py              FastAPI app
├── config.py            settings + structured JSON logging
├── models/workflow.py   workflow / node / edge / result schemas
├── api/routes.py        /workflows/run, /workflows/validate
└── engine/
    ├── graph.py         DAG build + validation + topological generations (NetworkX)
    ├── context.py       run state + $.node.field reference resolver
    ├── conditions.py    F2 condition evaluator
    ├── merging.py       F4 merge strategies
    ├── executor.py      F1/F2/F3 execution: branching, chaining, parallelism
    └── nodes/           input · processing · decision · merge · output (+ registry)
```

## Run with Docker

```bash
# Full stack: API + Jaeger + Prometheus + Grafana
docker compose up --build

# API only
docker compose up --build api

# Run the test suite (verifies F1–F8)
docker compose --profile test run --rm tests
```

| Service | URL |
|---------|-----|
| API + Swagger | http://localhost:8000/docs |
| Prometheus metrics (app) | http://localhost:8000/api/v1/metrics |
| Jaeger UI (traces) | http://localhost:16686 |
| Prometheus | http://localhost:9090 |
| Grafana (anonymous admin) | http://localhost:3000 → dashboard "RAG Orchestrator — Overview" |
| Vector stores / caches / budgets | `/api/v1/vectorstores` · `/api/v1/caches` · `/api/v1/budgets` |

## Try it

```bash
curl -s -X POST http://localhost:8000/api/v1/workflows/run \
  -H 'content-type: application/json' \
  -d @examples/f2_branching.json | python -m json.tool
```

Example workflows live in [`examples/`](examples/): `f1_linear.json`,
`f2_branching.json`, `f4_parallel_merge.json`.

## Node types

- **input** — exposes run inputs (optionally a single `key`)
- **processing** — named operation (`uppercase`, `set`, `multiply`, `score`, …); RAG ops (embed/chunk/search) plug in here later
- **decision** — forwards a value and can pre-compute named condition `checks`
- **switch** — multi-way routing: first matching `case` → `{case, value}` (F5)
- **merge** — combines active upstream branches via an F4 strategy
- **external** — calls an external HTTP API; opt-in circuit breaker via `breaker_key` (F7)
- **subworkflow** — runs an embedded workflow as one node, with `input_map` (F1 nested)
- **loop** — repeats an embedded workflow until an `until` condition or `max_iterations` (F1 looping)
- **ingest** — parses raw input (`text`/`content_base64`/`path`) into a normalized Document (F9)
- **chunk** — splits a Document into chunks via a strategy (F10)
- **embed** — attaches dense vectors to chunks or a query (F11)
- **upsert** — writes embedded chunks into a vector store + namespace (F12)
- **vector_search** — embeds a query and retrieves top-k chunks with metadata filtering (F12)
- **query_process** — query understanding: intent, entities, keywords, expansion (F13)
- **keyword_search** — BM25 sparse retrieval over a store namespace (F13)
- **rerank** — reorders candidates (`cross_encoder`/`mmr`/`recency`/`authority`/`multi_factor`) (F14)
- **augment** — builds token-budgeted context + citations + chat prompt `messages` (F15)
- **generate** — LLM answer generation via a pluggable provider (F16)
- **output** — final result (`value` may be a `$.` ref or a dict/list of refs)

The full RAG answer pipeline:
`query_process → [vector_search, keyword_search] → merge(rrf) → rerank → augment → generate`
(see [examples/f16_rag_answer.json](examples/f16_rag_answer.json)). Run an index
pipeline first (`examples/f11_f12_index.json`).

Hybrid search = `query_process → [vector_search, keyword_search] → merge(rrf)` —
parallel dense + sparse retrieval fused with Reciprocal Rank Fusion (F4 `rrf`).

## Streaming (F6)

```bash
curl -sN -X POST http://localhost:8000/api/v1/workflows/stream \
  -H 'content-type: application/json' -d @examples/f4_parallel_merge.json
# event: workflow_start / node_complete (×N) / workflow_end  (SSE)
```

## Workflow definition

```json
{
  "workflow": {
    "name": "demo",
    "nodes": [{ "id": "in", "type": "input" }, { "id": "out", "type": "output" }],
    "edges": [{ "source": "in", "target": "out",
                "condition": { "left": "$.inputs.x", "op": ">", "right": 1 } }]
  },
  "inputs": { "x": 2 }
}
```

References use `$.inputs.<field>` or `$.<node_id>.<field>` to read context.
Retry/fallback are per-node (`retry.max_attempts`, `fallback`).

## RAG pipeline layout

```
app/rag/
├── models.py              Document + Chunk (provenance, token estimates)
├── ingestion/             F9 — detection, parsers (lazy adapters), cleaning
│   ├── parsers.py         text · markdown · html · json · pdf · docx
│   └── registry.py        format map + Document assembly
├── chunking/strategies.py F10 — fixed · recursive · semantic · structure
├── embeddings/            F11 — Embedder protocol, local hash embedder, cache
│   ├── base.py            HashEmbedder (offline, deterministic)
│   └── registry.py        model registry + embedding cache
├── vectorstore/           F12 — upsert/search/delete over namespaces
│   ├── memory.py          in-process FLAT cosine index (NumPy)
│   └── registry.py        process-wide stores (persist across runs)
├── query/processor.py     F13 — intent · entities · expansion · normalization
├── search/bm25.py         F13 — BM25 Okapi sparse ranking
├── rerank/rerankers.py    F14 — cross-encoder · MMR · recency · authority · multi-factor
├── context/               F15 — context budgeting + citations, prompt assembly
│   ├── builder.py         token-budgeted context block + citation map
│   └── prompt.py          templates · chain-of-thought · chat messages
├── llm/                   F16 — answer generation
│   ├── stub.py            offline extractive answerer (default)
│   ├── gemini.py          Google Gemini REST adapter (opt-in via GEMINI_API_KEY)
│   └── registry.py        provider registry (default from settings)
├── cache/                 F17 — TTL/LRU cache + semantic answer cache
│   ├── cache.py           named TTL/LRU caches (Redis seam)
│   └── semantic.py        near-duplicate query reuse (on the vector store)
└── cost/                  F24 — token counting, pricing, budgets
    ├── pricing.py         per-model cost estimation
    └── budget.py          per-tenant spend limits + enforcement
```

Caching, cost, and budgets are opt-in via the `generate` node config
(`cache`, `semantic_cache`, `budget_key`/`budget_limit`).

Swap in real components without touching the pipeline: register a
sentence-transformers/OpenAI embedder in `embeddings/registry.py`, or a
Qdrant/Weaviate backend in `vectorstore/registry.py`. Heavy ingestion formats
(OCR/audio/unstructured.io) register a parser in `ingestion/registry.py::PARSERS`.

## LLM provider (F16)

Default is the **offline extractive stub** — no API key, fully deterministic, so
tests and local runs need nothing external. To use **Google Gemini** (free tier):

```bash
export GEMINI_API_KEY=your_key          # or set in docker-compose env
# then set the generate node config: {"provider": "gemini", "model": "gemini-1.5-flash"}
```

Add other providers (Claude/OpenAI/Ollama) by registering a factory in
`app/rag/llm/registry.py` — the `generate` node is provider-agnostic.

Every model call goes through the **LLM gateway** (`app/rag/llm/gateway.py`):
per-provider circuit breaker, per-tenant budget guard, cost ledger, Prometheus
usage and an OpenTelemetry `llm.call` span. A provider without an API key is
rejected up front and the pipeline degrades to the extractive answer.

## Vision ingestion (multimodal)

Uploaded PDFs are split per page, so every chunk and citation carries its `page`.
When **vision ingestion** is on for a workspace (*Settings → AI Settings*, or
`PUT /api/v1/tenant-settings/{tenant}` with `{"vision_enabled": true}`, admins only):

* a **scanned page** (under 40 characters of extractable text) has its page image
  sent to Gemini, which returns the page's text and a one-sentence description;
* a **figure or chart** embedded in a text page is described ("bar chart of …,
  rising from 22 to 25") with any labels and numbers it contains;
* a standalone **image upload** (`.png`, `.jpg`, …) is read the same way.

Results are stored as normal chunks with `source_type: "vision"` and the page
number, so they are searched, reranked and cited like text. Each image is one
gateway call (`operation="vision"`: circuit breaker, budget, cost ledger, span).

Limits and degradation: images under 150 px are ignored (icons, logos), at most
`VISION_MAX_IMAGES` (20) per document, sent at ≤ 1536 px. With vision off or no
`GEMINI_API_KEY`, ingestion keeps the text layer, logs why images were skipped,
and a scan-only file fails with a message saying how to enable it. One failing
image never fails the upload. Images are taken from the PDF's embedded images;
charts drawn as vector graphics aren't seen (that needs a page renderer such as
PyMuPDF). The free tier allows few `gemini-2.5-flash` requests per day; set
`VISION_MODEL` to a lite model for large scans.

## MCP server

`app/mcp_server.py` exposes the knowledge base to any MCP client (Claude Desktop,
Claude Code, IDEs) over stdio, using the official Python MCP SDK:

| Tool | Returns |
|---|---|
| `search_knowledge_base(query, top_k=5)` | ranked passages with source, heading, page and score |
| `ask(question, mode="fast", conversation_id?)` | `answer`, cited sources, `confidence`, `answerable`, `conversation_id` (pass it back for follow-ups) |

**1. Create a tenant-scoped API key.** In the web app: *Settings → MCP access*.
Or with the API (the key's role can't be higher than yours in that workspace):

```bash
curl -X POST http://localhost:8000/api/v1/security/api-keys \
  -H "Authorization: Bearer <your token>" -H "Content-Type: application/json" \
  -d '{"name": "claude-desktop", "tenant": "acme", "role": "viewer"}'
```

The key is shown once. It can only reach its own tenant: `viewer` may query,
`member` may also ingest, `admin`/`owner` may also change workspace settings.

**2. Start the API** (`uvicorn app.main:app --port 8000`). The MCP server is a
thin client of the REST API. Every tool call is an HTTP request carrying the key,
so it gets exactly the REST API's authentication, RBAC and tenant isolation. The
tenant is never a tool argument; it comes from the key (`GET /security/whoami`).

**3. Register it in Claude Desktop** (*Settings → Developer → Edit config*,
`claude_desktop_config.json`), using absolute paths:

```json
{
  "mcpServers": {
    "enterprise-rag": {
      "command": "C:\\path\\to\\backend\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "C:\\path\\to\\backend",
      "env": {
        "RAG_API_KEY": "sk-…",
        "RAG_API_URL": "http://localhost:8000/api/v1"
      }
    }
  }
}
```

On macOS/Linux use `.venv/bin/python`. If your Claude Desktop version ignores
`cwd`, add `"PYTHONPATH": "/path/to/backend"` to `env`. Restart Claude Desktop;
the two tools appear under the tools icon. For Claude Code, run
`claude mcp add enterprise-rag -e RAG_API_KEY=sk-… -- /path/to/.venv/bin/python -m app.mcp_server`
from `backend/`. To try it without a client, run `mcp dev app/mcp_server.py` (opens the MCP Inspector; needs Node.js).

## Concepts demonstrated

| Concept | Feature | Where |
|---|---|---|
| **Tool calling** | Three tools are exposed to the model as Gemini function declarations: `search_docs(query, filters?)` (hybrid search), `get_document(doc_id, page?)` and `lookup_entity(name)` (knowledge graph). The tenant is never a tool parameter: it comes from the server-side `ToolContext`, and any tenant-like argument the model sends is stripped and reported. | `app/rag/agent/tools.py`, `app/rag/llm/gemini.py` (`tools`, `tool_calls`), `app/rag/graph/store.py` (per-tenant graphs) |
| **Agents** | `mode: "agentic"` on `/rag/ask` and `/rag/stream/ask` runs an explicit state machine — plan → search → evaluate → answer \| rewrite — inside the DAG engine's `loop` node (one iteration per tool round). Hard limits: 3 tool rounds, 45 s deadline (also bounds each model call), 8 000-token planning budget. Without a tool-capable model a heuristic planner takes over (search + entity lookup, then a bridge-entity rewrite for questions like "the team that owns X"). `fast` stays the default. | `app/rag/agent/planner.py`, `app/engine/nodes/agent.py`, `app/api/pipelines.py` (`build_agentic_workflow`), `tests/test_agentic.py` |
| **Chain of thought** | The agent reasons before acting, but the reasoning stays internal: Gemini's thought parts are dropped, and the UI only receives the `steps` trace — tool calls, query rewrites, evidence coverage and why it stopped. | `app/rag/agent/planner.py` (`_step`), `app/rag/llm/gemini.py` |
| **Conversation memory** | Short-term: the last 5 turns per conversation, keyed by tenant + user + `conversation_id`. Before retrieval a follow-up is rewritten into a standalone question ("what about part-time staff?" → "What is the leave policy for part-time staff?") by the LLM, or by a heuristic offline; the rewrite appears as a `memory_rewrite` step in the trace. Long-term: topics of the sources cited in each answer go into the user's history, and inferred interests feed the personalization boosts through a new `personalize` node. `/conversations` lists, opens and clears chats; GDPR export/erase include memory. Anonymous callers get no memory. | `app/rag/memory/` (`store.py`, `rewriter.py`, `service.py`), `app/engine/nodes/personalize.py`, `app/api/conversations.py`, `app/rag/security/gdpr.py`, `tests/test_memory.py` |
| **Few-shot prompting** | Two example question → cited-answer pairs (one answerable, one "I don't know") are rendered into the system prompt in the exact JSON shape the model must return. Examples are configurable per tenant (`GET/PUT /api/v1/tenant-settings/{tenant}`) and can be toggled per request (`few_shot: false`). `eval/run_eval.py --compare-fewshot` measures their effect. | `app/rag/tenants/ai_settings.py`, `app/rag/context/prompt.py` (`render_examples`), `app/api/tenant_settings.py`, `eval/results/fewshot_comparison_*.json`, `tests/test_few_shot.py` |
| **Multimodal input** | Scanned PDF pages, embedded figures/charts and image uploads are sent to Gemini's vision input (inline image + JSON schema `{text, description}`). The output becomes ordinary chunks with `source_type: "vision"` and the page number, so citations point to the page. Opt-in per tenant (cost); skipped with a logged warning when there is no API key. | `app/rag/ingestion/vision.py`, `app/rag/ingestion/parsers.py` (per-page segments), `app/rag/chunking/strategies.py` (page-aware chunks), `app/api/upload.py`, `app/rag/llm/gateway.py` (image token budget), `tests/test_vision.py` |
| **MCP** | An MCP server (official Python SDK, stdio) exposes `search_knowledge_base(query, top_k)` and `ask(question)`. It authenticates with an API key scoped to one tenant and role, and calls the REST API with it, so the same RBAC check (`rbac.authorize`) and tenant isolation apply. Keys are hashed and persisted so a key made in the web UI works for the MCP process. | `app/mcp_server.py`, `app/rag/security/rbac.py`, `app/rag/security/apikeys.py`, `app/api/deps.py` (`require_tenant`), `tests/test_mcp.py` |
| **Structured output** | The answer is JSON `{answer, citations, confidence, answerable}` validated by Pydantic. Gemini runs in JSON-schema mode; invalid replies are retried once with the validation error, then fall back to the extractive answer. SSE streams the `answer` field and finishes with a `sources` event. | `app/rag/llm/structured.py`, `app/rag/llm/gemini.py` (`build_payload`), `app/engine/nodes/generate.py`, `app/api/stream.py`, `tests/test_structured_output.py` |

## Evaluation

```bash
python eval/run_eval.py --label my_run            # offline stub, deterministic
python eval/run_eval.py --provider gemini --delay 7  # real model, free-tier friendly
```

Indexes `eval/corpus/` into an isolated tenant, asks the 28 questions in
`eval/questions.jsonl` (answerable, multi-hop, unanswerable) and reports citation
rate/precision, answer accuracy, "I don't know" accuracy and latency. Results are
saved under `eval/results/`. `fallback_rate` counts answers that came from the
extractive fallback because the provider failed; a real-provider run aborts after
three consecutive fallbacks rather than saving stub numbers as model results.

**Few-shot comparison** (`eval/results/fewshot_comparison_gemini.json`,
`gemini-3.1-flash-lite`, 28 questions):

| metric | without examples | with examples |
|---|---|---|
| citation rate | 0.85 | 0.80 |
| citation precision | 1.00 | 1.00 |
| answer accuracy | 0.80 | 0.80 |
| "I don't know" accuracy | 1.00 | 1.00 |
| false "I don't know" | 0.20 | 0.20 |
| fallback rate | 0.07 | 0.00 |

On this small set the examples made no measurable difference: the model already
declines all unanswerable questions. The one behavioural change is format: without
examples it once answered "I don't know" while still listing a citation (m03);
with examples it returned `citations: []`, which is the whole citation-rate
difference. The false "I don't know" answers are the multi-hop questions, which
the single-pass fast mode cannot retrieve — see agentic mode. The free tier
allows only 20 `gemini-2.5-flash` requests per day, so run comparisons on a lite
model (`--model`).

**Fast vs agentic** (offline stub, `eval/results/f3_*_stub.json`):

| metric | fast | agentic |
|---|---|---|
| answer accuracy | 0.70 | 0.85 |
| multi-hop accuracy | 0.00 | 1.00 |
| citation precision | 1.00 | 1.00 |
| "I don't know" accuracy | 1.00 | 1.00 |

With Gemini (`gemini-3.1-flash-lite`) the model plans its own hops, e.g. for
"In which office is the team that owns Project Beacon based?" it searched
"Project Beacon owner team", then "Mobile team office location", and answered
"Lisbon" citing both files.

## Roadmap

**Done:** orchestration **F1–F8** + complete RAG pipeline **F9–F16**
(ingest → chunk → embed → upsert → query → hybrid search → RRF → rerank →
augment → generate). End-to-end, offline, with tracing/metrics on every node.

**Also done:** F17 caching · F18 multi-tenancy · F19 incremental indexing ·
F20 hybrid-search weighting · F21 knowledge graph · F22 personalization ·
F23 feedback loop · F24 cost/token controls · F25 security & compliance.

**AI concepts:** structured output · few-shot prompting · tool calling and an
agentic mode (agents, chain of thought) · conversation memory · MCP server with
tenant-scoped keys and RBAC · multimodal (vision) ingestion. See
[Concepts demonstrated](#concepts-demonstrated).
# RAG_SERVER
