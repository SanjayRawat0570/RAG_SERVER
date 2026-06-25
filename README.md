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

## Roadmap

**Done:** orchestration **F1–F8** + complete RAG pipeline **F9–F16**
(ingest → chunk → embed → upsert → query → hybrid search → RRF → rerank →
augment → generate). End-to-end, offline, with tracing/metrics on every node.

**Also done:** F17 caching & F24 cost/token controls.

**Next:** F18 multi-tenancy quotas · F19 incremental indexing · F20 hybrid-search
weighting · F21 knowledge graph · F22 personalization · F23 feedback loop ·
F25 security & RBAC.
# RAG_SERVER
