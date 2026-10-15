"""Tools the agent (and the LLM, via function calling) can use.

* ``search_docs(query, filters?)``  — hybrid search (dense + BM25 → RRF → rerank)
* ``get_document(doc_id, page?)``   — the full text of one indexed document
* ``lookup_entity(name)``           — entities and relations from the knowledge graph

Tenant isolation: the tenant is **never** a tool argument. It lives in the
server-side :class:`ToolContext` built from the authenticated request. Any
tenant-like argument the model invents (``tenant``, ``namespace`` …) is stripped
before the tool runs and reported in ``dropped_args``. Every store lookup is
scoped to ``ctx.tenant`` — vector namespaces and graph partitions alike.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.api.pipelines import DIM, STORE
from app.engine.context import ExecutionContext
from app.engine.merging import rrf
from app.engine.nodes.registry import get_node  # registry module: safe during package init
from app.models.workflow import NodeDef, NodeType
from app.rag.graph import find_entities, neighbours
from app.rag.rerank.onnx_cross_encoder import pipeline_method as rerank_method
from app.rag.vectorstore import lookup_store

RESERVED_ARGS = {"tenant", "tenant_id", "namespace", "org", "org_slug", "team", "team_slug", "user_id"}
FILTER_KEYS = {"source", "heading", "document_id", "page", "source_type", "format"}
_PREVIEW = 400

# Function declarations in Gemini's OpenAPI-subset schema. Note: no tenant parameter.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_docs",
        "description": "Search the knowledge base. Returns the most relevant passages with "
                       "their doc_id, source file and heading. Use specific queries; for "
                       "multi-hop questions search for one fact at a time.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Search query"},
                "filters": {
                    "type": "OBJECT",
                    "description": "Optional exact-match metadata filters",
                    "properties": {
                        "source": {"type": "STRING", "description": "File name, e.g. teams.md"},
                        "heading": {"type": "STRING"},
                        "page": {"type": "INTEGER"},
                    },
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Read the full text of one document (or one page of it) by doc_id or file name.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "doc_id": {"type": "STRING", "description": "doc_id from search results, or a file name"},
                "page": {"type": "INTEGER", "description": "Optional page number"},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "lookup_entity",
        "description": "Look up a named entity (person, team, project, place…) in the knowledge "
                       "graph and return its relations, e.g. which team owns a project.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING", "description": "Entity name"}},
            "required": ["name"],
        },
    },
]
TOOL_NAMES = {spec["name"] for spec in TOOL_SPECS}


@dataclass(frozen=True)
class ToolContext:
    """Server-side context for tool execution — the only source of the tenant."""

    tenant: str
    user_id: str = ""
    store: str = STORE
    dimension: int = DIM
    top_k: int = 5


def _nd(node_type: NodeType, node_id: str, config: dict[str, Any]):
    return get_node(NodeDef(id=node_id, type=node_type, config=config))


def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(str(meta.get(k)) == str(v) for k, v in filters.items())


async def hybrid_search(query: str, ctx: ToolContext, filters: dict[str, Any] | None = None
                        ) -> list[dict[str, Any]]:
    """Dense + keyword search in the tenant's namespace, fused with RRF and reranked."""
    filters = {k: v for k, v in (filters or {}).items() if k in FILTER_KEYS and v not in (None, "")}
    ectx = ExecutionContext(run_id=str(uuid.uuid4()), inputs={"question": query, "tenant": ctx.tenant})
    qp = await _nd(NodeType.QUERY_PROCESS, "qp", {"query": "$.inputs.question"}).run(ectx, {})
    ectx.set_output("qp", qp)
    k = ctx.top_k * 2 if filters else ctx.top_k
    common = {"store": ctx.store, "namespace": "$.inputs.tenant", "dimension": ctx.dimension, "top_k": k}
    dense, sparse = await asyncio.gather(
        _nd(NodeType.VECTOR_SEARCH, "dense", {**common, "query": "$.qp.normalized"}).run(ectx, {"qp": qp}),
        _nd(NodeType.KEYWORD_SEARCH, "sparse", {**common, "query": "$.qp.expanded_query"}).run(ectx, {"qp": qp}),
    )
    merged = rrf([dense or [], sparse or []], {"top_n": k})
    if filters:
        merged = [h for h in merged if _matches(h.get("metadata", {}), filters)]
    if not merged:
        return []
    reranked = await _nd(NodeType.RERANK, "rerank",
                         {"method": rerank_method(), "query": "$.qp.normalized",
                          "top_n": ctx.top_k}).run(ectx, {"fuse": merged})
    return reranked if isinstance(reranked, list) else merged[:ctx.top_k]


def _hit_summary(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata", {})
    return {
        "doc_id": meta.get("document_id") or hit.get("id"),
        "source": meta.get("source"),
        "heading": meta.get("heading"),
        "page": meta.get("page"),
        "text": str(meta.get("text", ""))[:_PREVIEW],
    }


async def search_docs(ctx: ToolContext, query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    hits = await hybrid_search(str(query), ctx, filters if isinstance(filters, dict) else None)
    return {"hits": hits, "summary": {"results": [_hit_summary(h) for h in hits]}}


async def get_document(ctx: ToolContext, doc_id: str, page: int | None = None) -> dict[str, Any]:
    store = lookup_store(ctx.store)
    records = store.list_records(namespace=ctx.tenant) if store else []
    doc_id = str(doc_id)
    chunks = [
        (rid, meta) for rid, meta in records
        if doc_id in (meta.get("document_id"), meta.get("source")) or rid.startswith(f"{doc_id}::")
    ]
    if page is not None:
        chunks = [(rid, meta) for rid, meta in chunks if meta.get("page") == page]
    chunks.sort(key=lambda c: (c[1].get("page") or 0, c[1].get("index") or 0))
    hits = [{"id": rid, "score": 1.0, "metadata": meta} for rid, meta in chunks]
    text = "\n\n".join(str(meta.get("text", "")) for _, meta in chunks)
    return {
        "hits": hits,
        "summary": {"doc_id": doc_id, "found": bool(chunks), "chunks": len(chunks),
                    "source": chunks[0][1].get("source") if chunks else None, "text": text[:2000]},
    }


async def lookup_entity(ctx: ToolContext, name: str) -> dict[str, Any]:
    name = str(name).strip()
    entities = find_entities(name=name, limit=5, tenant=ctx.tenant)
    if not entities:
        # Models add generic words ("Data Platform team"): match entities whose
        # name appears inside the requested name too, longest first.
        lowered = name.lower()
        entities = sorted(
            (e for e in find_entities(limit=10_000, tenant=ctx.tenant)
             if len(e.name) > 3 and e.name.lower() in lowered),
            key=lambda e: -len(e.name),
        )[:5]
    out = []
    for e in entities:
        rels = neighbours(e.id, tenant=ctx.tenant)
        out.append({
            "name": e.name,
            "type": e.type,
            "relations": [
                f"{e.name} {r['relation']} {r['entity']['name']}" if r["direction"] == "out"
                else f"{r['entity']['name']} {r['relation']} {e.name}"
                for r in rels[:10]
            ],
        })
    return {"hits": [], "facts": [f for ent in out for f in ent["relations"]],
            "summary": {"entities": out}}


_TOOLS = {"search_docs": search_docs, "get_document": get_document, "lookup_entity": lookup_entity}


async def run_tool(name: str, args: dict[str, Any] | None, ctx: ToolContext) -> dict[str, Any]:
    """Execute one tool call with the tenant taken from *ctx*, never from *args*."""
    args = dict(args or {})
    dropped = sorted(k for k in args if k in RESERVED_ARGS)
    clean = {k: v for k, v in args.items() if k not in RESERVED_ARGS}
    t0 = time.perf_counter()
    result: dict[str, Any] = {"name": name, "args": clean, "dropped_args": dropped,
                              "hits": [], "facts": [], "summary": {}, "error": None}
    if name not in _TOOLS:
        result["error"] = f"unknown tool '{name}'"
    else:
        try:
            out = await _TOOLS[name](ctx, **clean)
            result.update({k: out.get(k, result[k]) for k in ("hits", "facts", "summary")})
        except TypeError as exc:  # bad/missing arguments from the model
            result["error"] = f"invalid arguments: {exc}"
        except Exception as exc:  # noqa: BLE001 - tool failures are reported to the planner
            result["error"] = str(exc)
    result["duration_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result
