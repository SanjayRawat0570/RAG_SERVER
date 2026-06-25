"""RAG Answer Generation API (F16).

Endpoints
---------
GET  /rag/providers          List LLM providers + availability
GET  /rag/models             Full model catalogue with pricing
POST /rag/ask                Full pipeline: search → context → LLM → answer
POST /rag/ask/stream         Same but streams tokens via SSE
POST /rag/generate           Raw LLM call (bring-your-own messages)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.config import settings
from app.rag.cache import SemanticCache, get_cache, cache_stats as _cache_stats
from app.rag.cache.cache import MISS
from app.rag.context import (
    build_context,
    build_prompt,
    estimate_complexity,
    organize_context,
)
from app.rag.cost.pricing import estimate_cost
from app.rag.llm import LLMResponse, get_llm
from app.rag.llm.selector import model_catalogue, select_model
from app.rag.models import estimate_tokens
from app.rag.vectorstore import lookup_store

router = APIRouter(prefix="/rag", tags=["rag"])

# ── Pydantic models ────────────────────────────────────────────────────────────

_QUALITY   = Literal["best", "balanced", "fast", "free"]
_ORGANIZE  = Literal["relevance", "source", "chronological", "diversity"]
_TEMPLATE  = Literal["default", "qa", "summarize", "extract", "chain_of_thought"]


class AskRequest(BaseModel):
    query:          str   = Field(..., min_length=1)
    store:          str   = "default"
    namespace:      str   = "default"
    top_k:          int   = Field(5, ge=1, le=50)
    provider:       str | None = None   # force a specific provider
    quality:        _QUALITY   = "free" # model selection tier
    template:       _TEMPLATE  = "default"
    organize:       _ORGANIZE  = "relevance"
    max_tokens:     int   = Field(512, ge=1, le=4096)
    temperature:    float = Field(0.2, ge=0.0, le=2.0)
    text_field:     str   = "text"
    source_field:   str   = "source"
    use_highlight:  bool  = False
    company_name:   str   = ""
    chain_of_thought: bool = False
    # F17 cache controls
    use_cache:          bool  = True    # enable L1 exact + L2 semantic cache
    cache_ttl:          float = 300.0   # seconds
    semantic_threshold: float = 0.95   # cosine similarity for semantic cache hit
    # F18 tenant controls — if set, namespace is resolved from org/team membership
    org_slug:           str | None = None
    team_slug:          str | None = None


class GenerateRequest(BaseModel):
    messages:    list[dict[str, str]] = Field(..., min_length=1)
    provider:    str | None = None
    model:       str | None = None
    max_tokens:  int   = Field(512, ge=1, le=4096)
    temperature: float = Field(0.2, ge=0.0, le=2.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _post_process(text: str) -> tuple[str, list[str]]:
    """Extract citation markers like [1], [2] from the generated answer."""
    import re
    citations = re.findall(r"\[\d+\]", text)
    return text.strip(), list(dict.fromkeys(citations))


import hashlib

def _exact_cache_key(req: AskRequest) -> str:
    payload = json.dumps({"q": req.query, "s": req.store, "ns": req.namespace,
                          "p": req.provider, "qual": req.quality}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


async def _run_pipeline(req: AskRequest) -> dict[str, Any]:
    """Core RAG pipeline: L1 exact cache → L2 semantic cache → full retrieval."""
    cache_name = f"rag-answer:{req.store}"
    l1_cache   = get_cache(cache_name, ttl=req.cache_ttl)

    # L1 — exact match
    if req.use_cache:
        cached = l1_cache.get(_exact_cache_key(req))
        if cached is not MISS:
            return {**cached, "cache_hit": True, "cache_type": "exact"}

    # 1. Complexity estimate → model selection
    complexity  = estimate_complexity(req.query)
    model_info  = select_model(
        quality=req.quality,
        complexity=complexity["level"],
        provider=req.provider,
    )
    provider_name = model_info["provider"]
    model_hint    = model_info.get("model", "")

    # 2. Retrieve from vector store
    store = lookup_store(req.store)
    if store is None:
        raise HTTPException(404, f"Store '{req.store}' not found.")

    from app.rag.embeddings import embed_texts
    from app.rag.embeddings.registry import _MODEL_DIMS, DEFAULT_DIMENSION, DEFAULT_MODEL
    from app.rag.query import process_query

    processed   = process_query(req.query)
    dim         = _MODEL_DIMS.get(DEFAULT_MODEL, DEFAULT_DIMENSION)
    query_vec   = embed_texts([processed["expanded_query"]], DEFAULT_MODEL, dim)[0]

    # L2 — semantic cache (near-duplicate query reuses previous answer)
    sem_cache = None
    if req.use_cache:
        sem_cache = SemanticCache(cache_name, dim, req.semantic_threshold)
        hit = sem_cache.lookup(query_vec)
        if hit is not None:
            return {**hit, "cache_hit": True, "cache_type": "semantic"}

    raw_hits    = store.search(query_vec, top_k=req.top_k, namespace=req.namespace)

    hits = [
        {
            "id":        h.id,
            "score":     h.score,
            "metadata":  h.metadata,
            "highlight": "",
        }
        for h in raw_hits
    ]

    # 3. Context assembly
    ctx_result  = build_context(
        hits,
        config={
            "max_context_tokens": complexity["recommended_tokens"],
            "text_field":         req.text_field,
        },
    )
    docs = ctx_result["documents"]

    # 4. Build augmented prompt
    context_text = "\n\n".join(
        f"[{i+1}] From {d.get('source') or 'Document'}:\n{d['text']}"
        for i, d in enumerate(docs)
    )
    prompt_obj = build_prompt(
        req.query, context_text,
        config={
            "template_name":    req.template,
            "chain_of_thought": req.chain_of_thought,
        },
    )
    messages = prompt_obj["messages"]

    # 5. LLM generate
    llm     = get_llm(provider_name)
    config  = {
        "model":       model_hint or None,
        "max_tokens":  req.max_tokens,
        "temperature": req.temperature,
    }
    response: LLMResponse = await llm.generate(
        {"messages": messages, "documents": docs, "query": req.query},
        config,
    )

    # 6. Post-process
    answer, citations = _post_process(response.text)

    # Cost estimate
    usage    = response.usage or {}
    in_tok   = usage.get("input_tokens") or estimate_tokens(context_text + req.query)
    out_tok  = usage.get("output_tokens") or estimate_tokens(answer)
    cost     = estimate_cost(response.model, in_tok, out_tok)

    sources = list(dict.fromkeys(
        d.get("source") or f"doc-{i}" for i, d in enumerate(docs)
    ))

    result = {
        "query":      req.query,
        "answer":     answer,
        "citations":  citations,
        "sources":    sources,
        "provider":   response.provider,
        "model":      response.model,
        "complexity": complexity,
        "context": {
            "chunks_used":    len(docs),
            "chunks_dropped": len(ctx_result["dropped"]),
            "token_estimate": ctx_result["token_estimate"],
        },
        "usage": {
            "input_tokens":       in_tok,
            "output_tokens":      out_tok,
            "total_tokens":       in_tok + out_tok,
            "estimated_cost_usd": cost,
        },
        "finish_reason": response.finish_reason,
        "cache_hit":  False,
        "cache_type": None,
    }

    # Populate L1 + L2 caches for future requests.
    if req.use_cache:
        l1_cache.set(_exact_cache_key(req), result)
        if sem_cache is not None:
            sem_cache.put(req.query, query_vec, result)

    return result


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/providers")
async def list_providers(_: CurrentUser) -> dict[str, Any]:
    """List available LLM providers and which is currently active."""
    return {
        "active":    settings.llm_provider,
        "providers": {
            "stub":   {"available": True,  "free": True,
                       "description": "Offline extractive stub (default, no API key needed)"},
            "gemini": {"available": bool(settings.gemini_api_key),
                       "free": True,
                       "description": "Google Gemini free tier (set GEMINI_API_KEY)"},
            "openai": {"available": bool(settings.openai_api_key),
                       "free": False,
                       "description": "OpenAI GPT-4o / GPT-4o-mini (set OPENAI_API_KEY)"},
            "claude": {"available": bool(settings.anthropic_api_key),
                       "free": False,
                       "description": "Anthropic Claude Haiku/Sonnet/Opus (set ANTHROPIC_API_KEY)"},
        },
    }


@router.get("/models")
async def list_models(_: CurrentUser) -> dict[str, Any]:
    """Full model catalogue with pricing and availability."""
    catalogue = model_catalogue()
    return {
        "models":  catalogue,
        "total":   len(catalogue),
        "default": settings.llm_provider,
    }


@router.post("/answer")
async def ask(req: AskRequest, user: CurrentUser) -> dict[str, Any]:
    """Full RAG pipeline: search → context assembly → LLM → grounded answer.

    If org_slug/team_slug are provided the namespace is resolved from the
    caller's membership, giving automatic per-tenant data isolation (F18).
    """
    from app.rag.tenants import record_usage, resolve_namespace

    # F18: auto-resolve namespace from tenant membership.
    if req.namespace == "default" and (req.org_slug or req.team_slug):
        req = req.model_copy(update={
            "namespace": resolve_namespace(
                user["id"], org_slug=req.org_slug, team_slug=req.team_slug
            )
        })

    try:
        result = await _run_pipeline(req)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Pipeline error: {exc}")

    # F18: record usage per user / org.
    if req.org_slug:
        record_usage(
            org_slug=req.org_slug,
            user_id=user["id"],
            team_slug=req.team_slug,
            tokens=result.get("usage", {}).get("total_tokens", 0),
            cost_usd=result.get("usage", {}).get("estimated_cost_usd", 0.0),
            cache_hit=result.get("cache_hit", False),
        )

    return result


@router.post("/answer/stream")
async def ask_stream(req: AskRequest, _: CurrentUser) -> StreamingResponse:
    """Stream the LLM answer token by token as Server-Sent Events."""
    # Build context synchronously, then stream LLM tokens.
    store = lookup_store(req.store)
    if store is None:
        raise HTTPException(404, f"Store '{req.store}' not found.")

    complexity    = estimate_complexity(req.query)
    model_info    = select_model(
        quality=req.quality,
        complexity=complexity["level"],
        provider=req.provider,
    )
    provider_name = model_info["provider"]
    model_hint    = model_info.get("model", "")

    from app.rag.embeddings import embed_texts
    from app.rag.embeddings.registry import _MODEL_DIMS, DEFAULT_DIMENSION, DEFAULT_MODEL
    from app.rag.query import process_query

    processed  = process_query(req.query)
    dim        = _MODEL_DIMS.get(DEFAULT_MODEL, DEFAULT_DIMENSION)
    query_vec  = embed_texts([processed["expanded_query"]], DEFAULT_MODEL, dim)[0]
    raw_hits   = store.search(query_vec, top_k=req.top_k, namespace=req.namespace)

    hits = [{"id": h.id, "score": h.score, "metadata": h.metadata} for h in raw_hits]
    ctx_result   = build_context(hits, config={"max_context_tokens": complexity["recommended_tokens"],
                                               "text_field": req.text_field})
    docs         = ctx_result["documents"]
    context_text = "\n\n".join(
        f"[{i+1}] From {d.get('source') or 'Document'}:\n{d['text']}"
        for i, d in enumerate(docs)
    )
    prompt_obj = build_prompt(req.query, context_text,
                              config={"template_name": req.template,
                                      "chain_of_thought": req.chain_of_thought})
    messages   = prompt_obj["messages"]

    llm    = get_llm(provider_name)
    config = {"model": model_hint or None, "max_tokens": req.max_tokens,
              "temperature": req.temperature}

    async def _event_stream():
        # Send a preamble so the client knows sources before tokens arrive.
        sources = list(dict.fromkeys(
            d.get("source") or f"doc-{i}" for i, d in enumerate(docs)
        ))
        meta = json.dumps({"event": "meta", "sources": sources,
                           "provider": provider_name, "complexity": complexity["level"]})
        yield f"data: {meta}\n\n"

        try:
            async for token in llm.generate_stream(
                {"messages": messages, "documents": docs, "query": req.query}, config
            ):
                chunk = json.dumps({"event": "token", "text": token})
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            err = json.dumps({"event": "error", "detail": str(exc)})
            yield f"data: {err}\n\n"
            return

        done = json.dumps({"event": "done"})
        yield f"data: {done}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/generate")
async def generate(req: GenerateRequest, _: CurrentUser) -> dict[str, Any]:
    """Raw LLM call — supply your own messages, skip RAG retrieval."""
    provider_name = req.provider or settings.llm_provider
    llm    = get_llm(provider_name)
    config = {
        "model":       req.model,
        "max_tokens":  req.max_tokens,
        "temperature": req.temperature,
    }
    try:
        response: LLMResponse = await llm.generate({"messages": req.messages}, config)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))

    answer, citations = _post_process(response.text)
    usage   = response.usage or {}
    in_tok  = usage.get("input_tokens")  or estimate_tokens(" ".join(m["content"] for m in req.messages))
    out_tok = usage.get("output_tokens") or estimate_tokens(answer)
    cost    = estimate_cost(response.model, in_tok, out_tok)

    return {
        "answer":   answer,
        "citations": citations,
        "provider": response.provider,
        "model":    response.model,
        "usage": {
            "input_tokens":       in_tok,
            "output_tokens":      out_tok,
            "total_tokens":       in_tok + out_tok,
            "estimated_cost_usd": cost,
        },
        "finish_reason": response.finish_reason,
    }
