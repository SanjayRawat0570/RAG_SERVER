"""Context Augmentation API (F15).

Endpoints
---------
GET  /context/templates          List prompt templates
POST /context/estimate           Estimate query complexity + recommended context size
POST /context/select             Select and budget chunks from a provided hit list
POST /context/build              Full pipeline: search → select → organize → build prompt
POST /context/assemble           Build prompt from provided hits (no search step)
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.context import (
    build_context,
    build_prompt,
    estimate_complexity,
    group_by_source,
    list_templates,
    organize_context,
)
from app.rag.models import estimate_tokens
from app.rag.vectorstore import lookup_store

router = APIRouter(prefix="/context", tags=["context"])

# ── Pydantic models ────────────────────────────────────────────────────────────

class HitIn(BaseModel):
    id:       str
    score:    float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlight: str = ""


_ORGANIZE = Literal["relevance", "source", "chronological", "diversity"]
_TEMPLATE = Literal["default", "qa", "summarize", "extract", "chain_of_thought"]


class SelectRequest(BaseModel):
    hits:           list[HitIn]
    max_tokens:     int     = Field(2000, ge=10, le=16000)
    use_highlight:  bool    = False
    organize:       _ORGANIZE = "relevance"
    source_field:   str     = "source"
    date_field:     str     = "date"
    text_field:     str     = "text"


class BuildRequest(BaseModel):
    query:          str = Field(..., min_length=1)
    store:          str = "default"
    namespace:      str = "default"
    top_k:          int = Field(10, ge=1, le=50)
    max_tokens:     int = Field(2000, ge=10, le=16000)
    auto_budget:    bool = True   # use complexity estimator to set max_tokens
    organize:       _ORGANIZE = "relevance"
    template:       _TEMPLATE = "default"
    source_field:   str = "source"
    date_field:     str = "date"
    text_field:     str = "text"
    use_highlight:  bool = False
    company_name:   str = ""


class AssembleRequest(BaseModel):
    query:          str = Field(..., min_length=1)
    hits:           list[HitIn]
    max_tokens:     int     = Field(2000, ge=10, le=16000)
    auto_budget:    bool    = True
    organize:       _ORGANIZE = "relevance"
    template:       _TEMPLATE = "default"
    source_field:   str     = "source"
    date_field:     str     = "date"
    text_field:     str     = "text"
    use_highlight:  bool    = False
    company_name:   str     = ""
    chain_of_thought: bool  = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _select_and_organize(
    hits: list[dict],
    max_tokens: int,
    text_field: str,
    use_highlight: bool,
    organize: str,
    source_field: str,
    date_field: str,
) -> list[dict]:
    """Select chunks within budget, then organize them."""
    context_result = build_context(
        hits,
        config={
            "max_context_tokens": max_tokens,
            "text_field": text_field,
        },
    )
    # Rebuild hit list from what was included, augmenting with context_text.
    included_ids = set(context_result["included"])
    selected = []
    for doc in context_result["documents"]:
        match = next((h for h in hits if h["id"] == doc["id"]), None)
        if match:
            new = dict(match)
            new["context_text"] = doc["text"]
            new["token_count"]  = estimate_tokens(doc["text"])
            if use_highlight and match.get("highlight"):
                new["context_text"] = match["highlight"]
                new["token_count"]  = estimate_tokens(match["highlight"])
            selected.append(new)

    return organize_context(
        selected,
        strategy=organize,
        source_field=source_field,
        date_field=date_field,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/templates")
async def get_templates(_: CurrentUser) -> dict[str, Any]:
    """List all prompt templates with their system prompt text."""
    templates = list_templates()
    return {
        "templates": templates,
        "total":     len(templates),
        "default":   "default",
    }


@router.post("/estimate")
async def estimate_query_complexity(
    body: dict[str, str], _: CurrentUser
) -> dict[str, Any]:
    """Estimate query complexity and recommend context size."""
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(422, "query is required")
    return estimate_complexity(query)


@router.post("/select")
async def select_context_chunks(req: SelectRequest, _: CurrentUser) -> dict[str, Any]:
    """Select and organize chunks from a provided hit list within a token budget."""
    hits = [h.model_dump() for h in req.hits]
    organized = _select_and_organize(
        hits,
        max_tokens=req.max_tokens,
        text_field=req.text_field,
        use_highlight=req.use_highlight,
        organize=req.organize,
        source_field=req.source_field,
        date_field=req.date_field,
    )
    total_tokens = sum(c.get("token_count", 0) for c in organized)
    return {
        "selected":      organized,
        "total_chunks":  len(organized),
        "total_tokens":  total_tokens,
        "by_source":     {
            src: len(chunks)
            for src, chunks in group_by_source(organized, req.source_field).items()
        },
    }


@router.post("/build")
async def build_augmented_prompt(req: BuildRequest, _: CurrentUser) -> dict[str, Any]:
    """Full pipeline: search the store → select → organize → build prompt."""
    store = lookup_store(req.store)
    if store is None:
        raise HTTPException(404, f"Store '{req.store}' not found.")

    # 1. Embed and search
    from app.rag.search.semantic import semantic_search
    result = semantic_search(
        req.query,
        store_name=req.store,
        namespace=req.namespace,
        top_k=req.top_k,
        text_field=req.text_field,
    )
    hits = result.get("hits", [])
    if not hits:
        return {
            "query":          req.query,
            "prompt":         build_prompt(req.query, "", config={"template_name": req.template}),
            "context_chunks": [],
            "complexity":     estimate_complexity(req.query),
            "sources":        [],
            "total_tokens":   0,
            "warning":        "No results found in store.",
        }

    # 2. Complexity-aware budget
    complexity = estimate_complexity(req.query)
    max_tokens = complexity["recommended_tokens"] if req.auto_budget else req.max_tokens

    # 3. Select + organize
    organized = _select_and_organize(
        hits,
        max_tokens=max_tokens,
        text_field=req.text_field,
        use_highlight=req.use_highlight,
        organize=req.organize,
        source_field=req.source_field,
        date_field=req.date_field,
    )

    # 4. Build context block and prompt
    context_text = "\n\n".join(
        f"[{i+1}] From {c.get('metadata', {}).get(req.source_field, 'Document')}:\n{c.get('context_text', '')}"
        for i, c in enumerate(organized)
    )
    prompt_result = build_prompt(
        req.query, context_text,
        config={"template_name": req.template, "chain_of_thought": False},
    )

    sources = list(dict.fromkeys(
        c.get("metadata", {}).get(req.source_field, f"doc-{i}")
        for i, c in enumerate(organized)
    ))

    return {
        "query":          req.query,
        "prompt":         prompt_result,
        "context_chunks": organized,
        "complexity":     complexity,
        "sources":        sources,
        "total_tokens":   prompt_result.get("estimated_tokens",
                          estimate_tokens(prompt_result.get("prompt", ""))),
        "store":          req.store,
        "namespace":      req.namespace,
    }


@router.post("/assemble")
async def assemble_prompt(req: AssembleRequest, _: CurrentUser) -> dict[str, Any]:
    """Build an augmented prompt directly from provided hits (no search)."""
    hits = [h.model_dump() for h in req.hits]

    complexity = estimate_complexity(req.query)
    max_tokens = complexity["recommended_tokens"] if req.auto_budget else req.max_tokens

    organized = _select_and_organize(
        hits,
        max_tokens=max_tokens,
        text_field=req.text_field,
        use_highlight=req.use_highlight,
        organize=req.organize,
        source_field=req.source_field,
        date_field=req.date_field,
    )

    context_text = "\n\n".join(
        f"[{i+1}] From {c.get('metadata', {}).get(req.source_field, 'Document')}:\n{c.get('context_text', '')}"
        for i, c in enumerate(organized)
    )

    prompt_result = build_prompt(
        req.query, context_text,
        config={
            "template_name":  req.template,
            "chain_of_thought": req.chain_of_thought,
        },
    )

    sources = list(dict.fromkeys(
        c.get("metadata", {}).get(req.source_field, f"doc-{i}")
        for i, c in enumerate(organized)
    ))
    total_tokens = estimate_tokens(prompt_result.get("prompt", ""))

    return {
        "query":          req.query,
        "prompt":         prompt_result,
        "context_chunks": organized,
        "complexity":     complexity,
        "sources":        sources,
        "total_tokens":   total_tokens,
    }
