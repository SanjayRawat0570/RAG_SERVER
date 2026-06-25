"""Pre-built RAG workflows behind simple endpoints (for the frontend).

These assemble the same node graphs the example JSON files use, so the UI can
POST a document or a question without constructing workflow JSON itself. The
heavy lifting still runs through the orchestration engine (tracing, metrics,
retries, caching, budgets all apply).
"""
from __future__ import annotations

from typing import Any

from app.config import settings
from app.models.workflow import WorkflowDef

STORE = "kb"
DIM = 256


def build_index_workflow() -> WorkflowDef:
    return WorkflowDef(
        name="ui_index",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "ingest", "type": "ingest",
             "config": {"text": "$.inputs.text", "filename": "$.inputs.filename",
                        "format": "text",
                        "metadata": {"tenant": "$.inputs.tenant"}}},
            {"id": "chunk", "type": "chunk",
             "config": {"strategy": "structure", "chunk_size": 512, "size_unit": "tokens"}},
            {"id": "embed", "type": "embed", "config": {"dimension": DIM}},
            {"id": "upsert", "type": "upsert",
             "config": {"store": STORE, "namespace": "$.inputs.tenant", "dimension": DIM}},
            {"id": "out", "type": "output", "config": {"value": "$.upsert"}},
        ],
        edges=[
            {"source": "in", "target": "ingest"},
            {"source": "ingest", "target": "chunk"},
            {"source": "chunk", "target": "embed"},
            {"source": "embed", "target": "upsert"},
            {"source": "upsert", "target": "out"},
        ],
    )


def build_ask_workflow(provider: str) -> WorkflowDef:
    return WorkflowDef(
        name="ui_ask",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "qp", "type": "query_process", "config": {"query": "$.inputs.question"}},
            {"id": "dense", "type": "vector_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.normalized", "dimension": DIM, "top_k": 5}},
            {"id": "sparse", "type": "keyword_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.expanded_query", "dimension": DIM, "top_k": 5}},
            {"id": "fuse", "type": "merge", "config": {"strategy": "rrf", "top_n": 5}},
            {"id": "rerank", "type": "rerank",
             "config": {"method": "cross_encoder", "query": "$.qp.normalized", "top_n": 3}},
            {"id": "augment", "type": "augment",
             "config": {"query": "$.qp.normalized", "max_context_tokens": 600}},
            {"id": "generate", "type": "generate",
             "config": {"provider": provider, "query": "$.qp.normalized", "cache": True, "max_tokens": 2048},
             "retry": {"max_attempts": 2},
             "fallback": {"answer": "Generation is temporarily unavailable.",
                          "citations": [], "provider": provider, "cache_hit": False}},
            {"id": "out", "type": "output", "config": {"value": {
                "intent": "$.qp.intent",
                "answer": "$.generate.answer",
                "citations": "$.augment.citations",
                "provider": "$.generate.provider",
                "cost_usd": "$.generate.cost_usd",
                "cache_hit": "$.generate.cache_hit"}}},
        ],
        edges=[
            {"source": "in", "target": "qp"},
            {"source": "qp", "target": "dense"},
            {"source": "qp", "target": "sparse"},
            {"source": "dense", "target": "fuse"},
            {"source": "sparse", "target": "fuse"},
            {"source": "fuse", "target": "rerank"},
            {"source": "rerank", "target": "augment"},
            {"source": "augment", "target": "generate"},
            {"source": "generate", "target": "out"},
        ],
    )


def build_decompose_workflow() -> WorkflowDef:
    """Single-node workflow that decomposes a question into sub-questions."""
    return WorkflowDef(
        name="decompose",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "decompose", "type": "decompose",
             "config": {"question": "$.inputs.question", "max_sub_questions": 5}},
            {"id": "out", "type": "output", "config": {"value": "$.decompose"}},
        ],
        edges=[
            {"source": "in", "target": "decompose"},
            {"source": "decompose", "target": "out"},
        ],
    )


def build_synthesize_workflow() -> WorkflowDef:
    """Single-node workflow that synthesizes multiple sub-answers into one."""
    return WorkflowDef(
        name="synthesize",
        nodes=[
            {"id": "in", "type": "input"},
            {"id": "synth", "type": "synthesize",
             "config": {"answers": "$.inputs.sub_answers",
                        "original_question": "$.inputs.question"}},
            {"id": "out", "type": "output", "config": {"value": "$.synth"}},
        ],
        edges=[
            {"source": "in", "target": "synth"},
            {"source": "synth", "target": "out"},
        ],
    )


def build_doc_type_router_workflow() -> WorkflowDef:
    """F5 Example 1: Document Type Routing — classify file extension → route to parser."""
    return WorkflowDef(
        name="doc_type_router",
        description="Route a document to the correct parser based on file extension.",
        nodes=[
            {"id": "in", "type": "input"},
            # Classify by file extension
            {"id": "classify", "type": "classify", "config": {
                "input": "$.inputs.filename",
                "rules": [
                    {"category": "pdf",
                     "label": "PDF document → pypdf",
                     "when": {"left": "$.inputs.filename", "op": "endswith", "right": ".pdf"},
                     "confidence": 0.99},
                    {"category": "docx",
                     "label": "Word document → python-docx",
                     "when": {"or": [
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".docx"},
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".doc"},
                     ]}, "confidence": 0.99},
                    {"category": "image",
                     "label": "Image → OCR (tesseract)",
                     "when": {"or": [
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".png"},
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".jpg"},
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".jpeg"},
                     ]}, "confidence": 0.99},
                    {"category": "html",
                     "label": "HTML page → BeautifulSoup",
                     "when": {"or": [
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".html"},
                         {"left": "$.inputs.filename", "op": "endswith", "right": ".htm"},
                     ]}, "confidence": 0.99},
                    {"category": "json",
                     "label": "JSON data → json.loads",
                     "when": {"left": "$.inputs.filename", "op": "endswith", "right": ".json"},
                     "confidence": 0.99},
                ],
                "default": "text",
                "default_confidence": 0.7,
            }},
            # Route to the correct parser branch
            {"id": "router", "type": "switch", "config": {
                "input": "$.classify.category",
                "cases": [
                    {"label": "pdf",   "when": {"left": "$.classify.category", "op": "==", "right": "pdf"}},
                    {"label": "docx",  "when": {"left": "$.classify.category", "op": "==", "right": "docx"}},
                    {"label": "image", "when": {"left": "$.classify.category", "op": "==", "right": "image"}},
                    {"label": "html",  "when": {"left": "$.classify.category", "op": "==", "right": "html"}},
                ],
                "default": "text",
            }},
            # Parser info per branch
            {"id": "pdf_parse",   "type": "processing", "config": {"operation": "set", "value": {"parser": "pypdf",        "method": "pdf_extract"}}},
            {"id": "word_parse",  "type": "processing", "config": {"operation": "set", "value": {"parser": "python-docx",  "method": "docx_extract"}}},
            {"id": "image_parse", "type": "processing", "config": {"operation": "set", "value": {"parser": "tesseract",    "method": "ocr"}}},
            {"id": "html_parse",  "type": "processing", "config": {"operation": "set", "value": {"parser": "beautifulsoup","method": "html_extract"}}},
            {"id": "text_parse",  "type": "processing", "config": {"operation": "set", "value": {"parser": "generic",      "method": "text_extract"}}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in",       "target": "classify"},
            {"source": "classify", "target": "router"},
            {"source": "router", "target": "pdf_parse",   "condition": {"left": "$.router.case", "op": "==", "right": "pdf"}},
            {"source": "router", "target": "word_parse",  "condition": {"left": "$.router.case", "op": "==", "right": "docx"}},
            {"source": "router", "target": "image_parse", "condition": {"left": "$.router.case", "op": "==", "right": "image"}},
            {"source": "router", "target": "html_parse",  "condition": {"left": "$.router.case", "op": "==", "right": "html"}},
            {"source": "router", "target": "text_parse",  "condition": {"left": "$.router.case", "op": "==", "right": "text"}},
            {"source": "pdf_parse",   "target": "out"},
            {"source": "word_parse",  "target": "out"},
            {"source": "image_parse", "target": "out"},
            {"source": "html_parse",  "target": "out"},
            {"source": "text_parse",  "target": "out"},
        ],
    )


def build_search_strategy_workflow() -> WorkflowDef:
    """F5 Example 2: Search Strategy Selection — classify query → route to search method."""
    return WorkflowDef(
        name="search_strategy_router",
        description="Classify a query and route to the best-fit search strategy.",
        nodes=[
            {"id": "in", "type": "input"},
            # Detect intent (question / command / keyword / lookup)
            {"id": "qp", "type": "query_process", "config": {"query": "$.inputs.question"}},
            # Classify search strategy
            {"id": "classify", "type": "classify", "config": {
                "input": "$.inputs.question",
                "rules": [
                    # Quoted text → exact phrase → keyword search
                    {"category": "keyword",
                     "label": "Exact quoted phrase → keyword search",
                     "when": {"left": "$.inputs.question", "op": "regex", "right": '"[^"]+"'},
                     "confidence": 0.95},
                    # Command intent ("find", "list", "show" …) → keyword search
                    {"category": "keyword",
                     "label": "Command → keyword search",
                     "when": {"left": "$.qp.intent", "op": "==", "right": "command"},
                     "confidence": 0.80},
                    # Conceptual question → semantic (dense) search
                    {"category": "semantic",
                     "label": "Question → semantic search",
                     "when": {"left": "$.qp.intent", "op": "==", "right": "question"},
                     "confidence": 0.80},
                    # Lookup (entity / proper noun) → entity search
                    {"category": "entity",
                     "label": "Lookup → entity search",
                     "when": {"left": "$.qp.intent", "op": "==", "right": "lookup"},
                     "confidence": 0.80},
                ],
                "default": "hybrid",
                "default_confidence": 0.65,
            }},
            # Route
            {"id": "router", "type": "switch", "config": {
                "input": "$.classify.category",
                "cases": [
                    {"label": "semantic", "when": {"left": "$.classify.category", "op": "==", "right": "semantic"}},
                    {"label": "keyword",  "when": {"left": "$.classify.category", "op": "==", "right": "keyword"}},
                    {"label": "entity",   "when": {"left": "$.classify.category", "op": "==", "right": "entity"}},
                ],
                "default": "hybrid",
            }},
            # Branch actions (describe which search node to invoke)
            {"id": "use_semantic", "type": "processing", "config": {"operation": "set", "value": {"strategy": "semantic", "node": "vector_search"}}},
            {"id": "use_keyword",  "type": "processing", "config": {"operation": "set", "value": {"strategy": "keyword",  "node": "keyword_search"}}},
            {"id": "use_entity",   "type": "processing", "config": {"operation": "set", "value": {"strategy": "entity",   "node": "entity_search"}}},
            {"id": "use_hybrid",   "type": "processing", "config": {"operation": "set", "value": {"strategy": "hybrid",   "node": "rrf_merge"}}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in",       "target": "qp"},
            {"source": "qp",       "target": "classify"},
            {"source": "classify", "target": "router"},
            {"source": "router", "target": "use_semantic", "condition": {"left": "$.router.case", "op": "==", "right": "semantic"}},
            {"source": "router", "target": "use_keyword",  "condition": {"left": "$.router.case", "op": "==", "right": "keyword"}},
            {"source": "router", "target": "use_entity",   "condition": {"left": "$.router.case", "op": "==", "right": "entity"}},
            {"source": "router", "target": "use_hybrid",   "condition": {"left": "$.router.case", "op": "==", "right": "hybrid"}},
            {"source": "use_semantic", "target": "out"},
            {"source": "use_keyword",  "target": "out"},
            {"source": "use_entity",   "target": "out"},
            {"source": "use_hybrid",   "target": "out"},
        ],
    )


def build_confidence_router_workflow() -> WorkflowDef:
    """F5 Example 3: Response Quality — classify top-hit score → route response handling."""
    return WorkflowDef(
        name="confidence_router",
        description=(
            "Run a vector search, assess confidence from the top hit score, "
            "and route to the appropriate response action."
        ),
        nodes=[
            {"id": "in",    "type": "input"},
            {"id": "qp",    "type": "query_process", "config": {"query": "$.inputs.question"}},
            # Search to gather evidence
            {"id": "search", "type": "vector_search", "config": {
                "store": STORE, "namespace": "$.inputs.tenant",
                "query": "$.qp.normalized", "dimension": DIM, "top_k": 3,
            }},
            # Classify confidence from the top hit's score
            # $.search.0.score resolves via numeric list indexing in ExecutionContext
            {"id": "classify", "type": "classify", "config": {
                "input": "$.search",
                "rules": [
                    {"category": "high",
                     "label": ">80% confidence — return immediately",
                     "when": {"left": "$.search.0.score", "op": ">", "right": 0.80},
                     "confidence": 0.95},
                    {"category": "medium",
                     "label": ">60% confidence — return with citations",
                     "when": {"left": "$.search.0.score", "op": ">", "right": 0.60},
                     "confidence": 0.75},
                    {"category": "low",
                     "label": ">40% confidence — expand context and retry",
                     "when": {"left": "$.search.0.score", "op": ">", "right": 0.40},
                     "confidence": 0.50},
                ],
                "default": "none",
                "default_confidence": 0.20,
            }},
            # Route based on confidence level
            {"id": "router", "type": "switch", "config": {
                "input": "$.classify.category",
                "cases": [
                    {"label": "high",   "when": {"left": "$.classify.category", "op": "==", "right": "high"}},
                    {"label": "medium", "when": {"left": "$.classify.category", "op": "==", "right": "medium"}},
                    {"label": "low",    "when": {"left": "$.classify.category", "op": "==", "right": "low"}},
                ],
                "default": "none",
            }},
            # Response actions
            {"id": "return_direct",  "type": "processing", "config": {"operation": "set", "value": {"action": "return_immediately",     "cite_sources": False, "retry": False}}},
            {"id": "return_cited",   "type": "processing", "config": {"operation": "set", "value": {"action": "return_with_citations",  "cite_sources": True,  "retry": False}}},
            {"id": "expand_retry",   "type": "processing", "config": {"operation": "set", "value": {"action": "expand_context_retry",   "cite_sources": True,  "retry": True}}},
            {"id": "cannot_answer",  "type": "processing", "config": {"operation": "set", "value": {"action": "cannot_answer_confidently", "cite_sources": False, "retry": False, "message": "Insufficient confidence to answer."}}},
            {"id": "out", "type": "output"},
        ],
        edges=[
            {"source": "in",       "target": "qp"},
            {"source": "qp",       "target": "search"},
            {"source": "search",   "target": "classify"},
            {"source": "classify", "target": "router"},
            {"source": "router", "target": "return_direct",  "condition": {"left": "$.router.case", "op": "==", "right": "high"}},
            {"source": "router", "target": "return_cited",   "condition": {"left": "$.router.case", "op": "==", "right": "medium"}},
            {"source": "router", "target": "expand_retry",   "condition": {"left": "$.router.case", "op": "==", "right": "low"}},
            {"source": "router", "target": "cannot_answer",  "condition": {"left": "$.router.case", "op": "==", "right": "none"}},
            {"source": "return_direct",  "target": "out"},
            {"source": "return_cited",   "target": "out"},
            {"source": "expand_retry",   "target": "out"},
            {"source": "cannot_answer",  "target": "out"},
        ],
    )


def build_semantic_only_workflow() -> WorkflowDef:
    """F7 Scenario 1 primary: dense vector search only (no keyword fallback)."""
    return WorkflowDef(
        name="semantic_only",
        nodes=[
            {"id": "in",     "type": "input"},
            {"id": "qp",     "type": "query_process", "config": {"query": "$.inputs.question"}},
            {"id": "dense",  "type": "vector_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.normalized", "dimension": DIM, "top_k": 5}},
            {"id": "out",    "type": "output", "config": {"value": "$.dense"}},
        ],
        edges=[
            {"source": "in",    "target": "qp"},
            {"source": "qp",    "target": "dense"},
            {"source": "dense", "target": "out"},
        ],
    )


def build_keyword_only_workflow() -> WorkflowDef:
    """F7 Scenario 1 fallback 1: BM25 keyword search (no vector DB needed)."""
    return WorkflowDef(
        name="keyword_only",
        nodes=[
            {"id": "in",    "type": "input"},
            {"id": "qp",    "type": "query_process", "config": {"query": "$.inputs.question"}},
            {"id": "bm25",  "type": "keyword_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.expanded_query", "dimension": DIM, "top_k": 5}},
            {"id": "out",   "type": "output", "config": {"value": "$.bm25"}},
        ],
        edges=[
            {"source": "in",   "target": "qp"},
            {"source": "qp",   "target": "bm25"},
            {"source": "bm25", "target": "out"},
        ],
    )


def build_entity_only_workflow() -> WorkflowDef:
    """F7 Scenario 1 fallback 2: entity search (pattern/name matching)."""
    return WorkflowDef(
        name="entity_only",
        nodes=[
            {"id": "in",     "type": "input"},
            {"id": "qp",     "type": "query_process", "config": {"query": "$.inputs.question"}},
            {"id": "entity", "type": "entity_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.normalized", "dimension": DIM, "top_k": 5}},
            {"id": "out",    "type": "output", "config": {"value": "$.entity"}},
        ],
        edges=[
            {"source": "in",     "target": "qp"},
            {"source": "qp",     "target": "entity"},
            {"source": "entity", "target": "out"},
        ],
    )


def build_expanded_search_workflow() -> WorkflowDef:
    """F7 Scenario 3 fallback: search with query-processor's expanded_query (synonyms)."""
    return WorkflowDef(
        name="expanded_search",
        nodes=[
            {"id": "in",    "type": "input"},
            {"id": "qp",    "type": "query_process", "config": {"query": "$.inputs.question"}},
            # Use the expanded query (with added synonyms/related terms) for vector search
            {"id": "dense", "type": "vector_search",
             "config": {"store": STORE, "namespace": "$.inputs.tenant",
                        "query": "$.qp.expanded_query", "dimension": DIM, "top_k": 5}},
            {"id": "out",   "type": "output", "config": {"value": "$.dense"}},
        ],
        edges=[
            {"source": "in",    "target": "qp"},
            {"source": "qp",    "target": "dense"},
            {"source": "dense", "target": "out"},
        ],
    )


def default_provider() -> str:
    return settings.llm_provider
