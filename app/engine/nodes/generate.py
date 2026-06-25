"""Generate node — LLM answer generation with caching, cost & budgets (F16/F17/F24).

Consumes an ``augment`` output (messages + documents + citations) and produces
an answer via the configured provider. Default provider is the offline stub; set
``provider: "gemini"`` (with GEMINI_API_KEY) for real generation.

Performance & cost layers (all opt-in):
* exact response cache — identical messages return the cached answer (F17)
* semantic cache — a near-duplicate query reuses a prior answer (F17)
* token counting + cost estimation per call (F24)
* per-tenant budget enforcement before spending (F24)

Config::

    {
      "provider": "stub", "model": "gpt-4o", "temperature": 0.2, "max_tokens": 512,
      "cache": true, "cache_ttl": 300,
      "semantic_cache": {"enabled": true, "threshold": 0.95, "dimension": 256},
      "budget_key": "$.inputs.tenant", "budget_limit": 1.0
    }
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.config import settings
from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.observability.metrics import record_cache, record_llm_usage
from app.rag.cache import SemanticCache, get_cache
from app.rag.cache.cache import MISS
from app.rag.cost import count_tokens, estimate_cost, record_spend, reserve
from app.rag.embeddings import DEFAULT_DIMENSION, embed_texts
from app.rag.llm import get_llm


@register
class GenerateNode(Node):
    type = "generate"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        source = (
            ctx.resolve(self.config["input"])
            if "input" in self.config
            else (_single_upstream(upstream) if upstream else {})
        )
        request = self._build_request(ctx, source)
        provider_name = self.config.get("provider", settings.llm_provider)

        # 1) Exact response cache (F17).
        cache_name = self.config.get("cache_name", "llm-response")
        cache = get_cache(cache_name, ttl=float(self.config.get("cache_ttl", 300)))
        cache_key = self._cache_key(provider_name, request)
        if self.config.get("cache"):
            cached = cache.get(cache_key)
            record_cache(cache_name, cached is not MISS)
            if cached is not MISS:
                return {**cached, "cache_hit": True, "cache_type": "exact"}

        # 2) Semantic cache (F17) — reuse answers for near-duplicate queries.
        sem_cfg = self.config.get("semantic_cache") or {}
        query_vec = None
        sem_cache = None
        if sem_cfg.get("enabled") and request["query"]:
            dim = int(sem_cfg.get("dimension", DEFAULT_DIMENSION))
            sem_cache = SemanticCache(
                sem_cfg.get("name", cache_name), dim, float(sem_cfg.get("threshold", 0.95))
            )
            query_vec = embed_texts([request["query"]], dimension=dim)[0]
            hit = sem_cache.lookup(query_vec)
            record_cache(f"{cache_name}-semantic", hit is not None)
            if hit is not None:
                return {**hit, "cache_hit": True, "cache_type": "semantic"}

        # 3) Budget pre-check (F24): reject before spending if over limit.
        model = self.config.get("model") or self._provider_model(provider_name)
        prompt_text = "\n".join(m["content"] for m in request["messages"])
        in_tokens_est = count_tokens(prompt_text)
        budget_key = ctx.resolve(self.config.get("budget_key")) if "budget_key" in self.config else None
        budget_limit = float(self.config.get("budget_limit", 0.0))
        if budget_key and budget_limit > 0:
            est = estimate_cost(model, in_tokens_est, int(self.config.get("max_tokens", 512)))
            reserve(str(budget_key), budget_limit, est)

        # 4) Generate.
        llm = get_llm(provider_name)
        response = await llm.generate(request, self.config)

        # 5) Token counting + cost (F24).
        usage = response.usage or {}
        in_tokens = int(usage.get("input_tokens") or in_tokens_est)
        out_tokens = int(usage.get("output_tokens") or count_tokens(response.text))
        cost = estimate_cost(model, in_tokens, out_tokens)
        record_llm_usage(response.provider, model, in_tokens, out_tokens, cost)

        budget_info = None
        if budget_key and budget_limit > 0:
            b = record_spend(str(budget_key), budget_limit, cost)
            budget_info = {"key": b.key, "spent": b.spent, "limit": b.limit,
                           "remaining": round(b.limit - b.spent, 6)}

        result = {
            "answer": response.text,
            "provider": response.provider,
            "model": model,
            "finish_reason": response.finish_reason,
            "tokens": {"input": in_tokens, "output": out_tokens},
            "cost_usd": cost,
            "budget": budget_info,
            "citations": response.citations or [c["marker"] for c in request.get("citations", [])],
            "cache_hit": False,
            "cache_type": None,
        }

        # 6) Populate caches for next time.
        if self.config.get("cache"):
            cache.set(cache_key, result)
        if sem_cache is not None and query_vec is not None:
            sem_cache.put(request["query"], query_vec, result)
        return result

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _cache_key(provider: str, request: dict[str, Any]) -> str:
        payload = json.dumps({"p": provider, "m": request["messages"]}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _provider_model(provider: str) -> str:
        return settings.gemini_model if provider == "gemini" else "extractive-stub"

    def _build_request(self, ctx: ExecutionContext, source: Any) -> dict[str, Any]:
        if isinstance(source, dict) and "messages" in source:
            return {
                "messages": source.get("messages", []),
                "query": source.get("query", ""),
                "documents": source.get("documents", []),
                "citations": source.get("citations", []),
            }
        if isinstance(source, list):
            return {"messages": source, "query": "", "documents": [], "citations": []}
        prompt = str(source or ctx.resolve(self.config.get("prompt", "")))
        return {
            "messages": [{"role": "user", "content": prompt}],
            "query": str(ctx.resolve(self.config.get("query", ""))),
            "documents": [],
            "citations": [],
        }
