"""Generate node — LLM answer generation with caching, cost & budgets (F16/F17/F24).

Consumes an ``augment`` output (messages + documents + citations) and produces
an answer via the configured provider. Default provider is the offline stub; set
``provider: "gemini"`` (with GEMINI_API_KEY) for real generation.

Performance & cost layers:
* exact response cache — identical messages return the cached answer (F17)
* semantic cache — a near-duplicate query reuses a prior answer (F17)
* every call goes through the LLM gateway: circuit breaker, per-tenant budget
  guard, token counting + cost ledger, and an OTel span (F7/F24/F8)
* ``structured: true`` — validated JSON answer {answer, citations, confidence,
  answerable} with one repair retry and an extractive fallback

Config::

    {
      "provider": "stub", "model": "gpt-4o", "temperature": 0.2, "max_tokens": 512,
      "structured": true,
      "cache": true, "cache_ttl": 300,
      "semantic_cache": {"enabled": true, "threshold": 0.95, "dimension": 256},
      "tenant": "$.inputs.tenant",          # budget key; tenant default limit applies
      "budget_key": "$.inputs.tenant", "budget_limit": 1.0   # explicit limit
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
from app.observability.metrics import record_cache
from app.rag.cache import SemanticCache, get_cache
from app.rag.cache.cache import MISS
from app.rag.cost import get_budget
from app.rag.embeddings import DEFAULT_DIMENSION, embed_texts
from app.rag.llm.gateway import call_llm
from app.rag.llm.structured import generate_structured


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
        tenant, budget_limit = self._budget(ctx)
        # Tenant is part of the key: one tenant never receives another's cached answer.
        cache_key = self._cache_key(provider_name, request, bool(self.config.get("structured")),
                                    tenant)
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
            sem_name = sem_cfg.get("name", cache_name)
            sem_cache = SemanticCache(
                f"{sem_name}:{tenant}" if tenant else sem_name, dim,
                float(sem_cfg.get("threshold", 0.95)),
            )
            query_vec = embed_texts([request["query"]], dimension=dim)[0]
            hit = sem_cache.lookup(query_vec)
            record_cache(f"{cache_name}-semantic", hit is not None)
            if hit is not None:
                return {**hit, "cache_hit": True, "cache_type": "semantic"}

        # 3) Generate through the LLM gateway: circuit breaker, budget guard,
        #    cost ledger, metrics and an OTel span all apply (F7/F24/F8).
        if self.config.get("structured"):
            result = await self._generate_structured(provider_name, request, tenant, budget_limit)
        else:
            response = await call_llm(provider_name, request, self.config,
                                      tenant=tenant, budget_limit=budget_limit)
            result = self._plain_result(response, request)

        budget = get_budget(tenant) if tenant else None
        result["budget"] = (
            {"key": budget.key, "spent": budget.spent, "limit": budget.limit,
             "remaining": round(budget.limit - budget.spent, 6)}
            if budget else None
        )
        result["cache_hit"] = False
        result["cache_type"] = None

        # 6) Populate caches for next time — but never with the extractive fallback:
        #    a provider outage would otherwise keep serving fallback answers for the
        #    whole TTL after the provider has recovered.
        if result.get("provider") != "extractive-fallback":
            if self.config.get("cache"):
                cache.set(cache_key, result)
            if sem_cache is not None and query_vec is not None:
                sem_cache.put(request["query"], query_vec, result)
        return result

    # -- helpers -----------------------------------------------------------

    def _budget(self, ctx: ExecutionContext) -> tuple[str | None, float | None]:
        """Budget key (the tenant) and explicit limit; no key → no budget guard."""
        key = None
        for field in ("tenant", "budget_key"):
            if field in self.config:
                key = ctx.resolve(self.config[field]) or key
        limit = float(self.config["budget_limit"]) if "budget_limit" in self.config else None
        return (str(key) if key else None), limit

    def _plain_result(self, response, request: dict[str, Any]) -> dict[str, Any]:
        usage = response.usage or {}
        return {
            "answer": response.text,
            "provider": response.provider,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "tokens": {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)},
            "cost_usd": usage.get("cost_usd", 0.0),
            "citations": response.citations or [c["marker"] for c in request.get("citations", [])],
        }

    async def _generate_structured(self, provider: str, request: dict[str, Any],
                                   tenant: str | None, budget_limit: float | None) -> dict[str, Any]:
        out = await generate_structured(provider, request, self.config,
                                        tenant=tenant, budget_limit=budget_limit)
        response = out.pop("response")
        usage = (response.usage or {}) if response else {}
        sources = {c.get("marker"): c.get("source") for c in request.get("citations", [])}
        return {
            **out,
            "citations": out["cited_markers"],
            "cited_sources": list(dict.fromkeys(
                sources[m] for m in out["cited_markers"] if sources.get(m))),
            "provider": response.provider if response else "extractive-fallback",
            "model": response.model if response else "extractive-stub",
            "finish_reason": response.finish_reason if response else "fallback",
            "tokens": {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)},
            "cost_usd": usage.get("cost_usd", 0.0),
        }

    @staticmethod
    def _cache_key(provider: str, request: dict[str, Any], structured: bool = False,
                   tenant: str | None = None) -> str:
        payload = json.dumps({"p": provider, "m": request["messages"], "s": structured,
                              "t": tenant}, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

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
