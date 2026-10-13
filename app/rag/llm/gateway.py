"""LLM gateway — the single choke point for every model call.

Every call to a provider (answer generation, structured output retries, agent
planning, query rewriting, vision extraction) goes through :func:`call_llm` or
:func:`stream_llm`, which wrap it in the same guards:

1. an OpenTelemetry span ``llm.call`` (provider, model, operation, tenant, tokens, cost)
2. the provider's circuit breaker ``llm:<provider>`` (F7) — open → :class:`LLMUnavailableError`
3. a per-tenant budget reservation (F24) — over limit → ``BudgetExceededError``
4. accounting after the call: token counts, cost ledger event, Prometheus usage,
   and booking the spend against the tenant budget.

Unconfigured providers (no API key) are rejected up front without touching the
breaker, so a missing key never opens the circuit for everyone else.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator

import httpx
from opentelemetry.trace import Status, StatusCode

from app.config import settings
from app.engine.circuit_breaker import registry as breaker_registry
from app.observability.metrics import record_llm_usage
from app.observability.tracing import get_tracer
from app.rag.cost import count_tokens, estimate_cost, get_budget, record_spend, record_usage, reserve
from app.rag.llm.base import LLMResponse
from app.rag.llm.registry import get_llm

tracer = get_tracer("rag.llm.gateway")

_KEY_SETTINGS = {
    "gemini": "gemini_api_key",
    "openai": "openai_api_key",
    "claude": "anthropic_api_key",
}


class LLMUnavailableError(RuntimeError):
    """The provider cannot be called right now (no key, or circuit open)."""


# Overloaded/unreachable providers often recover within seconds: retry those
# (not 4xx, not quota 429s that last hours) before counting a breaker failure.
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_S = 1.0
_TRANSIENT_STATUS = {500, 502, 503, 504}
_SECRET_PARAM = re.compile(r"([?&](?:key|api_key|apikey)=)[^&\s'\"]+", re.IGNORECASE)


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(exc, httpx.TransportError)


def redact(text: str) -> str:
    """Remove credentials that may appear in URLs inside error messages."""
    return _SECRET_PARAM.sub(r"\1***", text)


def is_configured(provider: str, config: dict[str, Any] | None = None) -> bool:
    attr = _KEY_SETTINGS.get(provider)
    if attr is None:
        return True  # stub and custom providers need no key
    return bool((config or {}).get("api_key") or getattr(settings, attr, ""))


def model_name(provider: str, config: dict[str, Any] | None = None) -> str:
    config = config or {}
    if config.get("model"):
        return str(config["model"])
    if provider == "gemini":
        return settings.gemini_model
    if provider == "stub":
        return "extractive-stub"
    return provider


def _breaker(provider: str):
    return breaker_registry.get(
        f"llm:{provider}",
        failure_threshold=settings.llm_breaker_threshold,
        recovery_timeout=settings.llm_breaker_recovery,
    )


def _budget_limit(tenant: str | None, explicit: float | None) -> float | None:
    """Explicit limit wins; else the tenant's registered budget; else the default."""
    if not tenant:
        return None
    if explicit is not None:
        return explicit if explicit > 0 else None
    budget = get_budget(tenant)
    return budget.limit if budget else settings.default_tenant_budget_usd


def _prompt_text(request: dict[str, Any]) -> str:
    return "\n".join(str(m.get("content", "")) for m in request.get("messages", []))


def _preflight(provider, request, config, tenant, budget_limit, span):
    """Configuration, breaker and budget checks shared by both call styles."""
    if not is_configured(provider, config):
        raise LLMUnavailableError(f"LLM provider '{provider}' is not configured (missing API key)")
    breaker = _breaker(provider)
    if not breaker.allow():
        span.set_attribute("llm.circuit", "open")
        raise LLMUnavailableError(f"circuit 'llm:{provider}' is open")
    model = model_name(provider, config)
    in_est = count_tokens(_prompt_text(request))
    limit = _budget_limit(tenant, budget_limit)
    if limit is not None:
        reserve(tenant, limit, estimate_cost(model, in_est, int(config.get("max_tokens", 512))))
    return breaker, model, in_est, limit


def _account(*, provider, model, operation, tenant, user_id, limit, in_tokens, out_tokens, span) -> float:
    cost = estimate_cost(model, in_tokens, out_tokens)
    record_llm_usage(provider, model, in_tokens, out_tokens, cost)
    record_usage(user_id=user_id, tenant=tenant or "default", model=model, operation=f"llm:{operation}",
                 input_tokens=in_tokens, output_tokens=out_tokens)
    if limit is not None:
        record_spend(tenant, limit, cost)
    span.set_attribute("llm.tokens.input", in_tokens)
    span.set_attribute("llm.tokens.output", out_tokens)
    span.set_attribute("llm.cost_usd", cost)
    return cost


def _start_span(provider: str, operation: str, tenant: str | None):
    span = tracer.start_span("llm.call")
    span.set_attribute("llm.provider", provider)
    span.set_attribute("llm.operation", operation)
    if tenant:
        span.set_attribute("tenant.id", tenant)
    return span


async def call_llm(
    provider: str,
    request: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    tenant: str | None = None,
    user_id: str = "",
    operation: str = "generate",
    budget_limit: float | None = None,
) -> LLMResponse:
    """Run one guarded, accounted LLM call. ``usage`` gains ``cost_usd``."""
    config = config or {}
    span = _start_span(provider, operation, tenant)
    try:
        breaker, model, in_est, limit = _preflight(provider, request, config, tenant, budget_limit, span)
        span.set_attribute("llm.model", model)
        llm = get_llm(provider)
        for attempt in range(TRANSIENT_RETRIES + 1):
            try:
                response = await llm.generate(request, config)
                break
            except Exception as exc:
                if attempt < TRANSIENT_RETRIES and is_transient(exc):
                    span.add_event("llm.retry", {"attempt": attempt + 1, "error": redact(str(exc))[:200]})
                    await asyncio.sleep(TRANSIENT_BACKOFF_S * 2 ** attempt)
                    continue
                breaker.record_failure()
                raise
        breaker.record_success()

        usage = response.usage or {}
        in_tokens = int(usage.get("input_tokens") or in_est)
        out_tokens = int(usage.get("output_tokens") or count_tokens(response.text))
        cost = _account(provider=provider, model=model, operation=operation, tenant=tenant,
                        user_id=user_id, limit=limit, in_tokens=in_tokens,
                        out_tokens=out_tokens, span=span)
        response.usage = {**usage, "input_tokens": in_tokens, "output_tokens": out_tokens,
                          "cost_usd": cost}
        return response
    except Exception as exc:
        message = redact(str(exc))
        span.add_event("exception", {"exception.type": type(exc).__name__, "exception.message": message})
        span.set_status(Status(StatusCode.ERROR, message))
        raise
    finally:
        span.end()


async def stream_llm(
    provider: str,
    request: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    tenant: str | None = None,
    user_id: str = "",
    operation: str = "generate_stream",
    budget_limit: float | None = None,
) -> AsyncIterator[str]:
    """Streaming variant of :func:`call_llm`; accounting happens when the stream ends."""
    config = config or {}
    span = _start_span(provider, operation, tenant)
    try:
        breaker, model, in_est, limit = _preflight(provider, request, config, tenant, budget_limit, span)
        span.set_attribute("llm.model", model)
        llm = get_llm(provider)
        collected: list[str] = []
        try:
            if hasattr(llm, "generate_stream"):
                async for token in llm.generate_stream(request, config):
                    collected.append(token)
                    yield token
            else:
                response = await llm.generate(request, config)
                collected.append(response.text)
                yield response.text
        except Exception:
            breaker.record_failure()
            raise
        breaker.record_success()
        _account(provider=provider, model=model, operation=operation, tenant=tenant,
                 user_id=user_id, limit=limit, in_tokens=in_est,
                 out_tokens=count_tokens("".join(collected)), span=span)
    except Exception as exc:
        message = redact(str(exc))
        span.add_event("exception", {"exception.type": type(exc).__name__, "exception.message": message})
        span.set_status(Status(StatusCode.ERROR, message))
        raise
    finally:
        span.end()
