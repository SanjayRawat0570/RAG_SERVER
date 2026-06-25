"""Model selector — pick the right LLM for the job (F16).

Selection criteria
------------------
quality     : "best" | "balanced" | "fast" | "free"
complexity  : "simple" | "moderate" | "complex"   (from F15 complexity estimator)
provider    : force a specific provider name (overrides everything)

Selection matrix
----------------
                  simple          moderate        complex
best        claude-opus     claude-sonnet   claude-opus
balanced    claude-haiku    claude-sonnet   claude-sonnet
fast        gemini-flash    gpt-4o-mini     gpt-4o-mini
free        stub            stub/gemini     gemini

Priority: explicit provider > quality+complexity matrix > default (gemini → stub)
"""
from __future__ import annotations

from app.config import settings

# (provider, model_hint)
_MATRIX: dict[str, dict[str, tuple[str, str]]] = {
    "best": {
        "simple":   ("claude",  "opus"),
        "moderate": ("claude",  "sonnet"),
        "complex":  ("claude",  "opus"),
    },
    "balanced": {
        "simple":   ("claude",  "haiku"),
        "moderate": ("claude",  "sonnet"),
        "complex":  ("claude",  "sonnet"),
    },
    "fast": {
        "simple":   ("gemini",  "gemini-2.5-flash"),
        "moderate": ("openai",  "gpt-4o-mini"),
        "complex":  ("openai",  "gpt-4o-mini"),
    },
    "free": {
        "simple":   ("gemini",  "gemini-2.5-flash"),
        "moderate": ("gemini",  "gemini-2.5-flash"),
        "complex":  ("gemini",  "gemini-2.5-flash"),
    },
}

_PROVIDER_AVAILABLE: dict[str, str] = {
    "openai": "openai_api_key",
    "claude": "anthropic_api_key",
    "gemini": "gemini_api_key",
    "stub":   None,
}


def _has_key(provider: str) -> bool:
    attr = _PROVIDER_AVAILABLE.get(provider)
    if attr is None:
        return True     # stub always available
    return bool(getattr(settings, attr, ""))


def select_model(
    *,
    quality: str = "balanced",
    complexity: str = "moderate",
    provider: str | None = None,
) -> dict[str, str]:
    """Return the recommended ``{"provider": ..., "model": ...}`` dict.

    Falls back down the priority chain when an API key is not present:
    best/balanced → claude → gemini → stub
    fast          → openai → gemini → stub
    free          → gemini → stub
    """
    if provider:
        return {"provider": provider, "model": ""}

    row = _MATRIX.get(quality, _MATRIX["balanced"])
    prov, model = row.get(complexity, row["moderate"])

    # Try the recommended provider; fall back if the key is missing.
    fallback_chain = [prov, "gemini", "stub"]
    for candidate in fallback_chain:
        if _has_key(candidate):
            if candidate != prov:
                # switched providers — reset model hint to provider default
                model = ""
            return {"provider": candidate, "model": model}

    return {"provider": "stub", "model": ""}


def model_catalogue() -> list[dict]:
    """Return all known models with provider, tier, cost, and availability."""
    from app.rag.cost.pricing import PRICING
    return [
        {
            "provider":   "stub",
            "model":      "extractive-stub",
            "tier":       "free",
            "available":  True,
            "cost_per_1k_input":  0.0,
            "cost_per_1k_output": 0.0,
            "description": "Offline extractive stub — no API key, deterministic.",
        },
        {
            "provider":   "gemini",
            "model":      "gemini-2.5-flash",
            "tier":       "free",
            "available":  _has_key("gemini"),
            "cost_per_1k_input":  0.0,
            "cost_per_1k_output": 0.0,
            "description": "Google Gemini free tier — fast, generous rate limits.",
        },
        {
            "provider":   "openai",
            "model":      "gpt-4o-mini",
            "tier":       "fast",
            "available":  _has_key("openai"),
            "cost_per_1k_input":  PRICING.get("gpt-4o-mini", (0, 0))[0],
            "cost_per_1k_output": PRICING.get("gpt-4o-mini", (0, 0))[1],
            "description": "OpenAI GPT-4o-mini — fast and cheap, great for simple queries.",
        },
        {
            "provider":   "openai",
            "model":      "gpt-4o",
            "tier":       "best",
            "available":  _has_key("openai"),
            "cost_per_1k_input":  PRICING.get("gpt-4o", (0, 0))[0],
            "cost_per_1k_output": PRICING.get("gpt-4o", (0, 0))[1],
            "description": "OpenAI GPT-4o — high quality, higher cost.",
        },
        {
            "provider":   "claude",
            "model":      "claude-haiku-4-5-20251001",
            "tier":       "fast",
            "available":  _has_key("claude"),
            "cost_per_1k_input":  0.0008,
            "cost_per_1k_output": 0.0025,
            "description": "Claude Haiku — fastest Claude model, ideal for simple queries.",
        },
        {
            "provider":   "claude",
            "model":      "claude-sonnet-4-6",
            "tier":       "balanced",
            "available":  _has_key("claude"),
            "cost_per_1k_input":  0.003,
            "cost_per_1k_output": 0.015,
            "description": "Claude Sonnet — best price/quality balance for most queries.",
        },
        {
            "provider":   "claude",
            "model":      "claude-opus-4-8",
            "tier":       "best",
            "available":  _has_key("claude"),
            "cost_per_1k_input":  0.015,
            "cost_per_1k_output": 0.075,
            "description": "Claude Opus — highest reasoning quality, use for complex queries.",
        },
    ]
