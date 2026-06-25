"""Token counting and cost estimation (F24).

A pricing table maps model -> (USD per 1K input tokens, USD per 1K output
tokens). The offline default models cost nothing; representative paid prices are
included so cost controls and model-choice trade-offs can be exercised. Token
counts reuse the lightweight char-based estimator (a real tokenizer slots in
behind ``count_tokens``).
"""
from __future__ import annotations

from app.rag.models import estimate_tokens

# USD per 1,000 tokens: (input, output). Indicative values for demonstration.
PRICING: dict[str, tuple[float, float]] = {
    "extractive-stub":  (0.0,     0.0),
    "gemini-2.5-flash": (0.0,     0.0),   # free tier
    "gemini-2.0-flash": (0.0,     0.0),   # free tier
    "gemini-1.5-flash": (0.0,     0.0),   # legacy
    "gemini-1.5-pro":   (0.00125, 0.005),
    "gemini-2.5-pro":   (0.00125, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "claude-3-5-sonnet": (0.003, 0.015),
}

DEFAULT_PRICE = (0.0, 0.0)


def count_tokens(text: str) -> int:
    return estimate_tokens(text)


def price_for(model: str) -> tuple[float, float]:
    return PRICING.get(model, DEFAULT_PRICE)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = price_for(model)
    return round((input_tokens / 1000.0) * inp + (output_tokens / 1000.0) * out, 6)
