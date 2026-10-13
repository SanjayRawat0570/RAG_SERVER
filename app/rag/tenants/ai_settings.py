"""Per-tenant AI settings: few-shot examples and multimodal ingestion.

Keyed by the tenant/namespace string used by the RAG endpoints (e.g. ``acme``).
Persisted to ``<data_dir>/tenant_settings.json`` so the API and the MCP server
process see the same configuration.

Few-shot examples teach the answer format by demonstration: one answerable
question with a cited answer, one the context cannot answer ("I don't know").
Tenants can replace them with domain-specific pairs, or switch them off.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.rag.persist import data_file, load_json, save_json

_FILE = "tenant_settings.json"


class FewShotExample(BaseModel):
    question: str = Field(min_length=1)
    context: list[str] = Field(default_factory=list, description="Sources, numbered [1], [2], …")
    answer: str = Field(min_length=1)
    citations: list[int] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"
    answerable: bool = True


DEFAULT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        question="What is the daily meal limit when travelling?",
        context=[
            "(handbook.md) Employees can expense up to 40 USD per day for meals while travelling.",
            "(it.md) VPN access requires a hardware security key.",
        ],
        answer="Employees can expense up to 40 USD per day for meals while travelling [1].",
        citations=[1],
        confidence="high",
        answerable=True,
    ),
    FewShotExample(
        question="What is the office Wi-Fi password?",
        context=["(handbook.md) The office opens at 08:00 and closes at 19:00 on weekdays."],
        answer="I don't know based on the provided context.",
        citations=[],
        confidence="low",
        answerable=False,
    ),
]


class TenantAISettings(BaseModel):
    few_shot_enabled: bool = True
    # None → use DEFAULT_EXAMPLES; [] → enabled but no examples.
    few_shot_examples: list[FewShotExample] | None = None
    vision_enabled: bool = False


_cache: dict[str, TenantAISettings] | None = None
_cache_mtime: float | None = None


def _all() -> dict[str, TenantAISettings]:
    """Settings for every tenant, reloaded when another process rewrote the file."""
    global _cache, _cache_mtime
    path = data_file(_FILE)
    mtime = path.stat().st_mtime if path.exists() else None
    if _cache is None or mtime != _cache_mtime:
        raw = load_json(_FILE, {})
        _cache = {k: TenantAISettings.model_validate(v) for k, v in raw.items()}
        _cache_mtime = mtime
    return _cache


def _save(store: dict[str, TenantAISettings]) -> None:
    global _cache_mtime
    save_json(_FILE, {k: v.model_dump() for k, v in store.items()})
    _cache_mtime = data_file(_FILE).stat().st_mtime


def get_settings(tenant: str) -> TenantAISettings:
    return _all().get(tenant, TenantAISettings()).model_copy(deep=True)


def update_settings(tenant: str, **changes) -> TenantAISettings:
    current = get_settings(tenant)
    updated = TenantAISettings.model_validate({**current.model_dump(), **changes})
    store = _all()
    store[tenant] = updated
    _save(store)
    return updated


def effective_examples(tenant: str, override: bool | None = None) -> list[FewShotExample]:
    """Examples to put in the prompt; ``override`` forces them on/off for one request."""
    cfg = get_settings(tenant)
    enabled = cfg.few_shot_enabled if override is None else override
    if not enabled:
        return []
    return cfg.few_shot_examples if cfg.few_shot_examples is not None else list(DEFAULT_EXAMPLES)


def reset_settings() -> None:
    """Test helper: forget everything (in memory and on disk)."""
    global _cache
    _cache = {}
    _save(_cache)
