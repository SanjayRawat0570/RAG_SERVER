"""Per-tenant AI settings API — few-shot examples and vision ingestion.

GET  /tenant-settings/{tenant}   Current settings, effective examples and defaults
PUT  /tenant-settings/{tenant}   Update any subset of the settings
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.tenants.ai_settings import (
    DEFAULT_EXAMPLES,
    FewShotExample,
    effective_examples,
    get_settings,
    update_settings,
)

router = APIRouter(prefix="/tenant-settings", tags=["tenants"])


class SettingsUpdate(BaseModel):
    few_shot_enabled: bool | None = None
    # Send [] for "no examples", omit to keep, or set reset_examples to restore defaults.
    few_shot_examples: list[FewShotExample] | None = Field(default=None, max_length=8)
    reset_examples: bool = False
    vision_enabled: bool | None = None


def _view(tenant: str) -> dict[str, Any]:
    cfg = get_settings(tenant)
    return {
        "tenant": tenant,
        "few_shot_enabled": cfg.few_shot_enabled,
        "few_shot_custom": cfg.few_shot_examples is not None,
        "few_shot_examples": [e.model_dump() for e in effective_examples(tenant, override=True)],
        "default_examples": [e.model_dump() for e in DEFAULT_EXAMPLES],
        "vision_enabled": cfg.vision_enabled,
    }


@router.get("/{tenant}")
async def read_settings(tenant: str, _: CurrentUser) -> dict[str, Any]:
    return _view(tenant)


@router.put("/{tenant}")
async def write_settings(tenant: str, body: SettingsUpdate, _: CurrentUser) -> dict[str, Any]:
    changes: dict[str, Any] = body.model_dump(exclude_unset=True, exclude={"reset_examples"})
    if body.reset_examples:
        changes["few_shot_examples"] = None
    update_settings(tenant, **changes)
    return _view(tenant)
