"""Dynamic embedding model selector (F11).

Chooses the best available embedding model based on document signals and cost
preferences, without requiring callers to hard-code a model name.

Selection logic (in priority order)
-------------------------------------
1. Source code  →  all-mpnet-base-v2 (local; handles code well, free)
2. Non-English  →  para-multilingual (free) or text-embedding-3-small (API)
3. quality=best →  text-embedding-3-large → all-mpnet-base-v2
4. quality=free →  all-MiniLM-L6-v2 (smallest free model)
5. quality=cheap→  all-MiniLM-L6-v2
6. default      →  text-embedding-3-small (if API key) else all-MiniLM-L6-v2

The selector never raises — if the preferred model's dependency is missing it
cascades to the next viable option.
"""
from __future__ import annotations

from app.rag.embeddings.registry import _FACTORIES

# Shorthand: (model_name, dimension)
_Selection = tuple[str, int]

_MULTILINGUAL_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2",
    "embed-multilingual-v3.0",
}

_CODE_PREFERRED = ("all-mpnet-base-v2", 768)
_LARGE_FREE     = ("all-mpnet-base-v2", 768)
_SMALL_FREE     = ("all-MiniLM-L6-v2",  384)
_MULTILINGUAL_FREE = ("paraphrase-multilingual-MiniLM-L12-v2", 384)


def _available(model_name: str) -> bool:
    """Return True if the model is registered (its factory is present)."""
    return model_name in _FACTORIES


def select_model(
    *,
    content_type: str | None = None,
    language: str | None = None,
    quality: str = "balanced",
) -> _Selection:
    """Pick the best model given document signals.

    Parameters
    ----------
    content_type:
        One of ``"code"``, ``"text"``, ``"image"``, ``"structured"``.
    language:
        ISO-639-1 language code detected during quality assessment (F9).
        ``None`` or ``"en"`` → treated as English.
    quality:
        ``"best"``    — highest accuracy, API models preferred
        ``"balanced"``— default; prefer API if available, else free local
        ``"cheap"``   — prefer cheap API or free local
        ``"free"``    — never use paid API; local models only

    Returns
    -------
    (model_name, dimension) tuple ready to pass to ``get_embedder()``.
    """
    from app.config import settings
    has_openai = bool(settings.openai_api_key)
    has_cohere = bool(settings.cohere_api_key)
    is_multilingual = bool(language and language not in ("en", "und", None))

    # 1. Source-code content
    if content_type == "code":
        if _available("all-mpnet-base-v2"):
            return _CODE_PREFERRED
        return _SMALL_FREE

    # 2. Non-English
    if is_multilingual:
        if quality != "free" and has_openai and _available("text-embedding-3-small"):
            return ("text-embedding-3-small", 512)
        if _available("paraphrase-multilingual-MiniLM-L12-v2"):
            return _MULTILINGUAL_FREE
        return _SMALL_FREE

    # 3. Best quality
    if quality == "best":
        if has_openai and _available("text-embedding-3-large"):
            return ("text-embedding-3-large", 3072)
        if has_cohere and _available("embed-english-v3.0"):
            return ("embed-english-v3.0", 1024)
        if _available("all-mpnet-base-v2"):
            return _LARGE_FREE
        return _SMALL_FREE

    # 4. Free / cheap
    if quality in ("free", "cheap"):
        if _available("all-MiniLM-L6-v2"):
            return _SMALL_FREE
        return ("local-hash", 256)

    # 5. Balanced (default)
    if has_openai and _available("text-embedding-3-small"):
        return ("text-embedding-3-small", 512)
    if _available("all-MiniLM-L6-v2"):
        return _SMALL_FREE
    return ("local-hash", 256)


def model_info() -> dict[str, dict]:
    """Return a catalogue of all known models with capabilities and status."""
    from app.config import settings
    has_openai = bool(settings.openai_api_key)
    has_cohere = bool(settings.cohere_api_key)

    catalogue = [
        dict(name="local-hash", provider="builtin", dimension=256,
             requires_api_key=False, local=True, multilingual=False,
             best_for=["testing", "offline use", "development"],
             available=True),
        dict(name="all-MiniLM-L6-v2", provider="sentence-transformers",
             dimension=384, requires_api_key=False, local=True, multilingual=False,
             best_for=["English prose", "general search", "cost-sensitive use"],
             available=_available("all-MiniLM-L6-v2")),
        dict(name="all-mpnet-base-v2", provider="sentence-transformers",
             dimension=768, requires_api_key=False, local=True, multilingual=False,
             best_for=["English prose", "source code", "higher quality"],
             available=_available("all-mpnet-base-v2")),
        dict(name="paraphrase-multilingual-MiniLM-L12-v2",
             provider="sentence-transformers", dimension=384,
             requires_api_key=False, local=True, multilingual=True,
             best_for=["50+ languages", "multilingual search"],
             available=_available("paraphrase-multilingual-MiniLM-L12-v2")),
        dict(name="text-embedding-3-small", provider="openai",
             dimension=512, requires_api_key=True, local=False, multilingual=True,
             best_for=["balanced quality/cost", "multilingual", "production"],
             available=has_openai and _available("text-embedding-3-small")),
        dict(name="text-embedding-3-large", provider="openai",
             dimension=3072, requires_api_key=True, local=False, multilingual=True,
             best_for=["highest accuracy", "critical search", "RAG quality"],
             available=has_openai and _available("text-embedding-3-large")),
        dict(name="embed-english-v3.0", provider="cohere",
             dimension=1024, requires_api_key=True, local=False, multilingual=False,
             best_for=["English retrieval", "search-optimised"],
             available=has_cohere and _available("embed-english-v3.0")),
        dict(name="embed-multilingual-v3.0", provider="cohere",
             dimension=1024, requires_api_key=True, local=False, multilingual=True,
             best_for=["100+ languages", "multilingual retrieval"],
             available=has_cohere and _available("embed-multilingual-v3.0")),
    ]
    return {m["name"]: m for m in catalogue}
