"""OpenAI Embeddings adapter (F11).

Uses the openai Python SDK synchronously.  The EmbedNode wraps the call in
asyncio.to_thread so the async event loop is never blocked.

Supported models
----------------
text-embedding-3-small   512 dims   (default)  — fast, cheap, good quality
text-embedding-3-large   3072 dims             — best quality, higher cost
text-embedding-ada-002   1536 dims             — legacy, still widely deployed

Dimension override is supported for text-embedding-3-* via the OpenAI
``dimensions`` parameter (truncation at the API level).

Requirements: OPENAI_API_KEY env-var or settings.openai_api_key must be set.
"""
from __future__ import annotations

_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 512,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_SUPPORTS_DIMENSIONS = {"text-embedding-3-small", "text-embedding-3-large"}


class OpenAIEmbedder:
    """Embeds texts via the OpenAI Embeddings API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        dimension: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.dimension = dimension or _MODEL_DIMS.get(model_name, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from app.config import settings
        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Set it in your .env file or as an environment variable."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required. Install with: pip install openai"
            ) from exc

        client = OpenAI(api_key=api_key)
        kwargs: dict = {"input": texts, "model": self.model_name}
        if self.model_name in _SUPPORTS_DIMENSIONS:
            kwargs["dimensions"] = self.dimension
        resp = client.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]
