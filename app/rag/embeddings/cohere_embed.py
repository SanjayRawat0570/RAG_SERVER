"""Cohere Embeddings adapter (F11).

Cohere's embedding models are specialised for retrieval (search_document /
search_query input types).  The ``input_type`` kwarg is set automatically based
on whether the caller passes ``is_query=True``.

Supported models
----------------
embed-english-v3.0         1024 dims  — English retrieval, best quality
embed-multilingual-v3.0     1024 dims  — 100+ language support
embed-english-light-v3.0    384 dims  — fast, lightweight English
embed-multilingual-light-v3.0 384 dims — fast multilingual

Requirements: pip install cohere  AND  COHERE_API_KEY env-var / settings.cohere_api_key
"""
from __future__ import annotations

_MODEL_DIMS: dict[str, int] = {
    "embed-english-v3.0":             1024,
    "embed-multilingual-v3.0":        1024,
    "embed-english-light-v3.0":       384,
    "embed-multilingual-light-v3.0":  384,
}


class CohereEmbedder:
    """Embeds texts via the Cohere Embed API."""

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        dimension: int | None = None,
        is_query: bool = False,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.dimension = dimension or _MODEL_DIMS.get(model_name, 1024)
        self._is_query = is_query

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        from app.config import settings
        api_key = settings.cohere_api_key
        if not api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not configured. "
                "Set it in your .env file or as an environment variable."
            )
        try:
            import cohere  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "cohere package is required. Install with: pip install cohere"
            ) from exc

        client = cohere.Client(api_key)
        input_type = "search_query" if self._is_query else "search_document"
        response = client.embed(
            texts=texts,
            model=self.model_name,
            input_type=input_type,
        )
        return [list(e) for e in response.embeddings]
