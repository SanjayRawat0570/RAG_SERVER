"""Sentence-Transformers adapter (F11) — free, local, offline.

When the ``sentence-transformers`` library is available it delegates to the
real HuggingFace model.  When it is absent (CI / offline environments) it
falls back to :class:`_TrigramEmbedder`: a character-trigram bag-of-words
model that produces L2-normalised 384-dim vectors with meaningful cosine
similarities (texts sharing words/stems share trigram positions and therefore
cluster together).

Pre-configured models
---------------------
all-MiniLM-L6-v2    384 dims  — fast, lightweight, good for English
all-mpnet-base-v2   768 dims  — higher quality, works well for code too
paraphrase-multilingual-MiniLM-L12-v2  384 dims  — 50+ languages
"""
from __future__ import annotations

import hashlib

# Known output dimensions for common models (avoids loading to inspect).
_MODEL_DIMS: dict[str, int] = {
    "all-MiniLM-L6-v2":                          384,
    "all-mpnet-base-v2":                          768,
    "paraphrase-multilingual-MiniLM-L12-v2":      384,
    "paraphrase-multilingual-mpnet-base-v2":      768,
    "all-distilroberta-v1":                       768,
    "multi-qa-MiniLM-L6-cos-v1":                 384,
}

# Module-level model cache — avoids reloading across multiple embedder instances.
_loaded_models: dict[str, object] = {}


class _TrigramEmbedder:
    """Character-trigram bag-of-words fallback (no external library needed).

    Each word in the input text contributes the SHA-256 hashes of its character
    3-grams as vector positions.  After L2 normalisation the resulting vectors
    have meaningful cosine similarities because texts with lexically related
    words share trigram positions.
    """

    def __init__(self, model_name: str, dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        sentences: list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        **_: object,
    ):
        import numpy as np

        result = []
        for text in sentences:
            vec = [0.0] * self.dim
            for word in text.lower().split():
                # Character trigrams (or the whole word if shorter than 3 chars)
                for i in range(max(1, len(word) - 2)):
                    gram = word[i : i + 3]
                    pos = int(hashlib.sha256(gram.encode()).hexdigest(), 16) % self.dim
                    vec[pos] += 1.0
            result.append(vec)

        arr = np.array(result, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms
        return arr


class SentenceTransformerEmbedder:
    """Sentence-Transformers model wrapped as an Embedder.

    Uses the real ``sentence-transformers`` library when available;
    falls back to :class:`_TrigramEmbedder` otherwise so that all
    tests and offline deployments work without a model download.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.name       = model_name
        self.dimension  = _MODEL_DIMS.get(model_name, 384)

    def _model(self):
        global _loaded_models
        if self.model_name not in _loaded_models:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
                _loaded_models[self.model_name] = SentenceTransformer(self.model_name)
                # Sync dimension from the actual loaded model.
                try:
                    model_obj = _loaded_models[self.model_name]
                    getter = getattr(
                        model_obj,
                        "get_embedding_dimension",
                        getattr(model_obj, "get_sentence_embedding_dimension", None),
                    )
                    actual_dim = getter() if getter else None
                    if actual_dim:
                        self.dimension = actual_dim
                except Exception:
                    pass
            except ImportError:
                # Offline / no library: use the trigram fallback.
                _loaded_models[self.model_name] = _TrigramEmbedder(
                    self.model_name, self.dimension
                )
        return _loaded_models[self.model_name]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(e) for e in embeddings]
