"""Embedder interface (F11).

An Embedder maps text to a fixed-dimension dense vector. The default
implementation is a dependency-free deterministic *feature-hashing* embedder so
the whole pipeline runs and tests offline; real models (sentence-transformers,
OpenAI, Cohere) implement this same protocol and register themselves.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN = re.compile(r"\w+")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbedder:
    """Deterministic feature-hashing embedder.

    Each token is hashed to a bucket (with a signed contribution); the resulting
    vector is L2-normalized. Texts sharing vocabulary land near each other under
    cosine similarity — enough to exercise retrieval realistically without any
    model download. Not a substitute for a trained model in production.
    """

    def __init__(self, dimension: int = 256, name: str = "local-hash") -> None:
        self.dimension = dimension
        self.name = name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()
