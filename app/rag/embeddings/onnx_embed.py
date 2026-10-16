"""Local semantic embeddings on ONNX Runtime via fastembed — free, offline, no torch.

``bge-small-en-v1.5`` (384 dims, ~64 MB) runs on CPU through ONNX Runtime. Unlike
the hash embedder it understands paraphrase: "minimum password length" is close
to "at least 14 characters long". BGE models expect queries to carry an
instruction prefix, so queries go through :meth:`OnnxEmbedder.embed_queries`.

The model files live in ``settings.models_dir``. They are fetched once (with
TLS verified against the OS trust store, like the LLM providers) by::

    python -m app.rag.embeddings.onnx_embed download

After that everything runs offline. If the files are missing and cannot be
fetched, :func:`app.rag.embeddings.registry.active_embedding` falls back to the
hash embedder with a warning.
"""
from __future__ import annotations

import logging
import os
import ssl
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

# Registry name -> (fastembed model id, dimension)
MODELS: dict[str, tuple[str, int]] = {
    "bge-small-en-v1.5": ("BAAI/bge-small-en-v1.5", 384),
}

_loaded: dict[str, Any] = {}


def available() -> bool:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def os_trust_store():
    """Let model downloads verify TLS against the OS certificate store.

    Hugging Face downloads use a bundled CA list; machines whose antivirus or
    proxy re-signs HTTPS only trust that root in the OS store. Verification
    stays on — against the roots the OS trusts.
    """
    der = ssl.create_default_context().get_ca_certs(binary_form=True)
    if not der:
        # Linux loads OS roots lazily from a directory, so none are listed here;
        # an empty bundle would fail every download. Keep the default CA list.
        yield
        return
    pem = Path(settings.models_dir) / ".os_roots.pem"
    pem.parent.mkdir(parents=True, exist_ok=True)
    pem.write_text("".join(ssl.DER_cert_to_PEM_cert(c) for c in der), encoding="ascii")
    saved = {k: os.environ.get(k) for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")}
    os.environ.update(SSL_CERT_FILE=str(pem), REQUESTS_CA_BUNDLE=str(pem))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load(name: str, *, allow_download: bool | None = None):
    """The fastembed model for *name*, loading (and if allowed downloading) it once."""
    if name in _loaded:
        return _loaded[name]
    from fastembed import TextEmbedding

    model_id, _ = MODELS[name]
    try:
        model = TextEmbedding(model_id, cache_dir=settings.models_dir, local_files_only=True)
    except Exception:
        if not (settings.allow_model_download if allow_download is None else allow_download):
            raise
        log.info("downloading embedding model %s to %s", model_id, settings.models_dir)
        with os_trust_store():
            model = TextEmbedding(model_id, cache_dir=settings.models_dir)
    _loaded[name] = model
    return model


class OnnxEmbedder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dimension = MODELS[name][1]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Passages (chunks) — no instruction prefix."""
        if not texts:
            return []
        return [v.tolist() for v in load(self.name).embed(texts, batch_size=32)]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Search queries — with the model's query instruction."""
        if not texts:
            return []
        return [v.tolist() for v in load(self.name).query_embed(texts)]


if __name__ == "__main__" and sys.argv[1:] == ["download"]:
    logging.basicConfig(level=logging.INFO)
    for registry_name in MODELS:
        load(registry_name, allow_download=True)
        print(f"ready: {registry_name} in {settings.models_dir}")
