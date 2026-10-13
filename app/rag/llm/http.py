"""HTTP settings shared by the hosted LLM providers.

TLS certificates are verified against the operating system's trust store by
default (``ssl.create_default_context()``) rather than httpx's bundled certifi
list. Machines whose antivirus or corporate proxy re-signs HTTPS traffic (e.g.
Avast Web Shield) install their root certificate in the OS store only; with
certifi every call fails with ``CERTIFICATE_VERIFY_FAILED``. Verification stays
on either way. Set ``TLS_TRUST=certifi`` to use the certifi bundle instead.
"""
from __future__ import annotations

import ssl
from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _system_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def tls_verify() -> ssl.SSLContext | bool:
    """Value for httpx's ``verify=`` argument."""
    return True if settings.tls_trust == "certifi" else _system_context()
