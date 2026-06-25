import pytest


@pytest.fixture(autouse=True)
def _anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def _init_tracing():
    """Ensure OTel tracing (and the in-memory span store) is initialised once
    for the entire test session — mirrors what app.main does at import time."""
    from app.observability.tracing import init_tracing
    init_tracing()
