import os

# Tests run offline and never touch real services, even when backend/.env holds
# live keys. Real env vars take precedence over the .env file in pydantic-settings,
# so set them before any app module (and therefore Settings) is imported.
os.environ.update({
    "APP_ENV": "test",
    "LLM_PROVIDER": "stub",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "COHERE_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "SUPABASE_URL": "",
    "SUPABASE_KEY": "",
    "OTEL_EXPORTER_ENDPOINT": "",
})

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_llm_breakers():
    """LLM gateway breakers are process-wide; isolate them per test."""
    from app.engine.circuit_breaker import registry
    for provider in ("stub", "gemini", "openai", "claude"):
        registry.reset(f"llm:{provider}")
    yield


@pytest.fixture(autouse=True, scope="session")
def _init_tracing():
    """Ensure OTel tracing (and the in-memory span store) is initialised once
    for the entire test session — mirrors what app.main does at import time."""
    from app.observability.tracing import init_tracing
    init_tracing()
