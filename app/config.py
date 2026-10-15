"""Application configuration."""
from __future__ import annotations

import logging
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict
from pythonjsonlogger import jsonlogger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    # CORS origins allowed to call the API from a browser.
    cors_origins: list[str] = ["*"]
    # Safety limits for the engine (F1/F3).
    max_nodes: int = 1000
    max_loop_iterations: int = 100
    default_retries: int = 0

    # Supabase (user database). Leave empty to use the in-memory fallback.
    supabase_url: str = ""
    supabase_key: str = ""

    # JWT secret for auth tokens.
    jwt_secret: str = "rag-dev-secret-change-in-production"
    # Demo mode: callers without credentials may *query* (never write) workspaces
    # that are not registered organisations. Off: no credentials -> 401.
    demo_mode: bool = False

    # LLM generation (F16). Default is the offline deterministic stub; the
    # Gemini adapter is used only when an API key is provided.
    llm_provider: str = "stub"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    # Small auxiliary calls (follow-up rewriting, query planning): a fast model
    # without hidden "thinking" tokens and with a larger free-tier quota.
    gemini_fast_model: str = "gemini-flash-lite-latest"
    llm_timeout: int = 30
    # TLS trust for provider calls: "system" (OS store) or "certifi".
    tls_trust: str = "system"
    # LLM gateway guards: per-provider circuit breaker and the default per-tenant
    # spend limit applied when a tenant has no explicit budget (F7/F24).
    llm_breaker_threshold: int = 5
    llm_breaker_recovery: float = 30.0
    default_tenant_budget_usd: float = 10.0

    # Agentic retrieval hard limits (per request).
    agent_max_rounds: int = 3
    agent_timeout_s: float = 45.0
    agent_token_budget: int = 8000

    # Embeddings for the knowledge base. "local-hash" is lexical and needs nothing;
    # "bge-small-en-v1.5" is a local semantic model (fastembed/ONNX, ~64 MB, offline
    # once downloaded). An unavailable model falls back to local-hash with a warning.
    embedding_model: str = "bge-small-en-v1.5"
    models_dir: str = "models"
    allow_model_download: bool = True

    # Multimodal ingestion (per-tenant opt-in: tenant settings ``vision_enabled``).
    vision_provider: str = "gemini"
    vision_model: str = ""              # empty → gemini_model
    vision_max_images: int = 20         # per document — bounds cost and free-tier quota
    vision_min_image_px: int = 150      # skip icons, logos and rules smaller than this

    # Directory for state shared across processes (API keys, tenant AI settings).
    data_dir: str = "data"

    # Embedding API keys (F11).  Leave empty to use free local models only.
    openai_api_key: str = ""
    cohere_api_key: str = ""

    # LLM API keys (F16). Leave empty to use free/stub providers only.
    anthropic_api_key: str = ""

    # Observability (F8).
    service_name: str = "rag-orchestrator"
    # OTLP gRPC endpoint for traces, e.g. http://jaeger:4317. Empty disables
    # the exporter (spans are still created, just not shipped) so the app runs
    # fine locally and under test without a collector.
    otel_exporter_endpoint: str = ""
    metrics_enabled: bool = True


settings = Settings()


class _TraceContextFilter(logging.Filter):
    """Inject the active trace id into every log record for trace<->log
    correlation (F8). Imported lazily to avoid a circular import."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from app.observability.tracing import current_trace_id

            record.trace_id = current_trace_id()
        except Exception:  # observability must never break logging
            record.trace_id = None
        return True


def configure_logging() -> None:
    """Structured JSON logging with trace correlation (F8)."""
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    handler.addFilter(_TraceContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
