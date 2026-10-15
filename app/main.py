"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app import __version__
from app.api.auth import router as auth_router
from app.api.chains import router as chains_router
from app.api.chunks import router as chunks_router
from app.api.embeddings import router as embeddings_router
from app.api.cache import router as cache_router
from app.api.graph import router as graph_router
from app.api.cost     import router as cost_router
from app.api.security import router as security_router
from app.api.feedback import router as feedback_router
from app.api.personalization import router as personalization_router
from app.api.indexing import router as indexing_router
from app.api.tenants import router as tenants_router
from app.api.tenant_settings import router as tenant_settings_router
from app.api.context import router as context_router
from app.api.conversations import router as conversations_router
from app.api.rag import router as rag_router
from app.api.rerank import router as rerank_router
from app.api.search import router as search_router
from app.api.vectors import router as vectors_router
from app.api.decisions import router as decisions_router
from app.api.errors import router as errors_router
from app.api.ingest import router as ingest_router
from app.api.merge import router as merge_router
from app.api.monitoring import router as monitoring_router
from app.api.routes import router
from app.api.stream import router as stream_router
from app.api.upload import router as upload_router
from app.config import configure_logging, settings
from app.observability.tracing import init_tracing
from app.rag.security.rbac import AccessDenied

configure_logging()
init_tracing()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Reopen the journalled vector stores so indexed documents survive restarts.
    from app.rag.vectorstore import load_persisted_stores

    load_persisted_stores()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Enterprise RAG — Orchestration Engine",
    description="Workflow & orchestration engine (F1-F8): DAG execution, "
    "branching, chaining, merging, decision trees, streaming, resilience, "
    "and full observability (tracing + metrics).",
    version=__version__,
)

# Allow the frontend (and any deploy origin) to call the API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AccessDenied)
async def access_denied(_: Request, exc: AccessDenied) -> JSONResponse:
    """A tenant check deep in the data layer (vector stores, graph) refused the caller."""
    return JSONResponse(status_code=403, content={"detail": str(exc)})


# Auto-instrument HTTP requests so each call produces a trace spanning the
# whole pipeline (F8).
FastAPIInstrumentor.instrument_app(app)

app.include_router(auth_router,      prefix="/api/v1")
app.include_router(router,           prefix="/api/v1")
app.include_router(upload_router,    prefix="/api/v1")
app.include_router(chains_router,    prefix="/api/v1")
app.include_router(merge_router,     prefix="/api/v1")
app.include_router(decisions_router, prefix="/api/v1")
app.include_router(stream_router,    prefix="/api/v1")
app.include_router(errors_router,      prefix="/api/v1")
app.include_router(monitoring_router,  prefix="/api/v1")
app.include_router(ingest_router,      prefix="/api/v1")
app.include_router(chunks_router,      prefix="/api/v1")
app.include_router(embeddings_router,  prefix="/api/v1")
app.include_router(vectors_router,     prefix="/api/v1")
app.include_router(search_router,      prefix="/api/v1")
app.include_router(rerank_router,      prefix="/api/v1")
app.include_router(cache_router,       prefix="/api/v1")
app.include_router(graph_router,          prefix="/api/v1")
app.include_router(cost_router,           prefix="/api/v1")
app.include_router(security_router,       prefix="/api/v1")
app.include_router(feedback_router,        prefix="/api/v1")
app.include_router(personalization_router, prefix="/api/v1")
app.include_router(indexing_router,    prefix="/api/v1")
app.include_router(tenants_router,     prefix="/api/v1")
app.include_router(tenant_settings_router, prefix="/api/v1")
app.include_router(context_router,     prefix="/api/v1")
app.include_router(conversations_router, prefix="/api/v1")
app.include_router(rag_router,         prefix="/api/v1")


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "version": __version__}
