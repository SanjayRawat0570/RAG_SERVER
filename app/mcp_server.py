"""MCP server — the knowledge base as tools for MCP clients (e.g. Claude Desktop).

Built on the official Python MCP SDK (``FastMCP``) and run over stdio::

    RAG_API_KEY=sk-... python -m app.mcp_server

Tools:

* ``search_knowledge_base(query, top_k)`` — hybrid search, passages only
* ``ask(question, mode?, conversation_id?)`` — a cited answer

The server is a thin client of the REST API: every tool call is an HTTP request
that carries the API key, so authentication, RBAC (``rbac.authorize``) and
tenant isolation are exactly the checks the REST API runs — there is no second
code path to get wrong. The tenant is never a tool argument: it is the tenant
the key is scoped to, learned from ``GET /security/whoami``.

Environment:

* ``RAG_API_KEY`` (required) — a tenant-scoped key (Settings → MCP access, or
  ``POST /api/v1/security/api-keys`` with ``tenant``)
* ``RAG_API_URL`` — API base URL, default ``http://localhost:8000/api/v1``

stdout carries the MCP protocol, so this module only logs to stderr.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

DEFAULT_API_URL = "http://localhost:8000/api/v1"
_PREVIEW = 600

log = logging.getLogger("rag.mcp")


class RAGClient:
    """REST API calls made with one API key; the key decides the tenant."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_API_URL, *,
                 transport: httpx.AsyncBaseTransport | None = None, timeout: float = 120.0):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport,
                                       headers={"X-API-Key": api_key}, timeout=timeout)
        self._tenant: str | None = None

    async def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            resp = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ToolError(f"RAG API unreachable at {self._http.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ToolError(f"RAG API error {resp.status_code}: {detail}")
        return resp.json()

    async def tenant(self) -> str:
        if self._tenant is None:
            me = await self._call("GET", "/security/whoami")
            if not me.get("tenant"):
                raise ToolError("This API key is not scoped to a tenant. Create a key for a "
                                "workspace (Settings → MCP access) and use that one.")
            self._tenant = me["tenant"]
        return self._tenant

    async def search(self, query: str, top_k: int) -> dict[str, Any]:
        return await self._call("POST", "/rag/search",
                                json={"tenant": await self.tenant(), "query": query, "top_k": top_k})

    async def ask(self, question: str, mode: str, conversation_id: str | None) -> dict[str, Any]:
        body: dict[str, Any] = {"tenant": await self.tenant(), "question": question, "mode": mode}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return await self._call("POST", "/rag/ask", json=body)

    async def aclose(self) -> None:
        await self._http.aclose()


def _answer_view(out: dict[str, Any]) -> dict[str, Any]:
    """The parts of a /rag/ask response an MCP client needs: answer + cited sources."""
    markers = set(out.get("cited_markers") or [])
    citations = [
        {k: c.get(k) for k in ("marker", "source", "heading", "page", "id") if c.get(k) is not None}
        for c in out.get("citations") or [] if c.get("marker") in markers
    ]
    view: dict[str, Any] = {
        "answer": out.get("answer", ""),
        "citations": citations,
        "confidence": out.get("confidence"),
        "answerable": out.get("answerable"),
        "conversation_id": out.get("conversation_id"),
    }
    if out.get("standalone_question") and out["standalone_question"] != out.get("original_question"):
        view["standalone_question"] = out["standalone_question"]
    if out.get("steps"):
        view["steps"] = [{k: s.get(k) for k in ("type", "tool", "query", "round") if s.get(k) is not None}
                         for s in out["steps"]]
    return view


def build_server(client: RAGClient) -> FastMCP:
    mcp = FastMCP(
        "enterprise-rag",
        instructions="Search and question the organisation's knowledge base. Answers cite "
                     "their sources as [n] markers; quote the citations when you use an answer.",
    )

    @mcp.tool()
    async def search_knowledge_base(query: str, top_k: int = 5) -> dict[str, Any]:
        """Hybrid search (semantic + keyword, reranked) over the knowledge base.

        Returns the best matching passages with their source file, heading and page,
        without generating an answer. top_k: 1-20.
        """
        res = await client.search(query, max(1, min(int(top_k), 20)))
        return {
            "query": query,
            "results": [{**r, "text": str(r.get("text", ""))[:_PREVIEW]} for r in res.get("results", [])],
        }

    @mcp.tool()
    async def ask(question: str, mode: Literal["fast", "agentic"] = "fast",
                  conversation_id: str | None = None) -> dict[str, Any]:
        """Answer a question from the knowledge base, with citations.

        mode "fast" runs one retrieval pass; "agentic" lets the server-side agent search
        several times (use it for multi-hop questions). Pass the returned
        conversation_id back to ask a follow-up question in the same conversation.
        """
        return _answer_view(await client.ask(question, mode, conversation_id))

    return mcp


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    api_key = os.environ.get("RAG_API_KEY", "").strip()
    if not api_key:
        sys.exit("RAG_API_KEY is not set: create a tenant-scoped API key and pass it in the environment.")
    base_url = os.environ.get("RAG_API_URL", DEFAULT_API_URL)
    log.info("enterprise-rag MCP server → %s", base_url)
    build_server(RAGClient(api_key, base_url)).run("stdio")


if __name__ == "__main__":
    main()
