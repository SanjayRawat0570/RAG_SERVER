"""Knowledge Graph API (F21).

Endpoints
---------
POST /graph/ingest              Extract entities + relations from text
POST /graph/extract             Extract but do NOT persist (dry-run)
GET  /graph/entities            List / search entities
GET  /graph/entities/{id}       Entity detail + neighbours
GET  /graph/relations           List relations (filter by entity / predicate)
POST /graph/search              Graph search — find entities matching query
POST /graph/path                Shortest path between two entities
GET  /graph/stats               Entity count, relation count, top connected
DELETE /graph/reset             Clear the tenant's graph

Every endpoint works on one tenant's graph (``?tenant=``, default ``default``),
checked with the same RBAC as the rest of the API: reads need ``query``,
writes ``ingest``, reset ``manage``.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, require_tenant
from app.rag.graph import (
    Entity, Relation, ENTITY_TYPES,
    add_entity, add_relation, extract_entities_from_text, extract_relations_from_text,
    find_entities, get_entity, get_relations,
    graph_search, ingest_document, neighbours, reset_graph, shortest_path, stats,
)

router = APIRouter(prefix="/graph", tags=["graph"])

Tenant = Query("default", description="Tenant whose graph to use")


# ── Request models ─────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text:   str = Field(..., min_length=1)
    doc_id: str = ""


class ExtractRequest(BaseModel):
    text:   str = Field(..., min_length=1)
    doc_id: str = ""


class GraphSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)


class PathRequest(BaseModel):
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    max_depth: int = Field(4, ge=1, le=6)


class AddEntityRequest(BaseModel):
    name:     str = Field(..., min_length=1)
    type:     str = "OTHER"
    aliases:  list[str] = Field(default_factory=list)
    doc_ids:  list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AddRelationRequest(BaseModel):
    subject_id:  str = Field(..., min_length=1)
    predicate:   str = Field(..., min_length=1)
    object_id:   str = Field(..., min_length=1)
    confidence:  float = Field(1.0, ge=0.0, le=1.0)
    doc_ids:     list[str] = Field(default_factory=list)
    metadata:    dict[str, Any] = Field(default_factory=dict)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _entity_with_neighbours(entity: Entity, tenant: str) -> dict[str, Any]:
    near = neighbours(entity.id, tenant=tenant)
    return {**entity.model_dump(), "neighbours": near, "degree": len(near)}


# ── Ingestion endpoints ─────────────────────────────────────────────────────────

@router.post("/ingest", status_code=201)
async def ingest_text(user: CurrentUser, req: IngestRequest,
                     tenant: str = Tenant) -> dict[str, Any]:
    """Extract entities and relations from text and add them to the graph."""
    require_tenant(user, tenant, "ingest")
    result = ingest_document(req.text, req.doc_id, tenant=tenant)
    s = stats(tenant=tenant)
    return {
        "extracted":       result,
        "graph_totals":    {"entities": s.entity_count, "relations": s.relation_count},
        "status": "ok",
    }


@router.post("/extract")
async def extract_only(_: CurrentUser, req: ExtractRequest) -> dict[str, Any]:
    """Extract entities and relations without persisting (dry run)."""
    entities  = extract_entities_from_text(req.text, req.doc_id)
    relations = extract_relations_from_text(req.text, entities, req.doc_id)
    return {
        "entities":        [e.model_dump() for e in entities],
        "relations":       [r.model_dump() for r in relations],
        "entity_count":    len(entities),
        "relation_count":  len(relations),
    }


# ── Manual graph mutations ──────────────────────────────────────────────────────

@router.post("/entities", status_code=201)
async def create_entity(user: CurrentUser, req: AddEntityRequest,
                       tenant: str = Tenant) -> dict[str, Any]:
    """Manually add an entity to the graph."""
    require_tenant(user, tenant, "ingest")
    if req.type.upper() not in ENTITY_TYPES:
        raise HTTPException(422, f"Unknown entity type '{req.type}'. "
                                  f"Valid: {sorted(ENTITY_TYPES)}")
    import re, unicodedata
    slug = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", req.name).lower()).strip("_")
    eid  = f"{req.type.upper()}:{slug}"
    entity = Entity(id=eid, name=req.name, type=req.type.upper(),
                    aliases=req.aliases, doc_ids=req.doc_ids, metadata=req.metadata)
    added = add_entity(entity, tenant=tenant)
    return added.model_dump()


@router.post("/relations", status_code=201)
async def create_relation(user: CurrentUser, req: AddRelationRequest,
                         tenant: str = Tenant) -> dict[str, Any]:
    """Manually add a relation to the graph."""
    require_tenant(user, tenant, "ingest")
    rid      = f"{req.subject_id}::{req.predicate}::{req.object_id}"
    relation = Relation(id=rid, subject_id=req.subject_id, predicate=req.predicate,
                        object_id=req.object_id, confidence=req.confidence,
                        doc_ids=req.doc_ids, metadata=req.metadata)
    added = add_relation(relation, tenant=tenant)
    return added.model_dump()


# ── Query endpoints ─────────────────────────────────────────────────────────────

@router.get("/entities")
async def list_entities(
    user:  CurrentUser,
    tenant: str = Tenant,
    name:  str | None = Query(None, description="Substring match on entity name"),
    type:  str | None = Query(None, description="Filter by entity type (PERSON, ORGANIZATION, …)"),
    doc_id: str | None = Query(None, description="Filter entities from a specific document"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Search entities with optional name/type/document filters."""
    require_tenant(user, tenant, "query")
    entities = find_entities(name=name, entity_type=type, doc_id=doc_id, limit=limit, tenant=tenant)
    return {"total": len(entities), "entities": [e.model_dump() for e in entities]}


@router.get("/entities/{entity_id:path}")
async def get_entity_detail(user: CurrentUser, entity_id: str, tenant: str = Tenant) -> dict[str, Any]:
    """Return an entity's full detail including its neighbours."""
    require_tenant(user, tenant, "query")
    entity = get_entity(entity_id, tenant=tenant)
    if entity is None:
        raise HTTPException(404, f"Entity '{entity_id}' not found")
    return _entity_with_neighbours(entity, tenant)


@router.get("/relations")
async def list_relations(
    user:       CurrentUser,
    tenant: str = Tenant,
    entity_id:  str | None = Query(None),
    predicate:  str | None = Query(None),
    direction:  str        = Query("out", pattern="^(in|out|both)$"),
    limit:      int        = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """List relations, optionally filtered by entity / predicate / direction."""
    require_tenant(user, tenant, "query")
    rels = get_relations(entity_id=entity_id, predicate=predicate,
                         direction=direction, limit=limit, tenant=tenant)
    return {"total": len(rels), "relations": [r.model_dump() for r in rels]}


@router.post("/search")
async def search_graph(user: CurrentUser, req: GraphSearchRequest,
                      tenant: str = Tenant) -> dict[str, Any]:
    """Graph-aware search — extract query entities and find connected subgraphs."""
    require_tenant(user, tenant, "query")
    results = graph_search(req.query, top_k=req.top_k, tenant=tenant)
    return {
        "query":   req.query,
        "total":   len(results),
        "results": results,
    }


@router.post("/path")
async def find_path(user: CurrentUser, req: PathRequest, tenant: str = Tenant) -> dict[str, Any]:
    """Find the shortest path between two entities in the graph."""
    require_tenant(user, tenant, "query")
    if get_entity(req.source_id, tenant=tenant) is None:
        raise HTTPException(404, f"Source entity '{req.source_id}' not found")
    if get_entity(req.target_id, tenant=tenant) is None:
        raise HTTPException(404, f"Target entity '{req.target_id}' not found")

    path = shortest_path(req.source_id, req.target_id, max_depth=req.max_depth, tenant=tenant)
    if path is None:
        return {
            "source_id": req.source_id,
            "target_id": req.target_id,
            "path":      None,
            "hops":      None,
            "connected": False,
        }
    return {
        "source_id": req.source_id,
        "target_id": req.target_id,
        "path":      path,
        "hops":      len(path),
        "connected": True,
    }


@router.get("/stats")
async def get_stats(user: CurrentUser, tenant: str = Tenant) -> dict[str, Any]:
    """Return graph statistics — entity/relation counts, type breakdown, top nodes."""
    require_tenant(user, tenant, "query")
    return stats(tenant=tenant).model_dump()


@router.delete("/reset")
async def clear_graph(user: CurrentUser, tenant: str = Tenant) -> dict[str, Any]:
    """Clear this tenant's knowledge graph."""
    require_tenant(user, tenant, "manage")
    reset_graph(tenant=tenant)
    return {"status": "ok", "message": "Graph cleared"}
