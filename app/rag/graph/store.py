"""In-memory knowledge graph store (F21).

The graph is a simple adjacency structure:
  _entities:  entity_id → Entity
  _relations: relation_id → Relation
  _adj_out:   entity_id → list[Relation]  (outgoing)
  _adj_in:    entity_id → list[Relation]  (incoming)

Graph traversal supports:
  - lookup by entity name/type
  - one-hop neighbour queries
  - multi-hop path finding (BFS, max depth)
  - relation-type filtering
"""
from __future__ import annotations

from collections import deque
from typing import Any

from app.rag.graph.models import Entity, GraphStats, Relation

# ── Global state ────────────────────────────────────────────────────────────────

_entities:  dict[str, Entity]         = {}
_relations: dict[str, Relation]       = {}
_adj_out:   dict[str, list[Relation]] = {}   # entity_id → outgoing relations
_adj_in:    dict[str, list[Relation]] = {}   # entity_id → incoming relations


# ── Write ───────────────────────────────────────────────────────────────────────

def add_entity(entity: Entity) -> Entity:
    existing = _entities.get(entity.id)
    if existing:
        # Merge doc_ids and aliases.
        for d in entity.doc_ids:
            if d not in existing.doc_ids:
                existing.doc_ids.append(d)
        for a in entity.aliases:
            if a not in existing.aliases:
                existing.aliases.append(a)
        return existing
    _entities[entity.id] = entity
    _adj_out.setdefault(entity.id, [])
    _adj_in.setdefault(entity.id, [])
    return entity


def add_relation(relation: Relation) -> Relation:
    # Ensure both endpoints exist (as stubs if needed).
    for eid in (relation.subject_id, relation.object_id):
        if eid not in _entities:
            etype, name = eid.split(":", 1) if ":" in eid else ("OTHER", eid)
            _entities[eid] = Entity(id=eid, name=name.replace("_", " "), type=etype.upper())
            _adj_out.setdefault(eid, [])
            _adj_in.setdefault(eid, [])

    existing = _relations.get(relation.id)
    if existing:
        for d in relation.doc_ids:
            if d not in existing.doc_ids:
                existing.doc_ids.append(d)
        return existing

    _relations[relation.id] = relation
    _adj_out[relation.subject_id].append(relation)
    _adj_in[relation.object_id].append(relation)
    return relation


def ingest_document(text: str, doc_id: str = "") -> dict[str, int]:
    """Extract entities + relations from *text* and add them to the graph."""
    from app.rag.graph.extractor import (
        extract_entities_from_text, extract_relations_from_text,
    )
    entities  = extract_entities_from_text(text, doc_id)
    relations = extract_relations_from_text(text, entities, doc_id)
    for e in entities:
        add_entity(e)
    for r in relations:
        add_relation(r)
    return {"entities": len(entities), "relations": len(relations)}


# ── Read ────────────────────────────────────────────────────────────────────────

def get_entity(entity_id: str) -> Entity | None:
    return _entities.get(entity_id)


def find_entities(
    name: str | None = None,
    entity_type: str | None = None,
    doc_id: str | None = None,
    limit: int = 50,
) -> list[Entity]:
    results = list(_entities.values())
    if name:
        nl = name.lower()
        results = [e for e in results
                   if nl in e.name.lower() or any(nl in a.lower() for a in e.aliases)]
    if entity_type:
        results = [e for e in results if e.type == entity_type.upper()]
    if doc_id:
        results = [e for e in results if doc_id in e.doc_ids]
    return results[:limit]


def get_relations(
    entity_id: str | None = None,
    predicate: str | None = None,
    direction: str = "out",    # "out" | "in" | "both"
    limit: int = 50,
) -> list[Relation]:
    if entity_id:
        if direction == "out":
            rels = list(_adj_out.get(entity_id, []))
        elif direction == "in":
            rels = list(_adj_in.get(entity_id, []))
        else:
            seen: set[str] = set()
            rels = []
            for r in _adj_out.get(entity_id, []) + _adj_in.get(entity_id, []):
                if r.id not in seen:
                    seen.add(r.id)
                    rels.append(r)
    else:
        rels = list(_relations.values())

    if predicate:
        rels = [r for r in rels if r.predicate == predicate]

    return rels[:limit]


def neighbours(entity_id: str, predicate: str | None = None) -> list[dict[str, Any]]:
    """Return all entities directly connected to *entity_id* with relation info."""
    result = []
    for rel in _adj_out.get(entity_id, []):
        if predicate and rel.predicate != predicate:
            continue
        obj = _entities.get(rel.object_id)
        if obj:
            result.append({
                "entity":    obj.model_dump(),
                "relation":  rel.predicate,
                "direction": "out",
                "confidence": rel.confidence,
            })
    for rel in _adj_in.get(entity_id, []):
        if predicate and rel.predicate != predicate:
            continue
        subj = _entities.get(rel.subject_id)
        if subj:
            result.append({
                "entity":    subj.model_dump(),
                "relation":  rel.predicate,
                "direction": "in",
                "confidence": rel.confidence,
            })
    return result


def shortest_path(
    source_id: str,
    target_id: str,
    max_depth: int = 4,
) -> list[dict[str, Any]] | None:
    """BFS shortest path from source to target.  Returns edge list or None."""
    if source_id not in _entities or target_id not in _entities:
        return None
    if source_id == target_id:
        return []

    queue: deque[tuple[str, list[dict]]] = deque([(source_id, [])])
    visited: set[str] = {source_id}

    while queue:
        current_id, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for rel in _adj_out.get(current_id, []):
            nid = rel.object_id
            step = {
                "from":      current_id,
                "relation":  rel.predicate,
                "to":        nid,
            }
            new_path = path + [step]
            if nid == target_id:
                return new_path
            if nid not in visited:
                visited.add(nid)
                queue.append((nid, new_path))
    return None


def graph_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search the graph by extracting entities from *query* and finding matches."""
    from app.rag.graph.extractor import extract_entities_from_text

    query_entities = extract_entities_from_text(query)
    if not query_entities:
        # Fall back to name substring search.
        ql = query.lower()
        matches = [e for e in _entities.values() if ql in e.name.lower()]
        return [{"entity": e.model_dump(), "neighbours": neighbours(e.id)}
                for e in matches[:top_k]]

    results = []
    seen_ids: set[str] = set()
    for qe in query_entities:
        for eid, entity in _entities.items():
            if eid in seen_ids:
                continue
            if qe.name.lower() in entity.name.lower() or entity.name.lower() in qe.name.lower():
                seen_ids.add(eid)
                results.append({
                    "entity":     entity.model_dump(),
                    "neighbours": neighbours(eid),
                    "match_term": qe.name,
                })
    return results[:top_k]


def stats() -> GraphStats:
    type_counts: dict[str, int] = {}
    for e in _entities.values():
        type_counts[e.type] = type_counts.get(e.type, 0) + 1

    # Degree = outgoing + incoming relations.
    degree = {
        eid: len(_adj_out.get(eid, [])) + len(_adj_in.get(eid, []))
        for eid in _entities
    }
    top = sorted(degree, key=lambda x: degree[x], reverse=True)[:5]
    top_entities = [
        {**_entities[eid].model_dump(), "degree": degree[eid]}
        for eid in top
    ]
    return GraphStats(
        entity_count=len(_entities),
        relation_count=len(_relations),
        entity_types=type_counts,
        top_entities=top_entities,
    )


def reset_graph() -> None:
    _entities.clear()
    _relations.clear()
    _adj_out.clear()
    _adj_in.clear()
