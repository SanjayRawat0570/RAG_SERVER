"""Knowledge graph module (F21)."""
from app.rag.graph.models    import Entity, GraphStats, Relation, ENTITY_TYPES
from app.rag.graph.extractor import extract_entities_from_text, extract_relations_from_text
from app.rag.graph.store     import (
    add_entity, add_relation, find_entities, get_entity, get_relations,
    graph_search, ingest_document, neighbours, reset_graph, shortest_path, stats,
)

__all__ = [
    "Entity", "Relation", "GraphStats", "ENTITY_TYPES",
    "extract_entities_from_text", "extract_relations_from_text",
    "add_entity", "add_relation", "get_entity", "find_entities", "get_relations",
    "neighbours", "shortest_path", "graph_search", "ingest_document",
    "stats", "reset_graph",
]
