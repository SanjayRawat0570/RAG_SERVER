"""Tests for F21: Knowledge Graph."""
from __future__ import annotations

import pytest

from app.rag.graph import (
    Entity, Relation, ENTITY_TYPES,
    add_entity, add_relation, extract_entities_from_text, extract_relations_from_text,
    find_entities, get_entity, get_relations,
    graph_search, ingest_document, neighbours, reset_graph, shortest_path, stats,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean():
    reset_graph()
    yield
    reset_graph()


# ── Entity extraction ──────────────────────────────────────────────────────────

def test_f21_extract_person():
    entities = extract_entities_from_text("John Smith works at ABC Corp.")
    types = {e.type for e in entities}
    assert "PERSON" in types
    persons = [e for e in entities if e.type == "PERSON"]
    assert any("John" in p.name for p in persons)


def test_f21_extract_organization():
    entities = extract_entities_from_text("TechCorp Inc announced record revenues.")
    orgs = [e for e in entities if e.type == "ORGANIZATION"]
    assert len(orgs) >= 1
    assert any("TechCorp" in o.name for o in orgs)


def test_f21_extract_technology():
    entities = extract_entities_from_text("We use Python and FastAPI to build the API.")
    techs = [e for e in entities if e.type == "TECHNOLOGY"]
    tech_names = [t.name for t in techs]
    assert "Python" in tech_names or "FastAPI" in tech_names


def test_f21_extract_date():
    entities = extract_entities_from_text("The product launched in January 2024.")
    dates = [e for e in entities if e.type == "DATE"]
    assert len(dates) >= 1


def test_f21_extract_number():
    entities = extract_entities_from_text("Revenue grew by 25% to $10M.")
    nums = [e for e in entities if e.type == "NUMBER"]
    assert len(nums) >= 1


def test_f21_extract_place():
    entities = extract_entities_from_text("The company is based in New York.")
    places = [e for e in entities if e.type == "PLACE"]
    assert len(places) >= 1


def test_f21_extract_assigns_doc_id():
    entities = extract_entities_from_text("Python is used at Google Inc.", doc_id="doc-1")
    assert all("doc-1" in e.doc_ids for e in entities)


def test_f21_extract_entity_id_format():
    entities = extract_entities_from_text("John Smith is a developer.")
    for e in entities:
        assert ":" in e.id
        etype, _ = e.id.split(":", 1)
        assert etype.upper() in ENTITY_TYPES


def test_f21_extract_deduplicates():
    text = "John Smith leads the team. John Smith is the CEO."
    entities = extract_entities_from_text(text)
    johns = [e for e in entities if "John" in e.name and e.type == "PERSON"]
    assert len(johns) == 1   # deduplicated


# ── Relation extraction ────────────────────────────────────────────────────────

def test_f21_extract_works_at_relation():
    text = "Sarah Johnson works at Acme Corp."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    predicates = [r.predicate for r in relations]
    assert "works_at" in predicates


def test_f21_extract_ceo_relation():
    text = "Alice Brown is CEO of DataCorp Ltd."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    predicates = [r.predicate for r in relations]
    assert "is_ceo_of" in predicates


def test_f21_extract_manages_relation():
    text = "Bob Chen manages Project Alpha."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    predicates = [r.predicate for r in relations]
    assert "manages" in predicates


def test_f21_extract_located_in_relation():
    text = "Acme Corp is located in New York."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    predicates = [r.predicate for r in relations]
    assert "located_in" in predicates


def test_f21_relation_has_subject_and_object():
    text = "Sarah Johnson works at TechCorp Inc."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    for r in relations:
        assert r.subject_id
        assert r.object_id
        assert r.subject_id != r.object_id


def test_f21_relation_confidence_between_0_and_1():
    text = "Bob manages the Engineering team at DataSoft Ltd."
    entities = extract_entities_from_text(text)
    relations = extract_relations_from_text(text, entities)
    for r in relations:
        assert 0.0 <= r.confidence <= 1.0


# ── Graph store ────────────────────────────────────────────────────────────────

def test_f21_add_entity():
    e = Entity(id="PERSON:alice", name="Alice", type="PERSON")
    added = add_entity(e)
    assert get_entity("PERSON:alice") is added


def test_f21_add_entity_merges_doc_ids():
    e1 = Entity(id="PERSON:alice", name="Alice", type="PERSON", doc_ids=["doc-1"])
    e2 = Entity(id="PERSON:alice", name="Alice", type="PERSON", doc_ids=["doc-2"])
    add_entity(e1)
    add_entity(e2)
    merged = get_entity("PERSON:alice")
    assert "doc-1" in merged.doc_ids
    assert "doc-2" in merged.doc_ids


def test_f21_add_relation():
    add_entity(Entity(id="PERSON:john", name="John", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    r = Relation(id="PERSON:john::works_at::ORGANIZATION:acme",
                 subject_id="PERSON:john", predicate="works_at",
                 object_id="ORGANIZATION:acme")
    add_relation(r)
    rels = get_relations("PERSON:john", direction="out")
    assert any(r.predicate == "works_at" for r in rels)


def test_f21_add_relation_creates_stub_entities():
    r = Relation(id="X:a::knows::X:b", subject_id="X:a",
                 predicate="knows", object_id="X:b")
    add_relation(r)
    assert get_entity("X:a") is not None
    assert get_entity("X:b") is not None


def test_f21_get_entity_none_when_missing():
    assert get_entity("PERSON:nobody") is None


def test_f21_find_entities_by_name():
    add_entity(Entity(id="PERSON:alice_smith", name="Alice Smith", type="PERSON"))
    add_entity(Entity(id="PERSON:bob", name="Bob", type="PERSON"))
    results = find_entities(name="alice")
    assert len(results) == 1
    assert results[0].name == "Alice Smith"


def test_f21_find_entities_by_type():
    add_entity(Entity(id="PERSON:alice", name="Alice", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    persons = find_entities(entity_type="PERSON")
    assert all(e.type == "PERSON" for e in persons)


def test_f21_neighbours_returns_connected_entities():
    add_entity(Entity(id="PERSON:john", name="John", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    add_relation(Relation(id="PERSON:john::works_at::ORGANIZATION:acme",
                          subject_id="PERSON:john", predicate="works_at",
                          object_id="ORGANIZATION:acme"))
    nbrs = neighbours("PERSON:john")
    assert len(nbrs) >= 1
    assert nbrs[0]["relation"] == "works_at"


def test_f21_neighbours_predicate_filter():
    add_entity(Entity(id="PERSON:john", name="John", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    add_entity(Entity(id="PLACE:ny", name="New York", type="PLACE"))
    add_relation(Relation(id="PERSON:john::works_at::ORGANIZATION:acme",
                          subject_id="PERSON:john", predicate="works_at",
                          object_id="ORGANIZATION:acme"))
    add_relation(Relation(id="PERSON:john::located_in::PLACE:ny",
                          subject_id="PERSON:john", predicate="located_in",
                          object_id="PLACE:ny"))
    works_at_nbrs = neighbours("PERSON:john", predicate="works_at")
    assert len(works_at_nbrs) == 1
    assert works_at_nbrs[0]["relation"] == "works_at"


def test_f21_shortest_path_direct():
    add_entity(Entity(id="PERSON:john", name="John", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    add_relation(Relation(id="PERSON:john::works_at::ORGANIZATION:acme",
                          subject_id="PERSON:john", predicate="works_at",
                          object_id="ORGANIZATION:acme"))
    path = shortest_path("PERSON:john", "ORGANIZATION:acme")
    assert path is not None
    assert len(path) == 1
    assert path[0]["relation"] == "works_at"


def test_f21_shortest_path_two_hops():
    add_entity(Entity(id="PERSON:john", name="John", type="PERSON"))
    add_entity(Entity(id="ORGANIZATION:acme", name="Acme", type="ORGANIZATION"))
    add_entity(Entity(id="PLACE:ny", name="New York", type="PLACE"))
    add_relation(Relation(id="PERSON:john::works_at::ORGANIZATION:acme",
                          subject_id="PERSON:john", predicate="works_at",
                          object_id="ORGANIZATION:acme"))
    add_relation(Relation(id="ORGANIZATION:acme::located_in::PLACE:ny",
                          subject_id="ORGANIZATION:acme", predicate="located_in",
                          object_id="PLACE:ny"))
    path = shortest_path("PERSON:john", "PLACE:ny")
    assert path is not None
    assert len(path) == 2


def test_f21_shortest_path_no_connection():
    add_entity(Entity(id="PERSON:a", name="A", type="PERSON"))
    add_entity(Entity(id="PERSON:b", name="B", type="PERSON"))
    assert shortest_path("PERSON:a", "PERSON:b") is None


def test_f21_shortest_path_unknown_entity():
    assert shortest_path("PERSON:ghost", "PERSON:nobody") is None


def test_f21_ingest_document():
    text = "Sarah Johnson is CEO of TechCorp Inc. TechCorp Inc is located in San Francisco."
    result = ingest_document(text, "doc-1")
    assert result["entities"] >= 1
    s = stats()
    assert s.entity_count >= 1


def test_f21_graph_search_finds_entity():
    ingest_document("Alice Brown works at DataCorp Ltd.", "doc-1")
    results = graph_search("Alice Brown")
    assert len(results) >= 1
    entity_names = [r["entity"]["name"] for r in results]
    assert any("Alice" in n for n in entity_names)


def test_f21_stats_counts_correctly():
    add_entity(Entity(id="PERSON:a", name="A", type="PERSON"))
    add_entity(Entity(id="PERSON:b", name="B", type="PERSON"))
    add_relation(Relation(id="PERSON:a::knows::PERSON:b",
                          subject_id="PERSON:a", predicate="knows",
                          object_id="PERSON:b"))
    s = stats()
    assert s.entity_count == 2
    assert s.relation_count == 1
    assert s.entity_types.get("PERSON", 0) == 2


# ── API tests ──────────────────────────────────────────────────────────────────

AUTH = {"Authorization": "Bearer dev"}
CORP_TEXT = (
    "Sarah Johnson is CEO of TechCorp Inc. "
    "TechCorp Inc is headquartered in San Francisco. "
    "Sarah Johnson manages the AI product development."
)


def _client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def test_f21_api_extract_dry_run():
    with _client() as c:
        resp = c.post("/api/v1/graph/extract",
                      json={"text": CORP_TEXT, "doc_id": "d1"}, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "entities"  in data
    assert "relations" in data
    assert data["entity_count"] >= 1


def test_f21_api_ingest():
    with _client() as c:
        resp = c.post("/api/v1/graph/ingest",
                      json={"text": CORP_TEXT, "doc_id": "d1"}, headers=AUTH)
    assert resp.status_code == 201
    data = resp.json()
    assert data["extracted"]["entities"] >= 1
    assert data["status"] == "ok"


def test_f21_api_list_entities():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.get("/api/v1/graph/entities", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1


def test_f21_api_list_entities_type_filter():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.get("/api/v1/graph/entities?type=ORGANIZATION", headers=AUTH)
    assert resp.status_code == 200
    for e in resp.json()["entities"]:
        assert e["type"] == "ORGANIZATION"


def test_f21_api_get_entity_detail():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        ents = c.get("/api/v1/graph/entities?type=ORGANIZATION", headers=AUTH).json()
        if ents["total"] > 0:
            eid  = ents["entities"][0]["id"]
            resp = c.get(f"/api/v1/graph/entities/{eid}", headers=AUTH)
            assert resp.status_code == 200
            assert resp.json()["id"] == eid


def test_f21_api_get_entity_not_found():
    with _client() as c:
        resp = c.get("/api/v1/graph/entities/PERSON:nobody", headers=AUTH)
    assert resp.status_code == 404


def test_f21_api_list_relations():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.get("/api/v1/graph/relations", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "relations" in data


def test_f21_api_graph_search():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.post("/api/v1/graph/search",
                      json={"query": "Sarah Johnson", "top_k": 5}, headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data


def test_f21_api_create_entity_and_relation():
    with _client() as c:
        e1 = c.post("/api/v1/graph/entities",
                    json={"name": "Alice", "type": "PERSON"}, headers=AUTH)
        e2 = c.post("/api/v1/graph/entities",
                    json={"name": "Widgets Inc", "type": "ORGANIZATION"}, headers=AUTH)
        assert e1.status_code == 201
        assert e2.status_code == 201

        r = c.post("/api/v1/graph/relations", json={
            "subject_id": e1.json()["id"],
            "predicate":  "works_at",
            "object_id":  e2.json()["id"],
        }, headers=AUTH)
        assert r.status_code == 201
        assert r.json()["predicate"] == "works_at"


def test_f21_api_path():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        orgs = c.get("/api/v1/graph/entities?type=ORGANIZATION", headers=AUTH).json()
        persons = c.get("/api/v1/graph/entities?type=PERSON", headers=AUTH).json()
        if orgs["total"] > 0 and persons["total"] > 0:
            src = persons["entities"][0]["id"]
            tgt = orgs["entities"][0]["id"]
            resp = c.post("/api/v1/graph/path",
                          json={"source_id": src, "target_id": tgt}, headers=AUTH)
            assert resp.status_code == 200
            assert "connected" in resp.json()


def test_f21_api_path_unknown_entity():
    with _client() as c:
        resp = c.post("/api/v1/graph/path",
                      json={"source_id": "PERSON:ghost", "target_id": "PERSON:nobody"},
                      headers=AUTH)
    assert resp.status_code == 404


def test_f21_api_stats():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.get("/api/v1/graph/stats", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert "entity_count"   in data
    assert "relation_count" in data
    assert "entity_types"   in data


def test_f21_api_reset():
    with _client() as c:
        c.post("/api/v1/graph/ingest", json={"text": CORP_TEXT}, headers=AUTH)
        resp = c.delete("/api/v1/graph/reset", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_f21_api_no_auth():
    with _client() as c:
        resp = c.get("/api/v1/graph/stats")
    assert resp.status_code == 401
