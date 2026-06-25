"""Rule-based entity and relation extractor (F21).

No ML/NLP library required — uses regex patterns and a curated verb lexicon.
Works offline, deterministic, and fast.

Entity detection
----------------
PERSON       — "Firstname Lastname" (two+ capitalised words, not a known org suffix)
ORGANIZATION — Proper noun ending in Corp, Inc, Ltd, LLC, Company, Group, etc.
PLACE        — Proper noun preceded by "in", "at", "from", "based in"
TECHNOLOGY   — Known tech keywords or CamelCase identifiers
DATE         — ISO dates, year ranges, named months
NUMBER       — Monetary amounts, percentages, plain numbers
CONCEPT      — Everything else that is capitalised or quoted

Relation detection
------------------
The extractor scans each sentence for (subject, predicate_verb, object) triples
using a small predicate lexicon mapped to canonical relation names.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.rag.graph.models import Entity, Relation

# ── Predicate lexicon: surface verb phrase → canonical predicate ───────────────

_PREDICATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bis(?:\s+the)?\s+ceo\s+of\b", re.I),    "is_ceo_of"),
    (re.compile(r"\bis(?:\s+the)?\s+cto\s+of\b", re.I),    "is_cto_of"),
    (re.compile(r"\bis(?:\s+the)?\s+cfo\s+of\b", re.I),    "is_cfo_of"),
    (re.compile(r"\bis(?:\s+the)?\s+founder\s+of\b", re.I), "founded"),
    (re.compile(r"\bfounded\b", re.I),                       "founded"),
    (re.compile(r"\bworks?\s+(?:at|for)\b", re.I),          "works_at"),
    (re.compile(r"\bemployed\s+(?:at|by)\b", re.I),         "works_at"),
    (re.compile(r"\bmanages?\b", re.I),                      "manages"),
    (re.compile(r"\bleads?\b", re.I),                        "leads"),
    (re.compile(r"\bheads?\b", re.I),                        "leads"),
    (re.compile(r"\bowned\s+by\b", re.I),                    "owned_by"),
    (re.compile(r"\bacquired\b", re.I),                      "acquired"),
    (re.compile(r"\bmerged\s+with\b", re.I),                 "merged_with"),
    (re.compile(r"\bpartnered\s+with\b", re.I),              "partners_with"),
    (re.compile(r"\bis\s+located\s+in\b", re.I),             "located_in"),
    (re.compile(r"\bheadquartered\s+in\b", re.I),            "located_in"),
    (re.compile(r"\bbased\s+in\b", re.I),                    "located_in"),
    (re.compile(r"\boperates?\s+in\b", re.I),                "operates_in"),
    (re.compile(r"\bannounced\b", re.I),                     "announced"),
    (re.compile(r"\blaunched\b", re.I),                      "launched"),
    (re.compile(r"\bdeveloped?\b", re.I),                    "developed"),
    (re.compile(r"\buses?\b", re.I),                         "uses"),
    (re.compile(r"\bintegrates?\s+with\b", re.I),            "integrates_with"),
    (re.compile(r"\bpowers?\b", re.I),                       "powers"),
    (re.compile(r"\bspecialises?\s+in\b", re.I),             "specialises_in"),
    (re.compile(r"\bspecializes?\s+in\b", re.I),             "specialises_in"),
    (re.compile(r"\binvested\s+in\b", re.I),                 "invested_in"),
    (re.compile(r"\bfunded\s+by\b", re.I),                   "funded_by"),
    (re.compile(r"\breported\b", re.I),                      "reported"),
    (re.compile(r"\bcontributed\s+to\b", re.I),              "contributed_to"),
]

# ── Organisation suffixes ──────────────────────────────────────────────────────

_ORG_SUFFIX = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-]+(?:\s+[A-Z][A-Za-z0-9&\-]+)*)\s+"
    r"(?:Corp|Corporation|Inc|Incorporated|Ltd|Limited|LLC|LLP|"
    r"Company|Group|Holdings|Partners|Associates|Technologies|"
    r"Solutions|Services|Systems|Labs|Institute|Foundation|Agency)\b"
)

_PERSON = re.compile(
    r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b"
)

_PLACE_PREP = re.compile(
    r"(?:in|at|from|near|based\s+in)\s+([A-Z][a-zA-Z\s,]{2,40}?)(?:[,.]|$)",
    re.I,
)

_TECH_KEYWORDS = re.compile(
    r"\b(Python|JavaScript|TypeScript|Java|C\+\+|Rust|Go|Kotlin|Swift|"
    r"React|Angular|Vue|Node\.js|FastAPI|Django|Flask|"
    r"PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|"
    r"AWS|Azure|GCP|Docker|Kubernetes|Terraform|"
    r"GPT|ChatGPT|Claude|Gemini|LLM|AI|ML|NLP|"
    r"API|SDK|REST|GraphQL|gRPC|OAuth|JWT)\b"
)

_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{4}s?|(?:January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+\d{4})\b",
    re.I,
)

_MONEY_PCT = re.compile(r"\$[\d.,]+[MBKmb]?|\b\d+(?:\.\d+)?\s*%")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _make_entity_id(etype: str, name: str) -> str:
    return f"{etype}:{_slug(name)}"


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_entities_from_text(
    text: str,
    doc_id: str = "",
) -> list[Entity]:
    """Extract named entities from *text* and return a deduplicated list."""
    seen: dict[str, Entity] = {}

    def _add(name: str, etype: str) -> None:
        eid = _make_entity_id(etype, name)
        if eid not in seen:
            seen[eid] = Entity(id=eid, name=name, type=etype,
                               doc_ids=[doc_id] if doc_id else [])
        elif doc_id and doc_id not in seen[eid].doc_ids:
            seen[eid].doc_ids.append(doc_id)

    # Organizations (before persons to avoid partial overlap)
    for m in _ORG_SUFFIX.finditer(text):
        _add(m.group(0).strip(), "ORGANIZATION")

    # Technologies
    for m in _TECH_KEYWORDS.finditer(text):
        _add(m.group(1), "TECHNOLOGY")

    # Dates
    for m in _DATE.finditer(text):
        _add(m.group(1), "DATE")

    # Money / percentages
    for m in _MONEY_PCT.finditer(text):
        _add(m.group(0), "NUMBER")

    # Places (prep-based)
    for m in _PLACE_PREP.finditer(text):
        place = m.group(1).strip().rstrip(",.")
        if 2 < len(place) < 50:
            _add(place, "PLACE")

    # Persons (two+ capitalised words, not already classified as org/tech)
    org_texts = {e.name for e in seen.values() if e.type == "ORGANIZATION"}
    tech_texts = {e.name for e in seen.values() if e.type == "TECHNOLOGY"}
    skip = org_texts | tech_texts
    for m in _PERSON.finditer(text):
        name = m.group(1)
        if name not in skip and len(name.split()) >= 2:
            _add(name, "PERSON")

    return list(seen.values())


def extract_relations_from_text(
    text: str,
    entities: list[Entity],
    doc_id: str = "",
) -> list[Relation]:
    """Extract subject–predicate–object triples from *text*.

    Uses sentence-level scanning: for each sentence, checks every
    predicate pattern. If a match is found, it identifies the nearest
    entity before the match (subject) and after it (object).
    """
    if not entities:
        return []

    # Build a name → entity_id lookup (longest names first to avoid prefix collision).
    name_map: dict[str, str] = {}
    for e in sorted(entities, key=lambda x: -len(x.name)):
        name_map[e.name.lower()] = e.id

    def _nearest_entity_before(sentence: str, pos: int) -> str | None:
        best: tuple[int, str] | None = None
        for name, eid in name_map.items():
            idx = sentence.lower().rfind(name, 0, pos)
            if idx >= 0 and (best is None or idx > best[0]):
                best = (idx, eid)
        return best[1] if best else None

    def _nearest_entity_after(sentence: str, pos: int) -> str | None:
        best: tuple[int, str] | None = None
        for name, eid in name_map.items():
            idx = sentence.lower().find(name, pos)
            if idx >= 0 and (best is None or idx < best[0]):
                best = (idx, eid)
        return best[1] if best else None

    relations: dict[str, Relation] = {}
    sentences = _SENTENCE_SPLIT.split(text)

    for sentence in sentences:
        for pattern, predicate in _PREDICATES:
            m = pattern.search(sentence)
            if not m:
                continue
            subj_id = _nearest_entity_before(sentence, m.start())
            obj_id  = _nearest_entity_after(sentence, m.end())
            if not subj_id or not obj_id or subj_id == obj_id:
                continue
            rid = f"{subj_id}::{predicate}::{obj_id}"
            if rid not in relations:
                relations[rid] = Relation(
                    id=rid, subject_id=subj_id, predicate=predicate,
                    object_id=obj_id, doc_ids=[doc_id] if doc_id else [],
                    confidence=0.85,
                )
            elif doc_id and doc_id not in relations[rid].doc_ids:
                relations[rid].doc_ids.append(doc_id)

    return list(relations.values())
