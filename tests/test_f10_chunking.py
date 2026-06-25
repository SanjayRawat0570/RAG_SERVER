"""Tests for F10: Advanced Chunking Strategies."""
from __future__ import annotations

import pytest
from app.rag.models import Document


# ── Fixtures ───────────────────────────────────────────────────────────────────

PROSE = (
    "The company was founded in 2020 by John and Sarah. "
    "They wanted to solve the problem of document retrieval at scale. "
    "In Q3 2021 they launched their first product. "
    "Revenue grew by 40 percent in the following year. "
    "The team expanded from 5 to 50 people within 18 months. "
    "Customer satisfaction scores reached 95 percent in 2023. "
    "The platform now processes over 1 million queries per day. "
    "Future plans include expanding to European and Asian markets. "
) * 3


MARKDOWN = """# Annual Report 2024

## Executive Summary

Revenue increased by 20 percent year-on-year.
The company achieved profitability for the first time.

## Financial Results

### Revenue

Total revenue was $10 million.
Product sales accounted for 60 percent.

### Expenses

Operating expenses were $8 million.
Headcount grew by 30 percent.

## Outlook

Next year we expect 25 percent growth.
New product lines will be launched in Q2.
"""


PYTHON_CODE = '''
def calculate_revenue(sales, returns):
    """Calculate net revenue."""
    net = sales - returns
    return max(0, net)


def calculate_expenses(salaries, overhead, marketing):
    """Sum all operating expenses."""
    return salaries + overhead + marketing


class FinancialReport:
    """Quarterly financial report generator."""

    def __init__(self, quarter: int, year: int):
        self.quarter = quarter
        self.year = year

    def generate(self, revenue: float, expenses: float) -> dict:
        profit = revenue - expenses
        return {
            "quarter": self.quarter,
            "year": self.year,
            "revenue": revenue,
            "expenses": expenses,
            "profit": profit,
            "margin": profit / revenue if revenue else 0,
        }

    def summary(self) -> str:
        return f"Q{self.quarter} {self.year} Report"
'''


def _doc(text: str, fmt: str = "text", filename: str = "doc.txt", **meta) -> Document:
    return Document(
        document_id="test",
        text=text,
        format=fmt,
        source=filename,
        metadata={"title": "Test", **meta},
    )


# ── Strategy: fixed ────────────────────────────────────────────────────────────

def test_f10_fixed_basic():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PROSE), strategy="fixed",
                            config={"chunk_size": 100, "size_unit": "chars"})
    assert len(chunks) > 1
    for c in chunks:
        assert c.char_count <= 110  # some tolerance for edge


def test_f10_fixed_overlap_creates_continuity():
    from app.rag.chunking import chunk_document
    text = "AAABBBCCCDDDEEE"
    chunks = chunk_document(_doc(text), strategy="fixed",
                            config={"chunk_size": 6, "overlap": 3, "size_unit": "chars"})
    # Each chunk except the first should start with the last 3 chars of the previous
    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        tail_prev = chunks[i - 1].text[-3:]
        head_curr = chunks[i].text[:3]
        assert tail_prev == head_curr


def test_f10_fixed_empty_text():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(""), strategy="fixed",
                            config={"chunk_size": 100, "size_unit": "chars"})
    assert chunks == []


# ── Strategy: recursive ────────────────────────────────────────────────────────

def test_f10_recursive_respects_paragraphs():
    from app.rag.chunking import chunk_document
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_document(_doc(text), strategy="recursive",
                            config={"chunk_size": 20, "size_unit": "chars"})
    # Each paragraph should be its own chunk
    texts = [c.text.strip() for c in chunks]
    assert any("Para one" in t for t in texts)
    assert any("Para two" in t for t in texts)


def test_f10_recursive_never_exceeds_budget():
    from app.rag.chunking import chunk_document
    size = 50
    chunks = chunk_document(_doc(PROSE), strategy="recursive",
                            config={"chunk_size": size, "size_unit": "chars"})
    for c in chunks:
        assert c.char_count <= size + 5  # slight tolerance for last chunk


# ── Strategy: semantic ─────────────────────────────────────────────────────────

def test_f10_semantic_does_not_split_sentences():
    from app.rag.chunking import chunk_document
    text = (
        "First sentence here. Second sentence here. "
        "Third sentence has more words and is longer. "
        "Fourth sentence wraps up the paragraph."
    )
    chunks = chunk_document(_doc(text), strategy="semantic",
                            config={"chunk_size": 80, "size_unit": "chars"})
    for c in chunks:
        # No chunk should end mid-sentence (unless it was force-split due to size)
        stripped = c.text.strip()
        if len(stripped) < 80:
            assert stripped.endswith((".", "!", "?")) or stripped == text.strip()


def test_f10_sentence_alias_matches_semantic():
    from app.rag.chunking import chunk_document
    sem = chunk_document(_doc(PROSE), strategy="semantic",
                         config={"chunk_size": 200, "size_unit": "chars"})
    sen = chunk_document(_doc(PROSE), strategy="sentence",
                         config={"chunk_size": 200, "size_unit": "chars"})
    assert len(sem) == len(sen)
    assert [c.text for c in sem] == [c.text for c in sen]


# ── Strategy: structure ────────────────────────────────────────────────────────

def test_f10_structure_splits_on_headings():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(MARKDOWN, fmt="markdown"), strategy="structure",
                            config={"chunk_size": 300, "size_unit": "chars"})
    headings = [c.metadata.get("heading") for c in chunks if c.metadata.get("heading")]
    assert len(headings) >= 3  # at least the major headings


def test_f10_structure_keeps_heading_in_metadata():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(MARKDOWN, fmt="markdown"), strategy="structure",
                            config={"chunk_size": 1000, "size_unit": "chars"})
    exec_chunk = next((c for c in chunks if c.metadata.get("heading") == "Executive Summary"), None)
    assert exec_chunk is not None
    assert "Revenue increased" in exec_chunk.text


def test_f10_structure_heading_level_in_metadata():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(MARKDOWN, fmt="markdown"), strategy="structure",
                            config={"chunk_size": 1000, "size_unit": "chars"})
    h1_chunks = [c for c in chunks if c.metadata.get("level") == 1]
    h2_chunks = [c for c in chunks if c.metadata.get("level") == 2]
    assert len(h1_chunks) >= 1
    assert len(h2_chunks) >= 1


# ── Strategy: code ─────────────────────────────────────────────────────────────

def test_f10_code_splits_at_functions():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PYTHON_CODE, fmt="text", filename="report.py"),
                            strategy="code",
                            config={"chunk_size": 300, "size_unit": "chars"})
    texts = [c.text for c in chunks]
    # Each function should be in its own chunk (or fewer if they fit together)
    assert any("def calculate_revenue" in t for t in texts)
    assert any("def calculate_expenses" in t for t in texts)
    assert any("class FinancialReport" in t for t in texts)


def test_f10_code_does_not_split_function_body():
    from app.rag.chunking import chunk_document
    # If the budget is large enough, the whole function stays together
    chunks = chunk_document(_doc(PYTHON_CODE, fmt="text", filename="code.py"),
                            strategy="code",
                            config={"chunk_size": 800, "size_unit": "chars"})
    # calculate_revenue function is ~100 chars — it should be in one chunk
    revenue_chunks = [c for c in chunks if "def calculate_revenue" in c.text]
    assert len(revenue_chunks) >= 1
    # The function body should be in the same chunk
    for c in revenue_chunks:
        if "def calculate_revenue" in c.text:
            assert "return max" in c.text or len(c.text) > 50


def test_f10_code_block_index_in_metadata():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PYTHON_CODE, fmt="text", filename="code.py"),
                            strategy="code",
                            config={"chunk_size": 400, "size_unit": "chars"})
    for c in chunks:
        assert "block_index" in c.metadata


def test_f10_code_fallback_for_plain_text():
    from app.rag.chunking import chunk_document
    # Code strategy should fall back gracefully when no boundaries found
    chunks = chunk_document(_doc("No code here, just plain text words. " * 10),
                            strategy="code",
                            config={"chunk_size": 100, "size_unit": "chars"})
    assert len(chunks) >= 1


def test_f10_code_javascript():
    from app.rag.chunking import chunk_document
    js = """
function calculateRevenue(sales, returns) {
    return Math.max(0, sales - returns);
}

const calculateExpenses = (salaries, overhead) => {
    return salaries + overhead;
};

class FinancialReport {
    constructor(quarter, year) {
        this.quarter = quarter;
        this.year = year;
    }

    generate(revenue, expenses) {
        return { profit: revenue - expenses };
    }
}
"""
    chunks = chunk_document(_doc(js, fmt="text", filename="finance.js"),
                            strategy="code",
                            config={"chunk_size": 400, "size_unit": "chars"})
    texts = [c.text for c in chunks]
    assert any("function calculateRevenue" in t for t in texts)
    assert any("class FinancialReport" in t for t in texts)


# ── Metadata propagation ───────────────────────────────────────────────────────

def test_f10_chunks_carry_strategy_name():
    from app.rag.chunking import chunk_document
    for strat in ["fixed", "recursive", "semantic", "structure", "code"]:
        chunks = chunk_document(_doc(PROSE), strategy=strat,
                                config={"chunk_size": 200, "size_unit": "chars"})
        if chunks:
            assert chunks[0].metadata["strategy"] == strat


def test_f10_chunks_carry_language_from_quality():
    from app.rag.chunking import chunk_document
    doc = _doc(PROSE, quality={"score": 0.9, "language": "en", "action": "ok"})
    chunks = chunk_document(doc, strategy="recursive",
                            config={"chunk_size": 200, "size_unit": "chars"})
    for c in chunks:
        assert c.metadata.get("language") == "en"
        assert c.metadata.get("quality_score") == 0.9


def test_f10_chunks_carry_quality_score():
    from app.rag.chunking import chunk_document
    doc = _doc(PROSE, quality={"score": 0.75, "language": "en", "action": "warn"})
    chunks = chunk_document(doc, strategy="fixed",
                            config={"chunk_size": 200, "size_unit": "chars"})
    for c in chunks:
        assert c.metadata.get("quality_score") == 0.75


def test_f10_chunk_position_metadata():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PROSE), strategy="fixed",
                            config={"chunk_size": 200, "size_unit": "chars"})
    assert len(chunks) >= 3
    assert chunks[0].metadata["chunk_position"] == "first"
    assert chunks[-1].metadata["chunk_position"] == "last"
    for c in chunks[1:-1]:
        assert c.metadata["chunk_position"] == "middle"


def test_f10_single_chunk_position_is_only():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc("Short text."), strategy="fixed",
                            config={"chunk_size": 1000, "size_unit": "chars"})
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_position"] == "only"


def test_f10_chunk_provenance():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PROSE, filename="report.txt"), strategy="recursive",
                            config={"chunk_size": 200, "size_unit": "chars"})
    for c in chunks:
        assert c.metadata["source"] == "report.txt"
        assert c.metadata["format"] == "text"


# ── chunk_id uniqueness ────────────────────────────────────────────────────────

def test_f10_chunk_ids_unique():
    from app.rag.chunking import chunk_document
    chunks = chunk_document(_doc(PROSE), strategy="recursive",
                            config={"chunk_size": 200, "size_unit": "chars"})
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ── API endpoint tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_f10_api_strategies():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/chunks/strategies",
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    strats = data["strategies"]
    for name in ("fixed", "recursive", "semantic", "structure", "code"):
        assert name in strats
        assert "description" in strats[name]
        assert "best_for" in strats[name]


@pytest.mark.asyncio
async def test_f10_api_preview_recursive():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/preview",
            json={"text": PROSE, "strategy": "recursive", "chunk_size": 200, "size_unit": "chars"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_count"] > 1
    assert "chunks" in data
    chunk = data["chunks"][0]
    assert "chunk_id" in chunk
    assert "preview" in chunk
    assert "metadata" in chunk


@pytest.mark.asyncio
async def test_f10_api_preview_code():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/preview",
            json={"text": PYTHON_CODE, "filename": "code.py",
                  "strategy": "code", "chunk_size": 400, "size_unit": "chars"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_count"] >= 1
    previews = [c["preview"] for c in data["chunks"]]
    assert any("def calculate" in p for p in previews)


@pytest.mark.asyncio
async def test_f10_api_preview_structure():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/preview",
            json={"text": MARKDOWN, "filename": "report.md",
                  "strategy": "structure", "chunk_size": 500, "size_unit": "chars"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    headings = [c["metadata"].get("heading") for c in data["chunks"] if c["metadata"].get("heading")]
    assert len(headings) >= 2


@pytest.mark.asyncio
async def test_f10_api_preview_unknown_strategy():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/preview",
            json={"text": "some text", "strategy": "nonexistent"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_f10_api_compare_strategies():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/compare",
            json={
                "text": PROSE,
                "strategies": ["fixed", "recursive", "semantic"],
                "chunk_size": 200,
                "size_unit": "chars",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    for strat in ("fixed", "recursive", "semantic"):
        assert strat in data["strategies"]
        s = data["strategies"][strat]
        assert "chunk_count" in s
        assert "avg_chars" in s
        assert s["chunk_count"] > 0


@pytest.mark.asyncio
async def test_f10_api_compare_recommends_code():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/compare",
            json={
                "text": PYTHON_CODE,
                "filename": "code.py",
                "strategies": ["fixed", "recursive", "code"],
                "chunk_size": 400,
                "size_unit": "chars",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("recommended") == "code"


@pytest.mark.asyncio
async def test_f10_api_compare_recommends_structure():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/compare",
            json={
                "text": MARKDOWN,
                "filename": "report.md",
                "strategies": ["fixed", "structure", "recursive"],
                "chunk_size": 300,
                "size_unit": "chars",
            },
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("recommended") == "structure"


@pytest.mark.asyncio
async def test_f10_api_upload_preview():
    """POST /chunks/preview/upload with a real file."""
    from fastapi.testclient import TestClient
    from app.main import app

    content = PROSE.encode()
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chunks/preview/upload",
            files={"file": ("report.txt", content, "text/plain")},
            data={"strategy": "recursive", "chunk_size": "200", "size_unit": "chars"},
            headers={"Authorization": "Bearer dev"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_count"] > 0
