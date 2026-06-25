"""Document quality assessment (F9).

Analyses extracted text and returns a ``QualityResult`` that describes how
usable the text is.  Called automatically by ``registry.ingest()`` so every
ingested document carries a ``metadata.quality`` block.

Quality score (0.0 → 1.0)
--------------------------
  - Starts at 1.0
  - Deductions for: too short, likely OCR noise, undetectable language,
    unusual average word length, very high non-printable character ratio.

Action thresholds
-----------------
  ≥ 0.7  →  "ok"     — proceed normally
  ≥ 0.3  →  "warn"   — index but show user a quality warning
  < 0.3  →  "reject" — likely noise; advise re-upload or manual review

Language detection uses ``langdetect`` (installed) with a graceful fallback
when the text is too short or ambiguous.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from pydantic import BaseModel


class QualityResult(BaseModel):
    score: float                    # 0.0 – 1.0
    action: str                     # "ok" | "warn" | "reject"
    readable: bool
    language: str | None            # ISO-639-1, e.g. "en", "fr"
    language_confidence: float      # 0.0 – 1.0 (0 when unknown)
    word_count: int
    char_count: int
    avg_word_length: float
    noise_ratio: float              # ratio of unusual chars  (0 = clean)
    flags: list[str]                # human-readable issues, e.g. "too_short"
    note: str | None                # one-line summary for the user


# ── helpers ────────────────────────────────────────────────────────────────────

_PRINTABLE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e -\udfff]")


def _noise_ratio(text: str) -> float:
    """Ratio of characters that are unlikely in normal prose."""
    if not text:
        return 1.0
    noise = sum(
        1 for c in text
        if not (c.isalpha() or c.isdigit() or c in " \n\t.,;:!?-'\"()[]{}/@#%&*+=_<>|\\~`^")
    )
    return noise / len(text)


def _detect_language(text: str) -> tuple[str | None, float]:
    """Return (lang_code, probability) using langdetect; (None, 0) on failure."""
    # Need at least ~20 chars for a reliable detection.
    sample = " ".join(text.split()[:200])  # first 200 words, keep it fast
    if len(sample) < 20:
        return None, 0.0
    try:
        from langdetect import detect_langs  # type: ignore[import]
        results = detect_langs(sample)
        if results:
            top = results[0]
            return top.lang, round(top.prob, 3)
    except Exception:
        pass
    return None, 0.0


# ── main function ─────────────────────────────────────────────────────────────

def assess_quality(text: str, metadata: dict[str, Any] | None = None) -> QualityResult:
    """Analyse ``text`` and return a :class:`QualityResult`."""
    metadata = metadata or {}
    flags: list[str] = []
    score = 1.0

    words = text.split() if text else []
    word_count = len(words)
    char_count = len(text)
    avg_word_len = (sum(len(w) for w in words) / word_count) if words else 0.0
    noise = _noise_ratio(text)

    # ── quality checks ────────────────────────────────────────────────────────

    if word_count < 5:
        flags.append("too_short")
        score -= 0.5
    elif word_count < 20:
        flags.append("very_short")
        score -= 0.2

    if noise > 0.4:
        flags.append("ocr_noise")
        score -= 0.4
    elif noise > 0.2:
        flags.append("possible_ocr_noise")
        score -= 0.15

    if avg_word_len < 2 and word_count > 5:
        flags.append("unusual_word_length_low")
        score -= 0.15
    elif avg_word_len > 15 and word_count > 5:
        flags.append("unusual_word_length_high")
        score -= 0.1

    # Propagate parser-level quality flags (e.g. ocr_unavailable).
    qflag = metadata.get("quality_flag")
    if qflag and qflag not in flags:
        flags.append(qflag)
        if qflag in ("ocr_unavailable", "transcription_unavailable"):
            score -= 0.2  # stub content, not real text

    # ── language detection ────────────────────────────────────────────────────
    language, lang_confidence = _detect_language(text)
    if language is None and word_count >= 5:
        flags.append("unknown_language")
        score -= 0.1

    # ── finalise ──────────────────────────────────────────────────────────────
    score = round(max(0.0, min(1.0, score)), 3)
    readable = score >= 0.5

    if score >= 0.7:
        action = "ok"
        note = None
    elif score >= 0.3:
        action = "warn"
        note = (
            f"Document quality is low (score {score:.2f}). "
            f"Issues: {', '.join(flags) or 'none'}. "
            "Results may vary — consider uploading a cleaner version."
        )
    else:
        action = "reject"
        note = (
            f"Document quality is very low (score {score:.2f}). "
            f"Issues: {', '.join(flags) or 'none'}. "
            "Text extraction likely failed. Please re-upload a higher-quality version."
        )

    return QualityResult(
        score=score,
        action=action,
        readable=readable,
        language=language,
        language_confidence=lang_confidence,
        word_count=word_count,
        char_count=char_count,
        avg_word_length=round(avg_word_len, 2),
        noise_ratio=round(noise, 3),
        flags=flags,
        note=note,
    )
