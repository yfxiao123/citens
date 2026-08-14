"""Pydantic data models for the whole pipeline.

The inheritance chain ``Paper → ScoredPaper → ExtractedPaper`` mirrors the
pipeline stages (each stage enriches the same object). Grounding models
(:class:`Chunk`, :class:`Claim`, :class:`Citation`, …) back the
"verifiable-citations" feature layered on in later phases.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


def _stable_id(*parts: str) -> str:
    """Deterministic short id from parts (used for paper / chunk identity)."""
    h = hashlib.sha1("\u241F".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


class Paper(BaseModel):
    """Bibliographic metadata for a single paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    source: str = ""
    citation_count: int = 0
    url: str = ""
    doi: Optional[str] = None
    pdf_url: Optional[str] = None  # open-access full-text PDF, if known

    @computed_field  # type: ignore[misc]
    @property
    def id(self) -> str:
        return _stable_id(self.doi or "", self.title.lower().strip())

    def brief(self) -> str:
        authors = ", ".join(self.authors[:5])
        if len(self.authors) > 5:
            authors += " et al."
        return (
            f"标题: {self.title}\n"
            f"作者: {authors}\n"
            f"年份: {self.year} | 来源: {self.source} | 引用: {self.citation_count}\n"
            f"摘要: {self.abstract[:300]}..."
        )


class ScoredPaper(Paper):
    """A paper with a relevance score and filtering rationale."""

    relevance_score: int = 0
    filter_reason: str = ""

    def brief(self) -> str:
        return (
            f"评分: {self.relevance_score}/5 | 理由: {self.filter_reason}\n"
            f"{super().brief()}"
        )


class ExtractedPaper(ScoredPaper):
    """A paper enriched with structured, abstract-grounded fields."""

    research_question: str = ""
    methodology: str = ""
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    relevance_to_topic: str = ""


# --- Theme organization --------------------------------------------------


class ThemeInfo(BaseModel):
    name: str
    description: str = ""
    paper_indices: list[int] = Field(default_factory=list)
    grouping_reason: str = ""
    logical_relations: str = ""


class ThemeStructure(BaseModel):
    themes: list[ThemeInfo] = Field(default_factory=list)


# --- Citation grounding (Phase 2/3) --------------------------------------


class ChunkKind(str, Enum):
    ABSTRACT = "abstract"
    SECTION = "section"
    FULLTEXT = "fulltext"


class Chunk(BaseModel):
    """A unit of source text that a claim can be grounded against."""

    paper_id: str
    chunk_id: str
    text: str
    kind: ChunkKind = ChunkKind.ABSTRACT


class Citation(BaseModel):
    """A reference-table entry rendering as ``[n]`` in the review."""

    index: int
    paper_id: str
    label: str  # human-readable reference string

    @property
    def marker(self) -> str:
        return f"[{self.index}]"


class Claim(BaseModel):
    """A single assertion in the review, tied to one or more citations."""

    text: str
    citation_indices: list[int] = Field(default_factory=list)
    section: str = ""


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"  # cited source has no ground text (e.g. no abstract)


class VerificationResult(BaseModel):
    claim_text: str
    verdict: Verdict
    citation_indices: list[int] = Field(default_factory=list)
    note: str = ""


class SynthesisResult(BaseModel):
    """Cross-paper critical analysis produced by the Synth agent."""

    consensus: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# --- Run metadata --------------------------------------------------------


class RunMeta(BaseModel):
    topic: str
    keywords: list[str] = Field(default_factory=list)
    total_papers: int = 0
    filtered_papers: int = 0
    themes: list[str] = Field(default_factory=list)
    review_path: str = ""
    run_dir: str = ""
    citation_precision: Optional[float] = None  # filled by Verifier (Phase 3)
