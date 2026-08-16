"""Pydantic data models for the whole pipeline.

The inheritance chain ``Paper → ScoredPaper → ExtractedPaper`` mirrors the
pipeline stages (each stage enriches the same object). Grounding models
(:class:`Chunk`, :class:`Claim`, :class:`Citation`, …) back the
"verifiable-citations" feature layered on in later phases.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, Field, computed_field, field_validator


def _stable_id(*parts: str) -> str:
    """Deterministic short id from parts (used for paper / chunk identity)."""
    h = hashlib.sha1("\u241F".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
    "info:doi/",
)


class Paper(BaseModel):
    """Bibliographic metadata for a single paper."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    source: str = ""
    citation_count: int = 0
    url: str = ""
    doi: str | None = None
    pdf_url: str | None = None  # open-access full-text PDF, if known
    venue: str = ""  # journal / conference / repository name, when known
    keywords: list[str] = Field(default_factory=list)  # author/indexer keywords
    subfield: str = ""  # fine-grained field tag (assigned at pool collection)
    # Author-engagement signals (filled by citens collect; 0 = unknown).
    # A first author with many works / a high h-index is 深耕此领域 —
    # one more quality proxy beyond paper-level citations.
    first_author_h_index: int = 0
    first_author_works: int = 0

    @field_validator("authors", mode="before")
    @classmethod
    def _dedupe_authors(cls, v):
        """Deduplicate author names, order-preserving.

        OpenAlex emits one ``authorships`` entry per (author, affiliation), so a
        single author with three affiliations appears three times — e.g.
        "Bence Toth, Bence Toth, Bence Toth et al." in the rendered references.
        Case-insensitive match on the collapsed name.
        """
        if not v:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for name in v:
            if name is None:
                continue
            s = " ".join(str(name).split())
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @field_validator("doi", mode="before")
    @classmethod
    def _normalize_doi(cls, v):
        """Strip URL forms (OpenAlex returns 'https://doi.org/10.x/...') so the
        bare DOI is usable in Unpaywall/Crossref/BibTeX everywhere downstream."""
        if not v:
            return v
        s = str(v).strip()
        low = s.lower()
        for p in _DOI_URL_PREFIXES:
            if low.startswith(p):
                s = s[len(p):]
                break
        return s or None

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
    venue_quartile: str = ""  # SJR quartile of the venue ("" if unknown)
    rank_score: float = 0.0  # composite retrieval score (see citens.ranking)

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
    quality: dict = Field(default_factory=dict)  # evidence level, method rigor, etc.


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
    intent: str = ""  # what this claim is trying to prove (from claim_intent_manifest)


class ClaimIntentManifest(BaseModel):
    """Pre-declared intents that the review aims to prove.

    Inspired by Imbad0202/ARS's claim_intent_manifest. Each intent is mapped
    to the claims that attempt to support it, enabling intent-claim alignment checks.
    """

    intents: list[str] = Field(default_factory=list)  # what the review aims to prove
    intent_to_claims: dict[str, list[int]] = Field(default_factory=dict)  # intent -> claim indices
    alignment_notes: dict[str, str] = Field(default_factory=dict)  # intent -> alignment status


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"  # cited source has no ground text (e.g. no abstract)


class RunMode(str, Enum):
    """Adaptive pipeline modes based on user intent."""

    QUICK_SCAN = "quick_scan"  # Fast overview, 1-2 rounds, summary output
    DEEP_REVIEW = "deep_review"  # Full methodology, multi-round reflection
    INTERACTIVE = "interactive"  # User-driven exploration with mid-stream adjustments


class VerificationResult(BaseModel):
    claim_text: str
    verdict: Verdict
    citation_indices: list[int] = Field(default_factory=list)
    note: str = ""
    defense_result: dict | None = None  # bidirectional verification result


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
    citation_precision: float | None = None  # filled by Verifier (Phase 3)
