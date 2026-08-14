"""Citation grounding: source-text store, citation table, BibTeX, provenance."""

from __future__ import annotations

from litreview.grounding.chunkstore import ChunkStore
from litreview.grounding.citations import (
    CitationTable,
    build_provenance,
    parse_claims_from_review,
)
from litreview.grounding.enrichment import enrich_abstracts
from litreview.grounding.fetchlist import ensure_papers_dir, write_fetch_list
from litreview.grounding.fulltext import chunk_text, fetch_fulltext

__all__ = [
    "ChunkStore",
    "CitationTable",
    "build_provenance",
    "chunk_text",
    "ensure_papers_dir",
    "enrich_abstracts",
    "fetch_fulltext",
    "parse_claims_from_review",
    "write_fetch_list",
]
