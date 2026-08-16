"""Search sources. Importing this package registers all built-in sources."""

from __future__ import annotations

from citens.search import arxiv, crossref, openalex, semantic_scholar  # noqa: F401
from citens.search.base import (
    REGISTRY,
    SearchSource,
    blend_pool,
    deduplicate,
    register,
    search_papers,
    search_papers_with_health,
)

__all__ = [
    "REGISTRY",
    "SearchSource",
    "blend_pool",
    "deduplicate",
    "register",
    "search_papers",
    "search_papers_with_health",
]
