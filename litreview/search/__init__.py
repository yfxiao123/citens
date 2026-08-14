"""Search sources. Importing this package registers all built-in sources."""

from __future__ import annotations

from litreview.search import arxiv, crossref, openalex, semantic_scholar  # noqa: F401
from litreview.search.base import (
    REGISTRY,
    SearchSource,
    blend_pool,
    deduplicate,
    register,
    search_papers,
)

__all__ = [
    "REGISTRY",
    "SearchSource",
    "blend_pool",
    "deduplicate",
    "register",
    "search_papers",
]
