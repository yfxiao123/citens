"""Search-source protocol, registry, and concurrent multi-source orchestration."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict

from litreview.models import Paper

REGISTRY: dict[str, type[SearchSource]] = {}


def register(name: str):
    """Class decorator: register a SearchSource under ``name``."""

    def deco(cls: type[SearchSource]) -> type[SearchSource]:
        REGISTRY[name] = cls
        return cls

    return deco


class SearchSource(ABC):
    """A pluggable academic-search backend."""

    name: str = "base"

    @abstractmethod
    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        """Return up to ``max_results`` papers across all keywords."""


def _enabled_sources(sources: list[str] | None) -> list[str]:
    from litreview.config import settings

    if sources:
        return sources
    configured = [s.strip() for s in settings.search_sources.split(",") if s.strip()]
    if not configured or configured == ["all"]:
        return list(REGISTRY)
    return configured


async def search_papers(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
) -> list[Paper]:
    """Query all enabled sources concurrently, then deduplicate."""
    names = _enabled_sources(sources)
    instances = [REGISTRY[n]() for n in names if n in REGISTRY]
    if not instances:
        return []

    per_source = max(max_results // max(len(instances), 1), 5)
    tasks = [src.search(keywords, per_source) for src in instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[Paper] = []
    for src, res in zip(instances, results, strict=False):
        if isinstance(res, BaseException):
            print(f"[{src.name}] search failed: {res}")
            continue
        all_papers.extend(res)

    return deduplicate(all_papers)


def _norm_title(title: str) -> str:
    return title.lower().strip().rstrip(".").strip()


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Dedup by DOI or normalized title; keep the highest-cited variant."""
    seen: dict[str, Paper] = {}
    for p in papers:
        key = p.doi or _norm_title(p.title)
        if key not in seen or p.citation_count > seen[key].citation_count:
            seen[key] = p
    return list(seen.values())


def blend_pool(papers: list[Paper], cap: int) -> list[Paper]:
    """Cap the candidate pool while preserving source diversity.

    A naive global sort by citation starves sources whose citations are 0
    (notably arXiv). We cap per source first, then trim to ``cap``.
    """
    if len(papers) <= cap:
        return papers
    by_src: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        by_src[p.source.split(" (")[0]].append(p)
    per_src = max(cap // max(len(by_src), 1), 1)
    selected: list[Paper] = []
    for group in by_src.values():
        selected += sorted(group, key=lambda p: p.citation_count, reverse=True)[:per_src]
    return selected[:cap]
