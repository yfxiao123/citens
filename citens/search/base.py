"""Search-source protocol, registry, and concurrent multi-source orchestration."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict

from citens.models import Paper

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
    from citens.config import settings

    if sources:
        return sources
    configured = [s.strip() for s in settings.search_sources.split(",") if s.strip()]
    if not configured or configured == ["all"]:
        return list(REGISTRY)
    return configured


async def search_papers_with_health(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
) -> tuple[list[Paper], dict[str, str]]:
    """Query all enabled sources concurrently with retries; deduplicate.

    Returns (papers, health) where health maps each source name to
    "ok" | "empty" | "failed: <reason>". Callers that only need papers should
    use :func:`search_papers`.
    """
    names = _enabled_sources(sources)
    instances = [REGISTRY[n]() for n in names if n in REGISTRY]
    health: dict[str, str] = {}
    if not instances:
        return [], health

    per_source = max(max_results // max(len(instances), 1), 5)

    async def _search_with_retry(src: SearchSource) -> list[Paper]:
        # S2 without an API key 429s routinely; one backoff retry saves the
        # whole source instead of silently shrinking the pool.
        import asyncio

        last_err: BaseException | None = None
        for attempt in range(3):
            try:
                return await src.search(keywords, per_source)
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(1.5 * (2**attempt))
        assert last_err is not None
        raise last_err

    tasks = [asyncio.create_task(_search_with_retry(src)) for src in instances]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers: list[Paper] = []
    for src, res in zip(instances, results, strict=False):
        if isinstance(res, BaseException):
            health[src.name] = f"failed: {res}"[:200]
            print(f"[{src.name}] search failed after retries: {res}")
            continue
        health[src.name] = "ok" if res else "empty"
        all_papers.extend(res)

    return deduplicate(all_papers), health


async def search_papers(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
) -> list[Paper]:
    """Query all enabled sources concurrently, then deduplicate."""
    papers, _health = await search_papers_with_health(keywords, max_results, sources)
    return papers


def _norm_title(title: str) -> str:
    return title.lower().strip().rstrip(".").strip()


def _fuzzy_title(title: str) -> str:
    """Aggressive normalization for near-duplicate titles: lowercase, keep
    alphanumerics only, sorted token set — resilient to punctuation, casing,
    and small word-order/stopword differences between preprint and published
    versions of the same paper."""
    import re

    toks = sorted(t for t in re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 2)
    return " ".join(toks)


def _shared_author(a: Paper, b: Paper) -> bool:
    """True when the two papers share at least one surname.

    Guards the fuzzy-title merge: similar titles by different authors are
    usually different papers (e.g. survey variants), while the arXiv version
    and the journal version of one paper share authors.
    """

    def surnames(names: list[str]) -> set[str]:
        return {n.split()[-1].lower() for n in names if n.split()}

    return bool(surnames(a.authors) & surnames(b.authors))


def _merge_variants(keep: Paper, drop: Paper) -> Paper:
    """Merge two versions of one paper, preferring the published record.

    The published (DOI-carrying) variant wins because its venue/citation
    metadata feeds ranking and BibTeX; the preprint variant's open-access
    pdf_url (arXiv) is carried over because fulltext grounding needs it.
    """
    preferred, other = (keep, drop) if keep.doi else (drop, keep)
    if other.pdf_url and not preferred.pdf_url:
        preferred.pdf_url = other.pdf_url
    if other.abstract and len(other.abstract) > len(preferred.abstract):
        preferred.abstract = other.abstract
    preferred.citation_count = max(preferred.citation_count, other.citation_count)
    return preferred


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Deduplicate by DOI or normalized title; keep the highest-cited variant.

    A second fuzzy pass merges arXiv preprints with their published versions:
    the preprint carries no DOI, so its exact key is the title while the
    published paper keys on its DOI — they never collide without fuzzy
    matching, and the same study ends up occupying two pool slots with split
    citation counts. Requires an identically-normalized title AND a shared
    author to fire (conservative — a false merge is worse than a duplicate).
    """
    seen: dict[str, Paper] = {}
    for p in papers:
        key = p.doi or _norm_title(p.title)
        if key not in seen or p.citation_count > seen[key].citation_count:
            seen[key] = p
    out = list(seen.values())

    # fuzzy pass: compare across different exact keys only
    merged: list[Paper] = []
    fuzzy_keys = [_fuzzy_title(p.title) for p in out]
    dropped: set[int] = set()
    for i in range(len(out)):
        if i in dropped:
            continue
        for j in range(i + 1, len(out)):
            if j in dropped or out[i].doi == out[j].doi:
                continue
            if fuzzy_keys[i] == fuzzy_keys[j] and _shared_author(out[i], out[j]):
                out[i] = _merge_variants(out[i], out[j])
                dropped.add(j)
        merged.append(out[i])
    return merged


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
