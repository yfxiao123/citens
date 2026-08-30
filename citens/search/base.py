"""Search-source protocol, registry, and concurrent multi-source orchestration."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections import defaultdict

from citens.models import Paper

REGISTRY: dict[str, type[SearchSource]] = {}

# arXiv id from either form a Paper carries it in: the abs/pdf URL the arXiv
# source fills, or the 10.48550/arxiv.* DataCite DOI. Shared by the citation
# enrichment join and the bench's gold matching — they must extract the SAME
# id or enrichment and evaluation drift apart silently.
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", re.I)
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})", re.I)


def paper_arxiv_id(p: Paper) -> str | None:
    """The paper's arXiv id, or None when the record carries none."""
    for text in (p.url or "", p.doi or ""):
        m = _ARXIV_URL_RE.search(text) or _ARXIV_DOI_RE.search(text)
        if m:
            return m.group(1)
    return None

# per-source cap on concurrently in-flight queries. The fan-out used to be
# unbounded (12-16 queries fired at once per source); with concept-block
# planning plus the calibration waves query counts grew, and a burst of
# concurrent connections is exactly how polite APIs decide to throttle you.
SEARCH_CONCURRENCY = 6


def register(name: str):
    """Class decorator: register a SearchSource under ``name``."""

    def deco(cls: type[SearchSource]) -> type[SearchSource]:
        REGISTRY[name] = cls
        return cls

    return deco


class SearchSource(ABC):
    """A pluggable academic-search backend."""

    name: str = "base"

    def __init__(self) -> None:
        # RetrievalConstraints from the user's clarification answers (year
        # window, ...). Sources that support native filters apply them in
        # their queries; others may post-filter. None = unconstrained.
        self.constraints = None
        # per-query hit counts of the last search() — filled by each source,
        # read by search_round for the zero-hit calibration loop. Optional
        # for third-party sources (getattr-guarded).
        self.query_stats: dict[str, int] = {}

    def set_constraints(self, constraints) -> None:
        self.constraints = constraints
        self.query_stats = {}

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


async def search_round(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
    constraints=None,
) -> tuple[list[Paper], dict[str, str], dict[str, int]]:
    """One search wave: all enabled sources concurrently, with retries.

    Returns (papers, health, query_stats) where query_stats maps each query
    to its total hit count across sources that responded — the zero-hit
    queries feed the synonym-swap calibration loop. Constraints (year
    window from clarification answers) are compiled into each source's
    native filter syntax.
    """
    names = _enabled_sources(sources)
    instances = [REGISTRY[n]() for n in names if n in REGISTRY]
    health: dict[str, str] = {}
    if not instances:
        return [], health, {}

    for src in instances:
        # duck-typed: REGISTRY accepts plain classes too (see the health
        # tests) — constraints are opt-in for sources that understand them
        set_c = getattr(src, "set_constraints", None)
        if callable(set_c):
            set_c(constraints)

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
    stats: dict[str, int] = {}
    for src, res in zip(instances, results, strict=False):
        if isinstance(res, BaseException):
            health[src.name] = f"failed: {res}"[:200]
            print(f"[{src.name}] search failed after retries: {res}")
            continue
        health[src.name] = "ok" if res else "empty"
        all_papers.extend(res)
        # a failed source must not fake zero-hits: only responding sources
        # contribute counts
        for q, n in getattr(src, "query_stats", {}).items():
            stats[q] = stats.get(q, 0) + n

    return deduplicate(all_papers), health, stats


async def search_papers_with_health(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
    constraints=None,
) -> tuple[list[Paper], dict[str, str]]:
    """Query all enabled sources concurrently with retries; deduplicate.

    Returns (papers, health) where health maps each source name to
    "ok" | "empty" | "failed: <reason>". Callers that need per-query hit
    counts too should use :func:`search_round`.
    """
    papers, health, _stats = await search_round(keywords, max_results, sources, constraints)
    return papers, health


async def search_papers(
    keywords: list[str],
    max_results: int = 60,
    sources: list[str] | None = None,
    constraints=None,
) -> list[Paper]:
    """Query all enabled sources concurrently, then deduplicate."""
    papers, _health = await search_papers_with_health(
        keywords, max_results, sources, constraints
    )
    return papers


def _norm_title(title: str) -> str:
    return title.lower().strip().rstrip(".").strip()


def _fuzzy_title(title: str) -> str:
    """Aggressive normalization for near-duplicate titles: lowercase, keep
    alphanumerics only, sorted token set — resilient to punctuation, casing,
    and small word-order/stopword differences between preprint and published
    versions of the same paper."""
    toks = sorted(t for t in re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 2)
    return " ".join(toks)


def _shared_author(a: Paper, b: Paper) -> bool:
    """True when the two papers share at least one surname.

    Guards the fuzzy-title merge: similar titles by different authors are
    usually different papers (e.g. survey variants), while the arXiv version
    and the journal version of one paper share authors.
    """

    def surnames(names: list[str]) -> set[str]:
        out: set[str] = set()
        for n in names:
            n = n.strip()
            if not n:
                continue
            # "Cai, H" (repository house style) and "Han Cai" (OpenAlex)
            # must both yield "cai" — bench run 2026-08-30: a UCL-record
            # duplicate survived dedup because only "Firstname Lastname"
            # was understood
            if "," in n:
                out.add(n.split(",", 1)[0].strip().lower())
            else:
                out.add(n.split()[-1].lower())
        return out

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
    # retrieval provenance is cumulative — a paper found by BOTH the
    # preprint's query and the published version's query credits both
    preferred.matched_queries = list(
        dict.fromkeys((preferred.matched_queries or []) + (other.matched_queries or []))
    )
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
        if key not in seen:
            seen[key] = p
            continue
        prev = seen[key]
        merged_queries = list(
            dict.fromkeys((prev.matched_queries or []) + (p.matched_queries or []))
        )
        if p.citation_count > prev.citation_count:
            p.matched_queries = merged_queries
            seen[key] = p
        else:
            prev.matched_queries = merged_queries
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
    """Cap the candidate pool while preserving source diversity AND the
    field's spine.

    A naive global sort by citation starves sources whose citations are 0
    (notably arXiv). But a pure per-source cap proved coverage-blind the
    other way (measured on the RAG bench run): a field-defining work
    captured via the arXiv leg carries citation_count=0, sorts arbitrarily
    within its source group, and gets cut. Half the cap is reserved for
    the globally most-cited papers (the spine every review must see); the
    rest is per-source diversity fill.
    """
    if len(papers) <= cap:
        return papers
    spine_n = max(cap // 2, 1)
    spine = sorted(papers, key=lambda p: p.citation_count, reverse=True)[:spine_n]
    spine_ids = {id(p) for p in spine}
    rest_cap = cap - len(spine)
    by_src: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        if id(p) not in spine_ids:
            by_src[p.source.split(" (")[0]].append(p)
    per_src = max(rest_cap // max(len(by_src), 1), 1)
    selected: list[Paper] = list(spine)
    for group in by_src.values():
        selected += sorted(group, key=lambda p: p.citation_count, reverse=True)[:per_src]
    return selected[:cap]
