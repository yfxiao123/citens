"""Literature-pool collection: record first, recall later.

The systematic-review workflow this implements (the human way):

1.  SEARCH BROAD, RECORD EVERYTHING — for each keyword batch (multi-dimension
    queries + explicit survey-hunting queries), record structured entries:
    subfield, authors, year, abstract, keywords, citations, venue, SJR
    quartile, matched queries. NO full text yet.
2.  AUTHOR ENGAGEMENT — for the top entries, look up the first author's
    works count / h-index (深耕此领域 as a quality signal).
3.  PERSIST — append to ``data/litdb/<topic>.jsonl``; the pool accumulates
    across runs and collect sessions.
4.  RECALL LATER — ``load_pool`` feeds the pipeline's candidate pool;
    full text is fetched only for the papers that survive filtering, in
    batches, by the existing grounding machinery.

About WOB / Google Scholar / 知网: they have no public APIs (and scrape
hostilely). The same journals are indexed by OpenAlex/Crossref/S2, which we
already search; SJR quartiles proxy "high-quality journal". Records exported
from those sites can be merged with ``import_records``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx

from citens.config import settings
from citens.grounding.fulltext import slugify
from citens.models import Paper
from citens.search import search_papers

_AUTHORS_URL = "https://api.openalex.org/authors"


def pool_path(topic: str) -> Path:
    return Path(settings.litdb_dir) / f"{slugify(topic)}.jsonl"


def read_pool(topic: str) -> list[Paper]:
    """The accumulated pool for a topic (empty if never collected)."""
    p = pool_path(topic)
    if not p.is_file():
        return []
    out: list[Paper] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Paper(**json.loads(line)))
        except Exception:  # noqa: BLE001 — a bad line must not kill the pool
            continue
    return out


def append_pool(topic: str, papers: list[Paper]) -> int:
    """Dedup-merge papers into the topic's pool; returns the number ADDED."""
    existing = {pp.doi or pp.title.lower().strip() for pp in read_pool(topic)}
    added = 0
    path = pool_path(topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for p in papers:
            key = p.doi or p.title.lower().strip()
            if key in existing:
                continue
            existing.add(key)
            fh.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")
            added += 1
    return added


def import_records(topic: str, records: list[dict]) -> int:
    """Merge externally exported records (WOB / CNKI / Scholar exports)."""
    papers = []
    for r in records:
        try:
            papers.append(Paper(**r))
        except Exception:  # noqa: BLE001
            continue
    return append_pool(topic, papers)


def build_queries(topic: str, extra_queries: list[str] | None = None) -> list[str]:
    """Keyword batches: multi-dimension planner queries + survey hunting."""
    from citens.agents.planner import generate_keywords, generate_seed_papers

    queries = list(generate_keywords(topic))
    _, domain_terms = generate_seed_papers(topic)
    queries += domain_terms
    t = topic.strip()
    queries += [f"{t} survey", f"{t} review", f"{t} literature review"]
    if extra_queries:
        queries += extra_queries
    # dedupe, order-preserving
    seen: set[str] = set()
    out = []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out


async def _search_per_query(
    queries: list[str], per_query: int, sources: list[str] | None
) -> tuple[dict[str, Paper], dict[str, int]]:
    """One query at a time so every hit is ATTRIBUTED to the query that
    found it (query-level hit stats; merged search loses this)."""
    by_key: dict[str, Paper] = {}
    hits: dict[str, int] = {}
    for q in queries:
        papers = await search_papers([q], max_results=per_query, sources=sources)
        hits[q] = len(papers)
        for p in papers:
            key = p.doi or p.title.lower().strip()
            if key not in by_key:
                by_key[key] = p
    return by_key, hits


def _rewrite_pool(topic: str, papers: list[Paper]) -> None:
    path = pool_path(topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in papers:
            fh.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")


def _enrich_author_engagement(papers: list[Paper], top_n: int = 25) -> int:
    """Fill first-author works/h-index for the most-cited ``top_n`` papers
    that lack the signal.

    Name-based OpenAlex lookup — pragmatic (common-name collisions possible);
    a miss leaves the signal unknown rather than wrong.
    """
    enriched = 0
    candidates = [
        p for p in sorted(papers, key=lambda x: x.citation_count, reverse=True)
        if p.authors and p.first_author_works == 0
    ][:top_n]
    for p in candidates:
        name = p.authors[0]
        try:
            with httpx.Client(timeout=20) as client:
                r = client.get(
                    _AUTHORS_URL,
                    params={"search": name, "per_page": 1,
                            "select": "display_name,works_count,summary_stats"},
                )
                r.raise_for_status()
                res = r.json().get("results") or []
            if res:
                p.first_author_works = int(res[0].get("works_count") or 0)
                p.first_author_h_index = int(
                    (res[0].get("summary_stats") or {}).get("h_index") or 0
                )
                enriched += 1
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.05)  # polite
    return enriched


def collect(
    topic: str,
    target: int = 100,
    *,
    sources: list[str] | None = None,
    extra_queries: list[str] | None = None,
    enrich_authors: bool = True,
    on_progress=None,
) -> dict:
    """Build/extend the literature pool for a topic. Returns a summary.

    Full text is deliberately NOT fetched — that happens later, in batches,
    only for papers that survive the pipeline's filter.
    """
    queries = build_queries(topic, extra_queries)
    if on_progress:
        on_progress(f"共 {len(queries)} 条关键词批次（含综述定向查询）")

    by_key, hits = asyncio.run(
        _search_per_query(queries, per_query=max(target // max(len(queries), 1), 8), sources=sources)
    )
    papers = list(by_key.values())

    added = append_pool(topic, papers)

    # enrichment runs over the WHOLE pool (new finds + accumulated records
    # that still lack the signal), then persists — a second collect session
    # fills authors the first one didn't reach
    enriched = 0
    if enrich_authors:
        pool = read_pool(topic)
        enriched = _enrich_author_engagement(pool)
        if enriched:
            _rewrite_pool(topic, pool)
        if on_progress:
            on_progress(f"作者深耕信号已补全 {enriched} 篇（一作 works/h-index）")
    total = len(read_pool(topic))

    subfields: dict[str, int] = {}
    for p in read_pool(topic):
        k = p.subfield or "(未标注)"
        subfields[k] = subfields.get(k, 0) + 1

    return {
        "topic": topic,
        "queries": queries,
        "query_hits": hits,
        "found": len(papers),
        "added": added,
        "pool_total": total,
        "subfields": dict(sorted(subfields.items(), key=lambda kv: -kv[1])),
        "pool_path": str(pool_path(topic)),
    }
