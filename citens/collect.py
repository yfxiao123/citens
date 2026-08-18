"""Literature-pool collection: record first, recall later.

The systematic-review workflow this implements (the human way):

1.  SEARCH BROAD, RECORD EVERYTHING — for each keyword batch (multi-dimension
    queries + explicit survey-hunting queries), record structured entries:
    subfield, authors, year, abstract, keywords, citations, venue, matched
    queries, review flag. NO full text yet.
2.  BACKFILL — DOI-batched OpenAlex lookups fill subfield/keywords/author IDs
    for records from sources that don't carry them.
3.  FIELD-CONSTRAINED SECOND PASS — broad single-concept queries are re-run
    constrained to the topic's own OpenAlex topic IDs, so generic terms
    ("adverse selection") stop pulling in off-field classics.
4.  AUTHOR ENGAGEMENT (深耕) — in-topic works count for the first/last
    author (author.id + title.search), not the merged-author total.
5.  PERSIST — append to ``data/litdb/<topic>.jsonl``; the pool accumulates
    across runs and collect sessions.
6.  RECALL LATER — ``recall_from_pool`` BM25-preselects pool records for the
    pipeline; full text is fetched only post-filter, in batches.

WOB / Google Scholar / 知网 have no public APIs; their exports merge via
``import_records``. The same journals are indexed by our API sources.
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

_WORKS_URL = "https://api.openalex.org/works"
_AUTHORS_URL = "https://api.openalex.org/authors"
_HEADERS = {"User-Agent": "CiteLens/0.1 (open literature-review agent)"}


def _oa_get_sync(url: str, params: dict) -> httpx.Response:
    """OpenAlex GET with polite-pool + exponential backoff.

    A collect session makes dozens of OpenAlex calls in quick succession
    (search passes, DOI backfill batches, author lookups) — without retry,
    a single 429 silently zeroes an entire enrichment phase.
    """
    import time as _time

    if settings.openalex_email:
        params = {**params, "mailto": settings.openalex_email}
    last: Exception | None = None
    for attempt in range(4):
        resp = httpx.get(url, params=params, timeout=30, headers=_HEADERS)
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        last = RuntimeError(f"OpenAlex {resp.status_code}")
        _time.sleep(1.5 * (2**attempt))
    raise last or RuntimeError("OpenAlex unavailable")


async def _oa_get_async(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    if settings.openalex_email:
        params = {**params, "mailto": settings.openalex_email}
    import asyncio as _aio

    last: Exception | None = None
    for attempt in range(4):
        resp = await client.get(url, params=params)
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        last = RuntimeError(f"OpenAlex {resp.status_code}")
        await _aio.sleep(1.5 * (2**attempt))
    raise last or RuntimeError("OpenAlex unavailable")


# --- pool persistence --------------------------------------------------------


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
    """Dedup-merge papers into the topic's pool; returns the number ADDED.

    An existing record absorbs new metadata (subfield/keywords/author
    signals/matched queries) but never loses fields it already has.
    """
    existing = {pp.doi or pp.title.lower().strip(): pp for pp in read_pool(topic)}
    added = 0
    path = pool_path(topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in papers:
        key = p.doi or p.title.lower().strip()
        old = existing.get(key)
        if old is not None:
            merged = old.model_copy(
                update={
                    "subfield": old.subfield or p.subfield,
                    "keywords": list(dict.fromkeys(old.keywords + p.keywords))[:16],
                    "first_author_h_index": max(
                        old.first_author_h_index, p.first_author_h_index
                    ),
                    "first_author_works": max(old.first_author_works, p.first_author_works),
                    "author_field_works": max(
                        old.author_field_works, p.author_field_works
                    ),
                    "matched_queries": list(
                        dict.fromkeys(old.matched_queries + p.matched_queries)
                    )[:12],
                    "is_review": old.is_review or p.is_review,
                    "volume": old.volume or p.volume,
                    "issue": old.issue or p.issue,
                    "pages": old.pages or p.pages,
                    "citation_count": max(old.citation_count, p.citation_count),
                    "abstract": p.abstract if len(p.abstract) > len(old.abstract) else old.abstract,
                    "pdf_url": old.pdf_url or p.pdf_url,
                }
            )
            existing[key] = merged
            continue
        existing[key] = p
        added += 1
    _rewrite_pool(topic, list(existing.values()))
    return added


def _rewrite_pool(topic: str, papers: list[Paper]) -> None:
    path = pool_path(topic)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for p in papers:
            fh.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")


def import_records(topic: str, records: list[dict]) -> int:
    """Merge externally exported records (WOB / CNKI / Scholar exports)."""
    papers = []
    for r in records:
        try:
            papers.append(Paper(**r))
        except Exception:  # noqa: BLE001
            continue
    return append_pool(topic, papers)


# --- query building ------------------------------------------------------------


def build_queries(
    topic: str,
    extra_queries: list[str] | None = None,
    profile_name: str = "",
) -> tuple[list[str], list[str]]:
    """Keyword batches: multi-dimension planner queries + survey hunting.

    Returns (queries, broad_queries): the seed-paper agent's domain TERMS are
    single concepts ("adverse selection") that match whole disciplines —
    they are only safe to search field-constrained, so they are reported
    separately for the second pass.
    """
    from citens.agents.planner import generate_keywords, generate_seed_papers

    queries = list(generate_keywords(topic))
    _, domain_terms = generate_seed_papers(topic)
    queries += domain_terms
    t = topic.strip()
    queries += [f"{t} survey", f"{t} review", f"{t} literature review"]
    if extra_queries:
        queries += extra_queries
    seen: set[str] = set()
    out = []
    for q in queries:
        k = q.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    if profile_name:
        from citens.profiles import load_profile, merge_profile_terms

        prof = load_profile(profile_name)
        if prof is not None:
            prior = set(q.lower() for q in out)
            out = merge_profile_terms(out, prof)
            # only NEWLY-added glossary concepts go to the constrained pass —
            # a planner query that coincides with a term stays free-search
            domain_terms = domain_terms + [
                t for t in prof.domain_terms if t.lower() not in prior
            ]
    broad = [q for q in domain_terms if q in out] + [
        q for q in out if _is_broad(q) and q not in domain_terms
    ]
    return out, broad


def _is_broad(query: str) -> bool:
    """Single-concept queries match whole disciplines, not the topic —
    they belong in the field-constrained pass, not the free one."""
    return len(query.split()) < 2


# --- search passes ---------------------------------------------------------------


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
            # accumulate attribution on the kept record
            if q not in by_key[key].matched_queries:
                by_key[key].matched_queries = [*by_key[key].matched_queries, q]
    return by_key, hits


def _field_topic_ids(top: int = 3) -> list[str]:
    """The topic's dominant OpenAlex topic IDs, harvested by the backfill
    pass (module-level handoff — deliberately not persisted on Paper)."""
    return list(_TOPIC_IDS_CACHE.get("ids", []))[:top]


_TOPIC_IDS_CACHE: dict[str, list[str]] = {}


async def _field_constrained_pass(
    broad_queries: list[str], topic_ids: list[str], per_query: int
) -> list[Paper]:
    """Re-run broad queries constrained to the topic's own field.

    ``title.search:'adverse selection'`` alone pulls Card (1990) into an
    order-book pool; the same query ANDed with the field's topic IDs keeps
    the hits inside the discipline. Records are tagged with the query.
    """
    if not broad_queries or not topic_ids:
        return []
    filter_base = ",".join(f"topics.id:{tid}" for tid in topic_ids)

    async def _one(client: httpx.AsyncClient, q: str) -> list[Paper]:
        from citens.search.openalex import OpenAlexSearcher

        params: dict[str, str | int] = {
            "filter": f"title.search:{q},{filter_base}",
            "per_page": min(per_query, 25),
            "sort": "cited_by_count:desc",
            "select": (
                "id,title,authorships,publication_year,abstract_inverted_index,"
                "cited_by_count,doi,primary_location,open_access,topics,keywords,biblio,locations"
            ),
        }
        try:
            resp = await _oa_get_async(client, _WORKS_URL, params)
            resp.raise_for_status()
            out: list[Paper] = []
            for w in resp.json().get("results", []):
                p = OpenAlexSearcher.to_paper(w)
                p.matched_queries = [f"{q} (field-constrained)"]
                out.append(p)
            return out
        except Exception:  # noqa: BLE001
            return []

    async with httpx.AsyncClient(timeout=30, headers=_HEADERS) as client:
        results = await asyncio.gather(
            *(_one(client, q) for q in broad_queries), return_exceptions=True
        )
    out: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


async def _review_pass(topic_queries: list[str], per_query: int = 10) -> list[Paper]:
    """Explicit ``type:review`` hunting on OpenAlex; hits are flagged."""
    from citens.search.openalex import OpenAlexSearcher

    async def _one(client: httpx.AsyncClient, q: str) -> list[Paper]:
        params: dict[str, str | int] = {
            "filter": f"title.search:{q},type:review",
            "per_page": min(per_query, 25),
            "select": (
                "id,title,authorships,publication_year,abstract_inverted_index,"
                "cited_by_count,doi,primary_location,open_access,topics,keywords,biblio,locations"
            ),
        }
        try:
            resp = await _oa_get_async(client, _WORKS_URL, params)
            resp.raise_for_status()
            out = []
            for w in resp.json().get("results", []):
                p = OpenAlexSearcher.to_paper(w)
                p.is_review = True
                p.matched_queries = [f"{q} (type:review)"]
                out.append(p)
            return out
        except Exception:  # noqa: BLE001
            return []

    async with httpx.AsyncClient(timeout=30, headers=_HEADERS) as client:
        results = await asyncio.gather(
            *(_one(client, q) for q in topic_queries[:5]), return_exceptions=True
        )
    out: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


# --- metadata backfill (#2) and author engagement (#3) --------------------------


def _backfill_metadata(papers: list[Paper], batch: int = 40) -> int:
    """Fill subfield/keywords (+ harvest topic IDs) via DOI-batched OpenAlex.

    Records from S2/Crossref/arXiv carry no taxonomy; one batched call per
    ``batch`` DOIs fills them from the OpenAlex work record.
    """
    need = [p for p in papers if p.doi and (
        not p.subfield or not p.keywords or not p.volume or not p.pages
    )]
    filled = 0
    topic_counts: dict[str, int] = {}
    for start in range(0, len(need), batch):
        chunk = need[start : start + batch]
        dois = "|".join(p.doi for p in chunk if p.doi)  # type: ignore[arg-type]
        try:
            r = _oa_get_sync(
                _WORKS_URL,
                {
                    "filter": f"doi:{dois}",
                    "per_page": batch,
                    "select": "doi,topics,keywords,authorships,biblio,primary_location,locations",
                },
            )
            r.raise_for_status()
            by_doi = {w.get("doi", "").replace("https://doi.org/", ""): w
                      for w in r.json().get("results", [])}
        except Exception:  # noqa: BLE001
            continue
        for p in chunk:
            w = by_doi.get(p.doi or "")
            if not w:
                continue
            changed = False
            if not p.subfield:
                for t in w.get("topics") or []:
                    sf = (t.get("subfield") or {})
                    if sf.get("display_name"):
                        p.subfield = sf["display_name"]
                        changed = True
                        break
                    if t.get("id"):
                        topic_counts[t["id"]] = topic_counts.get(t["id"], 0) + 1
            else:
                for t in w.get("topics") or []:
                    if t.get("id"):
                        topic_counts[t["id"]] = topic_counts.get(t["id"], 0) + 1
            if not p.keywords:
                kws = [k.get("display_name", "") for k in w.get("keywords") or []
                       if k.get("display_name")][:12]
                if kws:
                    p.keywords = kws
                    changed = True
            biblio = w.get("biblio") or {}
            if not p.venue:
                from citens.search.openalex import best_venue

                venue = best_venue(w)
                if venue and venue != "OpenAlex":
                    p.venue = venue
                    changed = True
            if not p.volume and biblio.get("volume"):
                p.volume = str(biblio["volume"])
                changed = True
            if not p.issue and biblio.get("issue"):
                p.issue = str(biblio["issue"])
                changed = True
            if not p.pages:
                fp = str(biblio.get("first_page") or "")
                lp = str(biblio.get("last_page") or "")
                pages = f"{fp}-{lp}".strip("-") if (fp or lp) else ""
                if pages:
                    p.pages = pages
                    changed = True
            if changed:
                filled += 1
        time.sleep(0.05)
    _TOPIC_IDS_CACHE["ids"] = [
        tid for tid, _ in sorted(topic_counts.items(), key=lambda kv: -kv[1])
    ]
    return filled


def _field_works_count(author_id: str, topic_query: str) -> int:
    """The author's works matching the topic (深耕此领域, not academia-wide)."""
    try:
        r = _oa_get_sync(
            _WORKS_URL,
            {
                "filter": f"author.id:{author_id},title.search:{topic_query}",
                "select": "id",
                "per_page": 1,
            },
        )
        r.raise_for_status()
        return int((r.json().get("meta") or {}).get("count") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _enrich_author_engagement(
    papers: list[Paper], topic_query: str, top_n: int = 25
) -> int:
    """Fill in-topic works counts for the most-cited records lacking them.

    Uses the OpenAlex author IDs harvested by name lookup (first OR last
    author — the deeply-engaged one is often the senior author).
    """
    enriched = 0
    candidates = [
        p for p in sorted(papers, key=lambda x: x.citation_count, reverse=True)
        if p.authors and p.author_field_works == 0
    ][:top_n]
    for p in candidates:
        for name in dict.fromkeys([p.authors[0], p.authors[-1]]):
            try:
                r = _oa_get_sync(
                    _AUTHORS_URL,
                    {"search": name, "per_page": 1,
                     "select": "display_name,works_count,summary_stats"},
                )
                r.raise_for_status()
                res = r.json().get("results") or []
            except Exception:  # noqa: BLE001
                continue
            if not res:
                continue
            a = res[0]
            p.first_author_works = int(a.get("works_count") or 0)
            p.first_author_h_index = int(
                (a.get("summary_stats") or {}).get("h_index") or 0
            )
            aid = a.get("id", "").rsplit("/", 1)[-1]
            fw = _field_works_count(aid, topic_query) if aid else 0
            if fw > p.author_field_works:
                p.author_field_works = fw
            enriched += 1
            break
        time.sleep(0.05)
    return enriched


# --- recall (#4) and audit (#6) --------------------------------------------------


def _pool_key(p: Paper) -> str:
    return p.doi or p.title.lower().strip()


def _emb_path(topic: str) -> Path:
    return pool_path(topic).with_suffix(".emb.json")


def embed_pool(topic: str) -> int:
    """Embed title+abstract of every pool record into a persistent index.

    Skipped entirely (returns 0) when no embedding model is configured —
    the pool then serves BM25-only recall. Re-embeds only records missing
    from the index (disk cache makes repeat calls free).
    """
    from citens.grounding.retrieval import embed_texts

    pool = read_pool(topic)
    index: dict[str, list[float]] = {}
    if _emb_path(topic).is_file():
        try:
            index = json.loads(_emb_path(topic).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            index = {}
    missing = [p for p in pool if _pool_key(p) not in index]
    if missing:
        vecs = embed_texts([f"{p.title}. {p.abstract[:600]}" for p in missing])
        if vecs is None:
            return 0
        for p, v in zip(missing, vecs, strict=False):
            index[_pool_key(p)] = v
    _emb_path(topic).parent.mkdir(parents=True, exist_ok=True)
    _emb_path(topic).write_text(json.dumps(index), encoding="utf-8")
    return len(missing)


def _norm_venue(venue: str) -> str:
    from citens.ranking import _norm

    return _norm(venue or "")


def recall_from_pool(
    topic: str,
    queries: list[str],
    k: int,
    constraints=None,
    venue_whitelist: set[str] | None = None,
) -> list[Paper]:
    """Hybrid pre-recall: BM25 + vector (RRF-fused), top-k pool records.

    Lexical catches exact terminology (LOB, OFI), embeddings catch semantic
    neighbors worded differently — Reciprocal Rank Fusion merges the two
    ranked lists without needing score calibration. Falls back to BM25-only
    when no embedding index/model is available; the rest of the pool stays
    a deep reservoir (nothing is deleted).

    ``constraints`` (search.filters.RetrievalConstraints) applies the run's
    clarification answers at recall time — a year window and, in strict mode,
    the venue whitelist — so candidates can SATISFY the filter instead of
    being killed by it (the "仅顶刊+近5年 -> 6 篇通过" failure mode).
    """
    from citens.grounding.retrieval import bm25_rank_texts, cosine, embed_texts

    pool = read_pool(topic)
    if constraints is not None:
        pool = [p for p in pool if constraints.matches_paper(p)]
        if constraints.venue_strict and venue_whitelist:
            # reviews stay citable even off-whitelist: they are the field's
            # maps, and the writer only uses them as context anyway
            pool = [
                p for p in pool
                if p.is_review or _norm_venue(p.venue) in venue_whitelist
            ]
    if len(pool) <= k:
        return pool
    texts = [
        f"{p.title} {p.abstract[:400]} {' '.join(p.keywords)} {p.subfield}"
        for p in pool
    ]
    query = " ".join(queries)
    bm25_order = bm25_rank_texts(texts, query)

    # vector order: persistent pool index + one query embedding
    vec_order: list[int] = []
    index: dict[str, list[float]] = {}
    if _emb_path(topic).is_file():
        try:
            index = json.loads(_emb_path(topic).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            index = {}
    if index:
        qv = embed_texts([query])
        if qv is not None:
            scored = [
                (cosine(index[_pool_key(p)], qv[0]), i)
                for i, p in enumerate(pool)
                if _pool_key(p) in index
            ]
            vec_order = [i for _s, i in sorted(scored, reverse=True)]

    # RRF fuse: 1/(rrf_k + rank) per list that ranked the record
    rrf_k = 60
    fused: dict[int, float] = {}
    for rank, i in enumerate(bm25_order):
        fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
    for rank, i in enumerate(vec_order):
        fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
    order = sorted(fused, key=lambda i: fused[i], reverse=True)

    # reviews always survive the pre-recall; the fused order fills the rest
    picked: dict[str, Paper] = {}
    for p in (x for x in pool if x.is_review):
        picked[_pool_key(p)] = p
    for i in order:
        if len(picked) >= k:
            break
        p = pool[i]
        picked.setdefault(_pool_key(p), p)
    return list(picked.values())


def audit_recall(topic: str, top_reviews: int = 3) -> dict:
    """Recall vs reality: how much of the top surveys' bibliographies the
    pool already covers. Survey reference lists are expert-curated paper
    lists — the cheapest external recall anchor there is."""
    from citens.search.snowball import _openalex_references

    pool = read_pool(topic)
    known = {p.doi for p in pool if p.doi} | {
        p.title.lower().strip() for p in pool
    }
    reviews = sorted(
        [p for p in pool if p.is_review and p.doi], key=lambda x: -x.citation_count
    )[:top_reviews]
    per_review: list[dict] = []
    total_refs = 0
    total_hit = 0
    for rv in reviews:
        works = asyncio.run(_openalex_references(rv.doi or "", limit=25))  # type: ignore[arg-type]
        hit = 0
        for w in works:
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            title = (w.get("title") or "").lower().strip()
            if doi in known or title in known:
                hit += 1
        per_review.append(
            {"review": rv.title, "refs_checked": len(works), "in_pool": hit}
        )
        total_refs += len(works)
        total_hit += hit
    return {
        "topic": topic,
        "pool_size": len(pool),
        "reviews_checked": len(reviews),
        "refs_checked": total_refs,
        "in_pool": total_hit,
        "coverage": round(total_hit / total_refs, 3) if total_refs else None,
        "per_review": per_review,
    }


# --- entry point -------------------------------------------------------------------


def collect(
    topic: str,
    target: int = 100,
    *,
    sources: list[str] | None = None,
    extra_queries: list[str] | None = None,
    enrich_authors: bool = True,
    on_progress=None,
    profile: str | None = None,
) -> dict:
    """Build/extend the literature pool for a topic. Returns a summary.

    Full text is deliberately NOT fetched — that happens later, in batches,
    only for papers that survive the pipeline's filter.
    """
    from citens.profiles import load_profile, order_sources

    prof = load_profile(profile or settings.profile)
    # domain-preferred source order (finance: journal records win dedup)
    sources = order_sources(sources, prof)
    queries, broad = build_queries(topic, extra_queries, profile or settings.profile)
    if on_progress:
        on_progress(
            f"共 {len(queries)} 条关键词批次（含综述定向查询；"
            f"{len(broad)} 条宽概念词仅限场域内检索）"
        )
    free = [q for q in queries if q not in broad]

    per_query = max(target // max(len(queries), 1), 8)
    by_key, hits = asyncio.run(
        _search_per_query(free or queries, per_query=per_query, sources=sources)
    )
    papers = list(by_key.values())
    if on_progress:
        on_progress(f"自由检索 {len(free or queries)} 条查询 · 命中 {len(papers)} 篇")

    # adaptive narrowing (nature-academic-search's ">500 results → add
    # filters" rule, scaled to our caps): a query whose result list saturated
    # the per-query cap has unbounded recall — push it into the
    # field-constrained pass rather than trusting free-search ranking alone
    saturated = [
        q for q in (free or queries)
        if hits.get(q, 0) >= per_query and q not in broad
    ]
    broad = broad + saturated
    if saturated and on_progress:
        on_progress(
            f"{len(saturated)} 条查询命中数触顶（召回过宽），转入场域收窄二轮"
        )

    total_added = append_pool(topic, papers)

    # backfill taxonomy (subfield/keywords) — also harvests the field's
    # dominant topic IDs for the constrained pass
    pool = read_pool(topic)
    filled = _backfill_metadata(pool)
    if filled:
        _rewrite_pool(topic, pool)
    if on_progress:
        on_progress(f"元数据回填 {filled} 篇（subfield/keywords）")

    # field-constrained second pass for the broad queries
    topic_ids = _field_topic_ids()
    if broad and topic_ids:
        constrained = asyncio.run(
            _field_constrained_pass(broad, topic_ids, per_query=15)
        )
        added_c = append_pool(topic, constrained)
        total_added += added_c
        if on_progress:
            on_progress(
                f"场域收窄检索 {len(broad)} 条宽查询 · 新增 {added_c} 篇（限定 topics.id）"
            )

    # survey-hunting pass (type:review) — flagged records
    topic_core = next((q for q in queries if q not in broad), queries[0] if queries else "")
    reviews = asyncio.run(_review_pass([topic_core, *broad][:5]))
    total_added += append_pool(topic, reviews)
    if on_progress:
        on_progress("综述定向检索（type:review）完成")

    # author engagement over the whole pool
    enriched = 0
    if enrich_authors:
        pool = read_pool(topic)
        enriched = _enrich_author_engagement(pool, topic_core)
        if enriched:
            _rewrite_pool(topic, pool)
        if on_progress:
            on_progress(f"作者深耕信号已补全 {enriched} 篇（场域内 works/h-index）")

    # hybrid-recall index: embed title+abstract of the whole pool when an
    # embedding model is configured (vector channel of RRF recall)
    n_emb = 0
    if settings.embedding_model:
        try:
            n_emb = embed_pool(topic)
        except Exception as e:  # noqa: BLE001
            if on_progress:
                on_progress(f"嵌入索引失败（回退 BM25 召回）: {e}")

    final_pool = read_pool(topic)
    subfields: dict[str, int] = {}
    for p in final_pool:
        k = p.subfield or "(未标注)"
        subfields[k] = subfields.get(k, 0) + 1

    return {
        "topic": topic,
        "queries": queries,
        "query_hits": hits,
        "found": len(papers),
        "added": total_added,
        "pool_total": len(final_pool),
        "n_reviews": sum(1 for p in final_pool if p.is_review),
        "n_embedded": n_emb,
        "subfields": dict(sorted(subfields.items(), key=lambda kv: -kv[1])),
        "pool_path": str(pool_path(topic)),
    }
