"""Citation-graph + semantic snowballing: expand the candidate pool.

Three directions:
- BACKWARD: fetch the references of a top paper (find its intellectual roots,
  canonical predecessors we may have missed).
- FORWARD: fetch papers that cite a top paper (find follow-up work, newer
  developments, empirical validations).
- RELATED: OpenAlex related_works — semantic neighbors that are NOT on the
  citation chain. Citation edges miss same-topic papers that simply haven't
  cited each other (parallel work, different sub-community); the similarity
  layer recovers them.

Every direction is OpenAlex-first with a Semantic Scholar fallback. OpenAlex's
free tier has a DAILY BUDGET, and once it runs out every direction silently
returned zero (measured live: "Insufficient budget … resets at midnight UTC") —
the single-provider dependency killed snowball for the rest of the day. S2
serves references/citations from its graph API and related papers from its
recommendations endpoint. Results are Papers with a `snowball_from` marker so
provenance is auditable; `·s2` in the marker marks fallback-served candidates.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

import httpx

from citens import cache
from citens.config import settings
from citens.models import Paper

_OPENALEX_WORKS = "https://api.openalex.org/works"
_S2_GRAPH = "https://api.semanticscholar.org/graph/v1/paper"
_S2_RECS = "https://api.semanticscholar.org/recommendations/v1/papers"
_S2_FIELDS = "title,authors,year,abstract,citationCount,externalIds,venue,url"
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})", re.I)


def _s2_paper_id(doi: str) -> str:
    """S2 paper id for a DOI. S2 does not resolve arXiv's DataCite DOIs
    (DOI:10.48550/arxiv.* returns 404) — those papers are only reachable
    as ARXIV:<id>."""
    m = _ARXIV_DOI_RE.search(doi)
    return f"ARXIV:{m.group(1)}" if m else f"DOI:{doi}"


def _polite_params() -> dict:
    if settings.openalex_email:
        return {"mailto": settings.openalex_email}
    return {}


async def _openalex_work_id(doi: str) -> str | None:
    """Resolve a DOI to its OpenAlex ID (needed for the cites: filter)."""
    if not doi:
        return None
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{_OPENALEX_WORKS}/https://doi.org/{doi}",
                params={"select": "id", **_polite_params()},
            )
            if resp.status_code != 200:
                return None
            return resp.json().get("id")
    except Exception:  # noqa: BLE001
        return None


async def _openalex_cited_by(doi: str, limit: int = 10) -> list[dict]:
    """Papers that CITE the given DOI (forward snowball) via OpenAlex."""
    if not doi:
        return []
    work_id = await _openalex_work_id(doi)
    if not work_id:
        return []
    params = {
        "filter": f"cites:{work_id}",
        "per_page": min(limit, 25),
        "select": "id,title,authorships,publication_year,abstract_inverted_index,"
        "cited_by_count,doi,primary_location",
        "sort": "cited_by_count:desc",
        **_polite_params(),
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(_OPENALEX_WORKS, params=params)
            resp.raise_for_status()
            return resp.json().get("results", [])
    except Exception:  # noqa: BLE001
        return []


async def _openalex_references(doi: str, limit: int = 10) -> list[dict]:
    """Papers CITED BY the given DOI (backward snowball) via OpenAlex."""
    if not doi:
        return []
    # OpenAlex: first get the work's referenced_works ids, then batch-fetch
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{_OPENALEX_WORKS}/https://doi.org/{doi}",
                params={"select": "referenced_works", **_polite_params()},
            )
            resp.raise_for_status()
            ref_ids = (resp.json().get("referenced_works") or [])[:limit]
            if not ref_ids:
                return []
            # Batch fetch details
            filter_str = "|".join(ref_ids)
            resp2 = await client.get(
                _OPENALEX_WORKS,
                params={
                    "filter": f"openalex_id:{filter_str}",
                    "per_page": len(ref_ids),
                    "select": "id,title,authorships,publication_year,abstract_inverted_index,"
                    "cited_by_count,doi,primary_location",
                    **_polite_params(),
                },
            )
            resp2.raise_for_status()
            return resp2.json().get("results", [])
    except Exception:  # noqa: BLE001
        return []


async def _openalex_related(doi: str, limit: int = 10) -> list[dict]:
    """OpenAlex's RELATED works for this DOI (the semantic-graph direction).

    Citation chaining only finds papers connected by an explicit cite edge;
    related_works is OpenAlex's own similarity layer (n-gram + citation
    features), which surfaces topical neighbors that neither cite nor are
    cited by the anchor — exactly the recall the citation graph cannot give.
    """
    if not doi:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{_OPENALEX_WORKS}/https://doi.org/{doi}",
                params={"select": "related_works", **_polite_params()},
            )
            resp.raise_for_status()
            raw = resp.json().get("related_works") or []
            # full URLs -> bare W-ids for the batch filter
            ids = [u.rsplit("/", 1)[-1] for u in raw if u][:limit]
            if not ids:
                return []
            resp2 = await client.get(
                _OPENALEX_WORKS,
                params={
                    "filter": f"openalex_id:{'|'.join(ids)}",
                    "per_page": len(ids),
                    "select": "id,title,authorships,publication_year,abstract_inverted_index,"
                    "cited_by_count,doi,primary_location",
                    **_polite_params(),
                },
            )
            resp2.raise_for_status()
            return resp2.json().get("results", [])
    except Exception:  # noqa: BLE001
        return []


def _to_paper(work: dict, direction: str, from_doi: str) -> Paper | None:
    """Convert an OpenAlex work dict to a Paper with snowball provenance."""
    from citens.search.openalex import OpenAlexSearcher

    try:
        paper = OpenAlexSearcher.to_paper(work)
        if not paper.title or not paper.title.strip():
            return None
        # Mark provenance in the source field for auditability
        paper.source = f"snowball-{direction}({from_doi[:30]})"
        return paper
    except Exception:  # noqa: BLE001
        return None


def _s2_headers() -> dict:
    headers = {"User-Agent": "CiteLens/0.1"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    return headers


async def _s2_neighbors(doi: str, field: str, limit: int) -> list[dict]:
    """references/citations of a DOI from the S2 graph (the OpenAlex
    fallback). Shares the process-wide S2 throttle."""
    from citens.search.semantic_scholar import _throttled_get

    try:
        async with httpx.AsyncClient(timeout=30, headers=_s2_headers()) as client:
            resp = await _throttled_get(
                client,
                f"{_S2_GRAPH}/{_s2_paper_id(doi)}/{field}",
                params={"fields": _S2_FIELDS, "limit": min(limit, 100)},
            )
            if resp.status_code != 200:
                return []
            key = "citingPaper" if field == "citations" else "citedPaper"
            return [d[key] for d in resp.json().get("data", []) if d.get(key)]
    except Exception:  # noqa: BLE001
        return []


async def _s2_recommended(dois: list[str], limit: int = 10) -> list[dict]:
    """S2's learned related papers (the related_works fallback). One anchor
    uses the single-paper endpoint; the multi-paper one requires >=2 ids."""
    from citens.search.semantic_scholar import _throttled_get, _throttled_post

    try:
        async with httpx.AsyncClient(timeout=30, headers=_s2_headers()) as client:
            if len(dois) == 1:
                resp = await _throttled_get(
                    client,
                    f"{_S2_RECS}/forpaper/{_s2_paper_id(dois[0])}",
                    params={"fields": _S2_FIELDS, "limit": limit},
                )
            else:
                resp = await _throttled_post(
                    client,
                    f"{_S2_RECS}/",
                    params={"fields": _S2_FIELDS, "limit": limit},
                    json_body={
                        "positivePaperIds": [_s2_paper_id(d) for d in dois],
                        "negativePaperIds": [],
                    },
                )
            if resp.status_code != 200:
                return []
            return resp.json().get("recommendedPapers", []) or []
    except Exception:  # noqa: BLE001
        return []


def _s2_to_paper(item: dict, direction: str, from_doi: str) -> Paper | None:
    from citens.search.semantic_scholar import SemanticScholarSearcher

    try:
        paper = SemanticScholarSearcher._to_paper(item)
        if not paper.title or not paper.title.strip():
            return None
        paper.source = f"snowball-{direction}·s2({from_doi[:30]})"
        return paper
    except Exception:  # noqa: BLE001
        return None


async def _fetch_backward(doi: str, limit: int) -> list[Paper]:
    papers = [
        p for p in (
            _to_paper(w, "backward", doi)
            for w in await _openalex_references(doi, limit)
        ) if p
    ]
    if papers:
        return papers
    return [
        p for p in (
            _s2_to_paper(i, "backward", doi)
            for i in await _s2_neighbors(doi, "references", limit)
        ) if p
    ]


async def _fetch_forward(doi: str, limit: int) -> list[Paper]:
    papers = [
        p for p in (
            _to_paper(w, "forward", doi)
            for w in await _openalex_cited_by(doi, limit)
        ) if p
    ]
    if papers:
        return papers
    return [
        p for p in (
            _s2_to_paper(i, "forward", doi)
            for i in await _s2_neighbors(doi, "citations", limit)
        ) if p
    ]


async def _fetch_related(doi: str, limit: int) -> list[Paper]:
    papers = [
        p for p in (
            _to_paper(w, "related", doi)
            for w in await _openalex_related(doi, limit)
        ) if p
    ]
    if papers:
        return papers
    return [
        p for p in (
            _s2_to_paper(i, "related", doi)
            for i in await _s2_recommended([doi], limit)
        ) if p
    ]


def _relevance_score(paper: Paper, terms: set[str]) -> int:
    """Query-aware score: distinct topic terms present in title+abstract."""
    import re

    text = set(re.findall(r"[a-z]{4,}", f"{paper.title} {paper.abstract}".lower()))
    return len(terms & text)


async def snowball(
    seed_papers: Sequence[Paper],
    existing_ids: set[str],
    *,
    backward: bool = True,
    forward: bool = True,
    related: bool = True,
    limit_per_paper: int = 8,
    relevance_terms: list[str] | None = None,
    top: int = 20,
) -> list[Paper]:
    """Expand the candidate pool via citation + semantic snowballing.

    Args:
        seed_papers: High-relevance papers to snowball from
        existing_ids: Paper IDs already in the pool (skip these)
        backward: Include references (find canonical predecessors)
        forward: Include citing papers (find follow-up work)
        related: Include OpenAlex related_works (semantic neighbors off the
            citation chain — the query-aware "semantic graph" direction)
        limit_per_paper: Max papers to fetch per direction per seed
        relevance_terms: Topic query terms; when given, candidates are
            ranked by term overlap before citations (query-aware admission
            instead of pure popularity)
        top: How many ranked candidates to return (the funnel the caller
            sees). The bench passes a large value to observe the reachable
            ceiling before this cut.

    Returns:
        New papers not already in the pool, ranked (term overlap desc,
        then citations desc; citation-only order when no terms given)
    """
    if not seed_papers:
        return []

    import re as _re

    term_set = (
        {w for t in relevance_terms for w in _re.findall(r"[a-z]{4,}", t.lower())}
        if relevance_terms
        else None
    )

    cache_ns = "snowball"
    cache_key = {
        "dois": [p.doi for p in seed_papers if p.doi][:5],
        "backward": backward,
        "forward": forward,
        "related": related,
        "limit": limit_per_paper,
        "ranked": bool(term_set),
        "top": top,
    }
    cached = cache.get(cache_ns, cache_key)
    if cached is not None:
        return [
            Paper(**p) for p in cached if Paper(**p).id not in existing_ids
        ]

    seeds = [p for p in seed_papers[:5] if p.doi]  # cap seeds, bound API usage
    if not seeds:
        return []

    async def _expand(anchor: Paper) -> list[Paper]:
        doi = anchor.doi
        if not doi:  # narrowing for mypy; seeds are DOI-filtered already
            return []
        out: list[Paper] = []
        if backward:
            out += await _fetch_backward(doi, limit_per_paper)
        if forward:
            out += await _fetch_forward(doi, limit_per_paper)
        if related:
            out += await _fetch_related(doi, limit_per_paper)
        return out

    results = await asyncio.gather(*(_expand(p) for p in seeds),
                                   return_exceptions=True)

    new_papers: dict[str, Paper] = {}
    for result in results:
        if isinstance(result, BaseException):
            continue
        for p in result:
            # Basic quality gate: must have a title and some citations
            # (backward references may be canonical-but-uncited)
            if (
                p
                and p.id not in existing_ids
                and p.id not in new_papers
                and (p.citation_count >= 3
                     or p.source.startswith("snowball-backward"))
            ):
                new_papers[p.id] = p

    if term_set:
        # query-aware admission: topical overlap first, popularity as the
        # tiebreaker — a famous off-topic paper must not outrank a
        # less-cited paper that actually matches the query directions
        sorted_papers = sorted(
            new_papers.values(),
            key=lambda p: (_relevance_score(p, term_set), p.citation_count),
            reverse=True,
        )
    else:
        sorted_papers = sorted(
            new_papers.values(), key=lambda p: p.citation_count, reverse=True
        )

    # never cache an empty result: empties are usually provider failures
    # (OpenAlex budget-dead day), and caching one would poison every retry
    # for the TTL — measured: a whole bench rerun served cached zeros
    if sorted_papers:
        cache.put(
            cache_ns, cache_key, [p.model_dump() for p in sorted_papers[:top]]
        )

    return sorted_papers[:top]
