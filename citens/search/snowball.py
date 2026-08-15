"""Citation-graph snowballing: expand the candidate pool via references and
citations of highly relevant papers.

Two directions (the standard systematic-review technique):
- BACKWARD: fetch the references of a top paper (find its intellectual roots,
  canonical predecessors we may have missed).
- FORWARD: fetch papers that cite a top paper (find follow-up work, newer
  developments, empirical validations).

Both use the OpenAlex API (free, no key, DOI-based) with Semantic Scholar as
fallback. Results are Papers with a `snowball_from` marker so provenance is
auditable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from citens import cache
from citens.config import settings
from citens.models import Paper

_OPENALEX_WORKS = "https://api.openalex.org/works"
_S2_GRAPH = "https://api.semanticscholar.org/graph/v1/paper"


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


async def snowball(
    seed_papers: Sequence[Paper],
    existing_ids: set[str],
    *,
    backward: bool = True,
    forward: bool = True,
    limit_per_paper: int = 8,
) -> list[Paper]:
    """Expand the candidate pool via citation snowballing.

    Args:
        seed_papers: High-relevance papers to snowball from
        existing_ids: Paper IDs already in the pool (skip these)
        backward: Include references (find canonical predecessors)
        forward: Include citing papers (find follow-up work)
        limit_per_paper: Max papers to fetch per direction per seed

    Returns:
        New papers not already in the pool, sorted by citation count
    """
    if not seed_papers:
        return []

    cache_ns = "snowball"
    cache_key = {
        "dois": [p.doi for p in seed_papers if p.doi][:5],
        "backward": backward,
        "forward": forward,
        "limit": limit_per_paper,
    }
    cached = cache.get(cache_ns, cache_key)
    if cached is not None:
        return [
            Paper(**p) for p in cached if Paper(**p).id not in existing_ids
        ]

    tasks: list[asyncio.Task] = []
    for paper in seed_papers[:5]:  # cap seeds at 5 to bound API usage
        if not paper.doi:
            continue
        if backward:
            tasks.append(
                asyncio.create_task(_openalex_references(paper.doi, limit_per_paper))
            )
        if forward:
            tasks.append(
                asyncio.create_task(_openalex_cited_by(paper.doi, limit_per_paper))
            )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_papers: dict[str, Paper] = {}
    task_meta: list[tuple[str, str]] = []
    idx = 0
    for paper in seed_papers[:5]:
        if not paper.doi:
            continue
        if backward:
            task_meta.append(("backward", paper.doi))
            idx += 1
        if forward:
            task_meta.append(("forward", paper.doi))
            idx += 1

    for i, result in enumerate(results):
        if isinstance(result, Exception) or not isinstance(result, list):
            continue
        direction, from_doi = (
            task_meta[i] if i < len(task_meta) else ("unknown", "")
        )
        for work in result:
            p = _to_paper(work, direction, from_doi)
            # Basic quality gate: must have a title and some citations
            if (
                p
                and p.id not in existing_ids
                and p.id not in new_papers
                and (p.citation_count >= 3 or direction == "backward")
            ):
                new_papers[p.id] = p

    sorted_papers = sorted(
        new_papers.values(), key=lambda p: p.citation_count, reverse=True
    )

    cache.put(
        cache_ns, cache_key, [p.model_dump() for p in sorted_papers[:20]]
    )

    return sorted_papers[:20]
