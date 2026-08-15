"""Abstract enrichment — fill missing abstracts via cross-source DOI lookup.

No single source has every abstract (e.g. Semantic Scholar may lack one that
OpenAlex or Crossref has, and vice versa). Since papers carry a DOI, we query
several sources by DOI and take the first that returns an abstract. This
directly cuts the number of "unverifiable" claims in the Verifier and raises
citation precision.

Driven by the access layer: a Springer key, if provided, adds another source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import httpx

from litreview.config import settings
from litreview.models import Paper
from litreview.search.crossref import fetch_abstract_by_doi
from litreview.search.openalex import OpenAlexSearcher


def _openalex_by_doi(doi: str) -> str:
    if not doi:
        return ""
    try:
        with httpx.Client(timeout=20, headers={"User-Agent": "CiteLens/0.1"}) as client:
            r = client.get(f"https://api.openalex.org/works/https://doi.org/{doi}")
            if r.status_code != 200:
                return ""
            return OpenAlexSearcher.decode_abstract(r.json().get("abstract_inverted_index"))
    except Exception:  # noqa: BLE001
        return ""


def _springer_by_doi(doi: str) -> str:
    if not doi or not settings.springer_api_key:
        return ""
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(
                "https://api.springernature.com/metadata/json",
                params={"q": f"doi:{doi}", "api_key": settings.springer_api_key},
            )
            if r.status_code != 200:
                return ""
            records = (r.json().get("records") or [])
            return (records[0].get("abstract") or "") if records else ""
    except Exception:  # noqa: BLE001
        return ""


def _fill_one(paper: Paper) -> tuple[str | None, str]:
    """Try sources in priority order; return (source_name, abstract)."""
    doi = (paper.doi or "").strip()
    if doi:
        ab = _openalex_by_doi(doi)
        if ab:
            return ("openalex", ab)
        ab = fetch_abstract_by_doi(doi)
        if ab:
            return ("crossref", ab)
        ab = _springer_by_doi(doi)
        if ab:
            return ("springer", ab)
    return (None, "")


def enrich_abstracts(
    papers: Sequence[Paper],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[int, list[dict]]:
    """Fill empty abstracts via cross-source DOI lookup.

    Mutates papers in place. Returns (count_filled, provenance_log).
    """
    filled = 0
    log: list[dict] = []
    for i, p in enumerate(papers):
        if p.abstract.strip():
            continue
        if on_progress:
            on_progress(i + 1, len(papers), p.title[:40])
        source, ab = _fill_one(p)
        if ab:
            p.abstract = ab
            filled += 1
            log.append({"title": p.title, "doi": p.doi, "via": source, "chars": len(ab)})
    return filled, log
