"""Abstract enrichment — fill missing abstracts via cross-source DOI lookup.

No single source has every abstract (e.g. Semantic Scholar may lack one that
OpenAlex or Crossref has, and vice versa). Since papers carry a DOI, we query
several sources by DOI and take the first that returns an abstract. This
directly cuts the number of "unverifiable" claims in the Verifier and raises
citation precision.

Root cause this addresses (2026-08-18 run: 7/20 core papers abstract-less,
enrichment filled 0/7): Elsevier journals deposit no abstract to OpenAlex
(``abstract_inverted_index`` null) or Crossref, and SSRN preprint DOIs carry
none in Crossref — while Semantic Scholar, which crawls publisher and preprint
landing pages itself, often has them. S2 is therefore tried by DOI whenever a
key is configured (its authenticated tier allows 1 request/second shared).

Driven by the access layer: a Springer key, if provided, adds another source.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import httpx

from citens.config import settings
from citens.models import Paper
from citens.search.crossref import fetch_abstract_by_doi
from citens.search.openalex import OpenAlexSearcher

# S2 authenticated tier: 1 rps shared across ALL endpoints. Enrichment runs
# after the search stage (no overlap with the async search throttle), so a
# module-level sync spacing is enough — spaced starts + one 429 retry.
_S2_MIN_INTERVAL = 1.2
_s2_last = 0.0


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


def _s2_get(doi: str) -> tuple[int, str]:
    """One S2 DOI attempt. Returns (status_code, abstract)."""
    try:
        with httpx.Client(
            timeout=20, headers={"x-api-key": settings.semantic_scholar_api_key}
        ) as client:
            r = client.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "abstract"},
            )
        if r.status_code == 200:
            return 200, (r.json().get("abstract") or "").strip()
        return r.status_code, ""
    except Exception:  # noqa: BLE001
        return -1, ""


def _s2_by_doi(doi: str) -> str:
    global _s2_last
    if not doi or not settings.semantic_scholar_api_key:
        return ""
    for _attempt in range(2):
        wait = _S2_MIN_INTERVAL - (time.monotonic() - _s2_last)
        if wait > 0:
            time.sleep(wait)
        _s2_last = time.monotonic()
        status, abstract = _s2_get(doi)
        if status == 200:
            return abstract
        if status != 429:  # 404 etc. — retrying won't help
            return ""
        time.sleep(2.0)
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
        ab = _s2_by_doi(doi)
        if ab:
            return ("semantic_scholar", ab)
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
