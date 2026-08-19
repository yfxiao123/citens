"""OpenAlex search source (open, free).

Uses ``filter=default.search:...`` (title + abstract) with ``relevance_score``
ordering. Title-only search starved recall on the largest metadata source:
planner queries are 3-6 words, and most relevant papers carry some of those
words in the abstract, not the title. ``relevance_score`` ordering (not the
``search=`` parameter) keeps the generic-word-swamp problem away.
"""

from __future__ import annotations

import asyncio

import httpx

from citens import cache
from citens.config import settings
from citens.models import Paper
from citens.search.base import SearchSource, register

_SOURCES_URL = "https://api.openalex.org/sources"


def resolve_source_ids(venue_names: list[str]) -> list[str]:
    """OpenAlex source IDs (W...) for venue display names, disk-cached.

    One lookup per name (first hit wins — venue whitelists use exact journal
    names, so the top display_name.search hit is the right record)."""
    ids: list[str] = []
    for name in venue_names:
        if not name:
            continue
        hit = cache.get("oasources", {"name": name})
        if hit is None:
            try:
                r = httpx.get(
                    _SOURCES_URL,
                    params={"filter": f"display_name.search:{name}", "per_page": 1},
                    timeout=20,
                )
                r.raise_for_status()
                results = r.json().get("results") or []
                hit = results[0].get("id", "") if results else ""
            except Exception:  # noqa: BLE001
                hit = ""
            cache.put("oasources", {"name": name}, hit)
        if hit:
            ids.append(str(hit))
    return ids


async def search_venue_restricted(
    queries: list[str],
    source_ids: list[str],
    year_from: int | None = None,
    per_query: int = 15,
) -> list[Paper]:
    """Title search constrained to whitelisted journal source IDs (+ year).

    This is how a "top journals only" clarification reaches the retrieval
    side: instead of filtering whatever the pool contains, go get the
    top-journal papers the pool is missing."""
    if not source_ids:
        return []
    select = (
        "id,title,authorships,publication_year,abstract_inverted_index,"
        "cited_by_count,doi,primary_location,open_access,topics,keywords,biblio"
    )
    src_filter = "|".join(source_ids[:40])

    async def _one(client: httpx.AsyncClient, q: str) -> list[Paper]:
        flt = f"title.search:{q},primary_location.source.id:{src_filter}"
        if year_from:
            flt += f",from_publication_date:{year_from}-01-01"
        try:
            resp = await client.get(
                OpenAlexSearcher.BASE_URL,
                params={"filter": flt, "per_page": min(per_query, 25),
                        "sort": "cited_by_count:desc", "select": select},
            )
            resp.raise_for_status()
            return [OpenAlexSearcher.to_paper(w) for w in resp.json().get("results", [])]
        except Exception:  # noqa: BLE001
            return []

    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *(_one(client, q) for q in queries[:8]), return_exceptions=True
        )
    out: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


def best_venue(work: dict) -> str:
    """Prefer the formal journal over a preprint host for citation purposes.

    OpenAlex makes arXiv the primary_location for works whose journal record
    is thin (or that are still preprints); when the same work's ``locations``
    list contains a real journal, a reference list should cite the journal —
    "if it has a formal publication, cite the formal publication".
    """
    candidates = [work.get("primary_location")] + (work.get("locations") or [])
    seen: set[str] = set()
    for loc in candidates:
        src = (loc or {}).get("source") or {}
        name = (src.get("display_name") or "").strip()
        if not name:
            continue
        low = name.lower()
        if any(h in low for h in ("arxiv", "ssrn", "repec", "social science research")):
            continue
        if name in seen:
            continue
        seen.add(name)
        return name
    # no formal venue recorded: keep the preprint host as the source
    src = ((work.get("primary_location") or {}).get("source") or {})
    return (src.get("display_name") or "").strip() or "OpenAlex"


@register("openalex")
class OpenAlexSearcher(SearchSource):
    name = "OpenAlex"
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self) -> None:
        self.headers = {"User-Agent": "CiteLens/0.1"}
        if settings.openalex_email:
            self.headers["User-Agent"] = f"mailto:{settings.openalex_email}"

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        per_keyword = max(max_results // max(len(keywords), 1), 5)
        async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
            tasks = [self._one(client, kw, per_keyword) for kw in keywords]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Paper] = []
        for res in results:
            if isinstance(res, list):
                out.extend(res)
        return out

    async def _one(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Paper]:
        params: dict[str, str | int] = {
            "filter": f"default.search:{query}",
            "per_page": min(limit, 50),
            "sort": "relevance_score:desc",
            "select": (
                "id,title,authorships,publication_year,abstract_inverted_index,"
                "cited_by_count,doi,primary_location,open_access,relevance_score,"
                "topics,keywords,biblio"
            ),
        }
        resp = await client.get(self.BASE_URL, params=params)
        resp.raise_for_status()
        return [self.to_paper(w) for w in resp.json().get("results", [])]

    @staticmethod
    def to_paper(work: dict) -> Paper:
        """Convert an OpenAlex work record to a Paper.

        Public because snowball.py and enrichment.py reuse the same conversion.
        """
        authors: list[str] = []
        for a in work.get("authorships", []):
            name = (a.get("author") or {}).get("display_name", "")
            if name:
                authors.append(name)
        abstract = OpenAlexSearcher.decode_abstract(work.get("abstract_inverted_index"))
        loc = work.get("primary_location") or {}
        # venue = the formal journal when one exists (arXiv primary_location
        # loses to the published version); PDF url still prefers OA locations
        source_name = best_venue(work)
        pdf_url = (loc.get("pdf_url") or "").strip() or None
        oa = work.get("open_access") or {}
        if not pdf_url and oa.get("oa_url"):
            pdf_url = oa["oa_url"].strip()
        # Subfield: OpenAlex's own taxonomy beats guessing (topics carry the
        # indexer-assigned field/subfield of this work).
        subfield = ""
        for t in work.get("topics") or []:
            sf = (t.get("subfield") or {}).get("display_name", "")
            if sf:
                subfield = sf
                break
        keywords = [
            kw.get("display_name", "")
            for kw in work.get("keywords") or []
            if kw.get("display_name")
        ][:12]
        biblio = work.get("biblio") or {}
        fp, lp = str(biblio.get("first_page") or ""), str(biblio.get("last_page") or "")
        return Paper(
            title=work.get("title", ""),
            authors=authors,
            year=work.get("publication_year"),
            abstract=abstract,
            source=f"OpenAlex ({source_name})",
            citation_count=work.get("cited_by_count", 0) or 0,
            url=work.get("id", ""),
            doi=work.get("doi"),
            pdf_url=pdf_url,
            venue=source_name if source_name != "OpenAlex" else "",
            keywords=keywords,
            subfield=subfield,
            volume=str(biblio.get("volume") or ""),
            issue=str(biblio.get("issue") or ""),
            pages=f"{fp}-{lp}".strip("-") if (fp or lp) else "",
        )

    @staticmethod
    def decode_abstract(inverted_index: dict | None) -> str:
        if not inverted_index:
            return ""
        words: dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        return " ".join(words.get(i, "") for i in range(len(words)))
