"""OpenAlex search source (open, free).

Uses ``filter=default.search:...`` (title + abstract) with ``relevance_score``
ordering. Title-only search starved recall on the largest metadata source:
planner queries are 3-6 words, and most relevant papers carry some of those
words in the abstract, not the title. ``relevance_score`` ordering (not the
``search=`` parameter) keeps the generic-word-swamp problem away.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
                note_openalex_calls()
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
                params=apply_api_key({
                    "filter": flt, "per_page": min(per_query, 25),
                    "sort": "cited_by_count:desc", "select": select,
                }),
            )
            note_openalex_calls()
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


def apply_api_key(params: dict) -> dict:
    """Attach the premium api_key when configured (100k/day, higher rps);
    without it OpenAlex is the 1000-req/day anonymous pool that dies by
    mid-morning on any bench-heavy day."""
    if settings.openalex_api_key:
        params = {**params, "api_key": settings.openalex_api_key}
    return params


# --- daily credit budget -----------------------------------------------------
# OpenAlex's keyed API is CREDIT-based: the free plan includes $1/day =
# 10000 credits, midnight-UTC refill. MEASURED LIVE (2026-08-30, x-ratelimit
# remaining deltas): EVERY works query costs 10 credits — search= AND
# filter= alike (the dashboard's "filtered list: 1 credit" does NOT apply
# to works queries); a single-record view (works/doi:...) is FREE.
# → the free plan buys ~1000 works queries/day; a full bench leg costs
# ~$0.28. The counter below is persisted to disk so repeated bench runs
# share ONE daily budget. When exhausted, OPTIONAL channels (oa_title,
# snowball) skip outright; the core search only warns.

_BUDGET_WARNED: dict[str, bool] = {}

QUERY_CREDITS = 10  # any works query (search= or filter=) — measured
VIEW_CREDITS = 0  # single-record view works/doi:... — free
FILTER_CREDITS = QUERY_CREDITS  # kept for call sites; filters are queries too


def _usage_path() -> Path:
    return Path(settings.cache_dir) / "openalex_usage.json"


def _today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def note_openalex_calls(credits: int = FILTER_CREDITS) -> None:
    """Record ``credits`` spent against today's persisted budget."""
    data: dict = {"date": _today(), "used": 0}
    try:
        path = _usage_path()
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("date") == _today():
                data["used"] = int(stored.get("used", 0))
        data["used"] += credits
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:  # noqa: BLE001 — telemetry must never break retrieval
        return
    budget = settings.openalex_daily_budget
    if budget <= 0:
        return
    used = data["used"]
    for mark, msg in (
        (0.8, f"[openalex] 80% of daily credit budget used ({used}/{budget})"),
        (1.0, f"[openalex] DAILY CREDIT BUDGET REACHED ({used}/{budget}) — "
              "optional channels (oa-title, snowball) will skip"),
    ):
        key = str(mark)
        if used >= budget * mark and not _BUDGET_WARNED.get(key):
            _BUDGET_WARNED[key] = True
            print(msg, flush=True)


def budget_exceeded() -> bool:
    """True when today's persisted usage has reached the configured budget."""
    if settings.openalex_daily_budget <= 0:
        return False
    try:
        stored = json.loads(_usage_path().read_text(encoding="utf-8"))
        if stored.get("date") != _today():
            return False
        return int(stored.get("used", 0)) >= settings.openalex_daily_budget
    except Exception:  # noqa: BLE001 — unreadable counter = no guard
        return False


def usage_today() -> tuple[int, int]:
    """(credits used today, configured credit budget). Budget 0 = unlimited."""
    try:
        stored = json.loads(_usage_path().read_text(encoding="utf-8"))
        used = int(stored.get("used", 0)) if stored.get("date") == _today() else 0
    except Exception:  # noqa: BLE001
        used = 0
    return used, settings.openalex_daily_budget


@register("openalex")
class OpenAlexSearcher(SearchSource):
    name = "OpenAlex"
    BASE_URL = "https://api.openalex.org/works"

    def __init__(self) -> None:
        super().__init__()
        self.headers = {"User-Agent": "CiteLens/0.1"}
        if settings.openalex_email:
            self.headers["User-Agent"] = f"mailto:{settings.openalex_email}"

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        per_keyword = max(max_results // max(len(keywords), 1), 5)
        from citens.search.base import SEARCH_CONCURRENCY

        sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

        async def _guarded(q: str) -> list[Paper]:
            async with sem:
                return await self._one(client, q, per_keyword)

        async with httpx.AsyncClient(timeout=30, headers=self.headers) as client:
            tasks = [_guarded(kw) for kw in keywords]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[Paper] = []
        for res in results:
            if isinstance(res, list):
                out.extend(res)
        return out

    async def _one(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Paper]:
        # year window from the clarification answers rides OpenAlex's NATIVE
        # filters (text hacks like "2023..2026" inside the query only work by
        # accident on some backends and pollute relevance ranking)
        flt = f"default.search:{query}"
        if self.constraints and self.constraints.year_from:
            flt += f",from_publication_date:{self.constraints.year_from}-01-01"
        if self.constraints and self.constraints.year_to:
            flt += f",to_publication_date:{self.constraints.year_to}-12-31"
        params: dict[str, str | int] = {
            "filter": flt,
            "per_page": min(limit, 50),
            "sort": "relevance_score:desc",
            "select": (
                "id,title,authorships,publication_year,abstract_inverted_index,"
                "cited_by_count,doi,primary_location,open_access,relevance_score,"
                "topics,keywords,biblio"
            ),
        }
        resp = await client.get(self.BASE_URL, params=apply_api_key(params))
        note_openalex_calls()
        resp.raise_for_status()
        papers = [self.to_paper(w) for w in resp.json().get("results", [])]
        for p in papers:
            p.matched_queries.append(query)  # retrieval provenance
        self.query_stats[query] = len(papers)
        return papers

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
