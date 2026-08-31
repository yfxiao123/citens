"""Web search source — the vocabulary bridge lexical corpora cannot offer.

LitSearch-measured failure: the question speaks the CONTRIBUTION story
("world model planning to ensure safe decisions") while the gold title
speaks SYSTEM names ("SafeDreamer") — every lexical scholarly source ranks
the gold >100 for our best paraphrased queries (depth-100 autopsy, 2026-08).
Blog/forum/tutorial text is written in exactly the question's story
language while LINKING the paper — Zhihu/CSDN/LessWrong/OpenReview
discussions are a human-paraphrase pivot corpus with exact-paper anchors.

Channel (all free, no API keys):
1. DuckDuckGo HTML endpoint, one fetch per query (English + Chinese
   variants; kl=cn-zh boosts Zhihu/CSDN for the Chinese ones).
2. Anchor extraction from result URLs (arXiv/aclanthology/DOI links are
   exact papers) and from title-like strings (re-searched on arXiv/
   Crossref, similarity-gated).
3. Anything anchored joins the pool as a Paper with source="web".

Failure posture: DDG throttles or the markup changes → return [] (the
channel is optional, never fatal).
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from citens.config import settings
from citens.models import Paper
from citens.search.base import SearchSource, register

_DDG_URL = "https://html.duckduckgo.com/html/"
_BING_URL = "https://www.bing.com/search"
_SERPER_URL = "https://google.serper.dev/search"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_ARXIV_HREF_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)|alphaxiv\.org/abs)/([0-9]{4}\.[0-9]{4,5})",
    re.I,
)
# aclanthology.org/<venue-id>(.pdf|/) → DOI 10.18653/v1/<venue-id>, no request
_ACL_HREF_RE = re.compile(r"aclanthology\.org/([A-Z0-9.\-]+?)(?:\.[a-z]+)?(?:[/?#]|$)", re.I)
# DOI anywhere it appears in a result URL: doi.org/..., publisher /doi/ or
# /article/ paths (dl.acm.org, journals.sagepub.com, link.springer.com, ...)
_DOI_HREF_RE = re.compile(
    r"(?:doi\.org/|/doi/(?:abs/|full/)?|/article/)(10\.[0-9]{4,9}/[^?&#\s]+)",
    re.I,
)
# semanticscholar.org/paper/<slug>/<40-hex paperId>
_S2_HREF_RE = re.compile(r"semanticscholar\.org/paper/[^/]+/([0-9a-f]{40})", re.I)
# redirect-wrapped result hrefs: //duckduckgo.com/l/?uddg=<urlencoded>
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")

QUERY_SYSTEM = """Write web-search queries to find WHICH PAPER a literature \
question is about. Blogs and forums describe papers by their contribution \
story, not their title, so search the story.

Rules:
1. Reply ONLY JSON: {"en": ["...", ...], "zh": ["...", ...]} — 3 English and \
2 Chinese queries.
2. FIRST English query: quote VERBATIM the question's most DISTINCTIVE span \
— 4-8 words carrying its unusual technical wording, the part a generic paper \
on the topic would NOT share ("re-attempting the abstained instances", \
"unifying diffusion process across resolutions"). NOT the generic topic \
words. Questions paraphrase abstracts, so the question's own phrasing often \
matches the paper's title wording on blogs and paper pages.
3. Second English query: a DIFFERENT distinctive verbatim span from another \
part of the question.
4. Third English query: broader concept keywords (3-8 words) describing the \
contribution story.
5. The Chinese ones must be natural Chinese CS-community phrasing (the kind \
used in Zhihu/CSDN tutorials), not translations of the English ones.
6. Plain keyword phrases only — no operators, no quotes, no site:."""


def _decode_href(href: str) -> str:
    """DDG wraps results in /l/?uddg=<encoded>; unwrap to the real URL."""
    from urllib.parse import unquote, parse_qs, urlparse

    if "uddg=" in href:
        try:
            qs = parse_qs(urlparse(href).query)
            if qs.get("uddg"):
                return unquote(qs["uddg"][0])
        except Exception:  # noqa: BLE001
            pass
    return href


async def _serper_results(
    client: httpx.AsyncClient, kw: str, limit: int
) -> list[dict] | None:
    """Google SERP via serper.dev (keyed). None when unset/failed so the
    caller falls through to keyless engines."""
    from citens.config import settings

    key = settings.serper_api_key
    if not key:
        return None
    body: dict = {"q": kw, "num": min(limit, 10)}
    if _has_cjk(kw):
        body["hl"] = "zh-cn"
    try:
        r = await client.post(
            _SERPER_URL,
            content=json.dumps(body),
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
        )
        if r.status_code != 200:
            return None
        return [
            {
                "title": it.get("title") or "",
                "snippet": it.get("snippet") or "",
                "url": it.get("link") or "",
            }
            for it in (r.json().get("organic") or [])
            if it.get("link")
        ][:limit]
    except Exception:  # noqa: BLE001 — fall through to keyless engines
        return None


def parse_bing_results(html: str, limit: int = 12) -> list[dict]:
    """(title, snippet, url) triples from Bing's organic results (li.b_algo).

    Primary engine: Bing tolerates moderate keyless scraping, indexes
    Zhihu/CSDN well, and serves full results where DDG's HTML endpoint
    answers 202 + anomaly-challenge after a few dozen fetches (measured
    2026-08-30). Markup is stable legacy server-side HTML.
    """
    out: list[dict] = []
    blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.S)
    for block in blocks:
        m = re.search(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        sm = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        snippet = re.sub(r"<[^>]+>", "", sm.group(1)).strip() if sm else ""
        if title and url.startswith("http"):
            out.append({"title": title, "snippet": snippet, "url": url})
        if len(out) >= limit:
            break
    return out


def parse_ddg_results(html: str, limit: int = 12) -> list[dict]:
    """(title, snippet, url) triples from the DDG HTML endpoint.

    Two-pass (anchors then snippets, zipped in document order) — a single
    mega-regex across the extras block proved brittle. Regex-based on
    purpose: the endpoint is stable legacy markup and the source must
    degrade to [] on any layout change, never crash.
    """
    anchors = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S
    )
    snippets = [
        re.sub(r"<[^>]+>", "", s).strip()
        for s in re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.S)
    ]
    out: list[dict] = []
    for k, (href, title) in enumerate(anchors):
        title = re.sub(r"<[^>]+>", "", title).strip()
        url = _decode_href(href)
        if title and url:
            out.append({
                "title": title,
                "snippet": snippets[k] if k < len(snippets) else "",
                "url": url,
            })
        if len(out) >= limit:
            break
    return out


_SITE_SUFFIXES = (
    "知乎", "CSDN博客", "CSDN", "csdn.net", "博客园", "简书", "掘金",
    "GitHub", "Stack Overflow", "Medium", "arXiv", "YouTube", "Bilibili",
    "哔哩哔哩", "博客", "论文", "AI", "深度学习", "机器学习",
)


def _clean_title_candidate(raw: str) -> str:
    """'论文标题 - 知乎' -> the paper-ish part. Suffix whitelist only: a
    generic split-on-dash/colon would truncate titles like 'MISC: ...'
    (measured) — paper titles carry colons far more often than site names
    carry informative tails."""
    t = raw.strip()
    for sep in (" - ", " | ", " – ", " — ", "｜"):
        if sep in t:
            head, tail = t.rsplit(sep, 1)
            if tail.strip() in _SITE_SUFFIXES:
                t = head.strip()
                break
    t = re.sub(r"\s+", " ", t)
    return t[:150]


async def _anchor_papers(results: list[dict], client: httpx.AsyncClient) -> list[Paper]:
    """Resolve web hits to exact scholarly records.

    URL anchors (arXiv/ACL/DOI links) are exact — resolve directly. Title
    strings are candidates — searched on arXiv and Crossref and kept only
    on a strong title match. All free endpoints.
    """
    from citens.search.openalex import OpenAlexSearcher

    papers: list[Paper] = []
    seen: set[str] = set()

    async def _add(paper: Paper | None, provenance: str) -> None:
        if paper is None or not paper.title.strip():
            return
        k = ("d:" + (paper.doi or "")) if paper.doi else ("t:" + paper.title.lower())
        if k in seen:
            return
        seen.add(k)
        paper.source = provenance
        papers.append(paper)

    async def _resolve_doi(doi: str, provenance: str) -> None:
        try:
            rr = await client.get(
                f"https://api.crossref.org/works/{doi}", timeout=15
            )
            msg = rr.json().get("message") or {}
            title = (msg.get("title") or [""])[0]
            if title:
                await _add(Paper(
                    title=title, authors=[],
                    doi=doi.split("?")[0],
                    citation_count=msg.get("is-referenced-by-count") or 0,
                ), provenance)
        except Exception:  # noqa: BLE001
            pass

    async def _arxiv_lookup(arxiv_id: str, provenance: str) -> None:
        # export.arxiv.org drops connections intermittently (measured
        # 2026-08-31: plain ConnectError mid-batch) — one retry recovers it.
        for attempt in range(2):
            try:
                rr = await client.get(
                    "https://export.arxiv.org/api/query",
                    params={"id_list": arxiv_id, "max_results": 1}, timeout=15,
                )
                entries = re.findall(r"<entry>(.*?)</entry>", rr.text, re.S)
                if entries:
                    t = re.search(r"<title>(.*?)</title>", entries[0], re.S)
                    if t:
                        await _add(Paper(
                            title=re.sub(r"\s+", " ", t.group(1)).strip(),
                            authors=[], doi=f"10.48550/arxiv.{arxiv_id}",
                            url=f"https://arxiv.org/abs/{arxiv_id}",
                        ), provenance)
                return
            except Exception:  # noqa: BLE001
                if attempt:
                    return
                await asyncio.sleep(1.5)

    # 1. URL anchors — exact
    for r in results:
        url = r["url"]
        m = _ARXIV_HREF_RE.search(url)
        if m:
            await _arxiv_lookup(m.group(1), "web-arxiv-link")
            continue
        m = _ACL_HREF_RE.search(url)
        if m and re.search(r"\d", m.group(1)):
            # anthology venue-id → ACL DOI directly, no page fetch
            await _resolve_doi(f"10.18653/v1/{m.group(1)}", "web-acl-link")
            continue
        m = _DOI_HREF_RE.search(url)
        if m:
            from urllib.parse import unquote

            await _resolve_doi(unquote(m.group(1)), "web-doi-link")
            continue
        m = _S2_HREF_RE.search(url)
        if m:
            try:
                from citens.search.semantic_scholar import _throttled_get

                headers = {}
                if settings.semantic_scholar_api_key:
                    headers["x-api-key"] = settings.semantic_scholar_api_key
                rr = await _throttled_get(
                    client,
                    f"https://api.semanticscholar.org/graph/v1/paper/{m.group(1)}",
                    params={"fields": "title,externalIds,citationCount"},
                    headers=headers or None,
                )
                if rr.status_code == 200:
                    data = rr.json()
                    ext = data.get("externalIds") or {}
                    arxiv_id = ext.get("ArXiv")
                    await _add(Paper(
                        title=data.get("title") or "",
                        authors=[],
                        doi=ext.get("DOI")
                        or (f"10.48550/arxiv.{arxiv_id}" if arxiv_id else None),
                        citation_count=data.get("citationCount") or 0,
                        url=(f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url),
                    ), "web-s2-link")
            except Exception:  # noqa: BLE001
                pass

    # 2. Title-string anchors — candidates, similarity-gated.
    # Crossref covers journals/proceedings; arXiv ti-search rescues the ML
    # preprint majority Crossref does not index (measured: SafeDreamer's
    # SERP subtitle hits its arXiv title at sim 0.88 while Crossref returns
    # nothing). arXiv calls are capped per batch and spaced — API etiquette.
    arxiv_calls = 0
    for r in results:
        cand = _clean_title_candidate(r["title"])
        if len(cand.split()) < 5 and len(cand) < 20:
            continue
        paper = await _title_anchor_crossref(client, cand)
        if paper is None and arxiv_calls < 3:
            if arxiv_calls:
                await asyncio.sleep(1.0)
            arxiv_calls += 1
            paper = await _title_anchor_arxiv(client, cand)
        if paper is not None:
            await _add(paper, "web-title-anchor")
    return papers


async def _title_anchor_crossref(
    client: httpx.AsyncClient, cand: str
) -> Paper | None:
    try:
        rr = await client.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": cand, "rows": 3},
            timeout=15,
        )
        items = (rr.json().get("message") or {}).get("items") or []
        best, best_sim = None, 0.0
        for it in items:
            t = (it.get("title") or [""])[0]
            sim = _title_sim(cand, t)
            if sim > best_sim:
                best, best_sim = (t, it), sim
        if best and best_sim >= 0.75:
            t, it = best
            return Paper(
                title=t, authors=[],
                doi=(it.get("DOI") or "") or None,
                citation_count=it.get("is-referenced-by-count") or 0,
            )
    except Exception:  # noqa: BLE001
        return None
    return None


async def _title_anchor_arxiv(
    client: httpx.AsyncClient, cand: str
) -> Paper | None:
    for attempt in range(2):  # export.arxiv.org drops connections at times
        try:
            rr = await client.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": f'ti:"{cand}"', "max_results": 3},
                timeout=15,
            )
            entries = re.findall(r"<entry>(.*?)</entry>", rr.text, re.S)
            best_title, best_id, best_sim = None, None, 0.0
            for e in entries:
                t = re.search(r"<title>(.*?)</title>", e, re.S)
                iid = re.search(r"/abs/([^<]+)</id>", e)
                if not t or not iid:
                    continue
                title = re.sub(r"\s+", " ", t.group(1)).strip()
                sim = _title_sim(cand, title)
                if sim > best_sim:
                    best_title, best_id, best_sim = title, iid.group(1), sim
            if best_title and best_sim >= 0.75:
                return Paper(
                    title=best_title, authors=[],
                    doi=f"10.48550/arxiv.{best_id}",
                    url=f"https://arxiv.org/abs/{best_id}",
                )
            return None
        except Exception:  # noqa: BLE001
            if attempt:
                return None
            await asyncio.sleep(1.5)
    return None


def _title_sim(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()  # noqa: E731
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def generate_web_queries(question: str) -> dict[str, list[str]]:
    """Bilingual concept queries (EN + Zhihu-style Chinese). [] on failure."""
    from citens.llm import chat_json

    try:
        result = chat_json(
            # Strong model, not cheap: span selection ("which 4-8 words of the
            # question are unusual enough to match the paper's own wording?")
            # is exactly where the cheap model flops — measured on seed-42
            # misses it kept picking the generic topic span (2026-08-31).
            QUERY_SYSTEM, f"Literature question: {question}",
            max_tokens=400, thinking=False,
        )
        def pick(v, cap: int):
            return [
                str(q).strip()[:120] for q in (v or [])
                if isinstance(q, str) and 2 <= len(q.split()) <= 12
            ][:cap]

        return {"en": pick(result.get("en"), 3), "zh": pick(result.get("zh"), 2)}
    except Exception:  # noqa: BLE001 — optional channel
        return {"en": [], "zh": []}


def generate_web_query_list(question: str) -> list[str]:
    """Flat EN+ZH query list for the cached-generation path."""
    wq = generate_web_queries(question)
    return (wq.get("en") or []) + (wq.get("zh") or [])


@register("web")
class WebSearchSource(SearchSource):
    name = "Web (blog/forum bridge)"
    BASE_URL = _DDG_URL

    async def search(self, keywords: list[str], max_results: int) -> list[Paper]:
        # depth ≥8 per query: the SERP call count is fixed per query, so
        # deeper pages are free — they catch golds ranked 6-8 under siblings
        per_query = max(max_results // max(len(keywords), 1), 8)
        out: list[Paper] = []
        async with httpx.AsyncClient(
            timeout=20, headers={"User-Agent": _UA}, follow_redirects=True
        ) as client:
            for kw in keywords:
                try:
                    results = await self._fetch_results(client, kw, per_query)
                    found = await _anchor_papers(results, client)
                    for p in found:
                        p.matched_queries.append(kw)
                    out.extend(found)
                    self.query_stats[kw] = len(found)
                except Exception:  # noqa: BLE001 — optional channel
                    self.query_stats[kw] = 0
                await asyncio.sleep(2.5)  # engine politeness
        return out

    async def _fetch_results(
        self, client: httpx.AsyncClient, kw: str, limit: int
    ) -> list[dict]:
        """Serper (keyed, Google SERP) primary; keyless scraping as fallback
        (Bing serves degraded SERPs to keyless clients and DDG 202-challenges
        after bursts — both measured 2026-08-30)."""
        serper = await _serper_results(client, kw, limit)
        if serper:
            return serper
        r = await client.get(_BING_URL, params={"q": kw})
        if r.status_code == 200 and 'class="b_algo"' in r.text:
            results = parse_bing_results(r.text, limit=limit)
            if results:
                return results
        r = await client.get(
            _DDG_URL,
            params={"q": kw, "kl": "cn-zh" if _has_cjk(kw) else "wt-wt"},
        )
        if r.status_code == 200:
            return parse_ddg_results(r.text, limit=limit)
        return []


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)
