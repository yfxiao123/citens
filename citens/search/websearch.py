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
_ARXIV_HREF_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5})", re.I)
_ACL_HREF_RE = re.compile(r"aclanthology\.org/([A-Z0-9.\-]+)/", re.I)
_DOI_HREF_RE = re.compile(r"doi\.org/(10\.[0-9]{4,9}/\S+)", re.I)
# redirect-wrapped result hrefs: //duckduckgo.com/l/?uddg=<urlencoded>
_UDDG_RE = re.compile(r"[?&]uddg=([^&]+)")

QUERY_SYSTEM = """Write web-search queries to find WHICH PAPER a literature \
question is about. Blogs and forums describe papers by their contribution \
story, not their title, so search the story.

Rules:
1. Reply ONLY JSON: {"en": ["...", ...], "zh": ["...", ...]} — 2 English and \
2 Chinese queries.
2. Queries are plain keyword phrases (3-8 words), no operators, no site:.
3. The Chinese ones must be natural Chinese CS-community phrasing (the kind \
used in Zhihu/CSDN tutorials), not translations of the English ones.
4. Cover different aspects of the question across the queries."""


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

    # 1. URL anchors — exact
    for r in results:
        url = r["url"]
        m = _ARXIV_HREF_RE.search(url)
        if m:
            try:
                rr = await client.get(
                    f"https://export.arxiv.org/api/query",
                    params={"id_list": m.group(1), "max_results": 1},
                )
                entries = re.findall(r"<entry>(.*?)</entry>", rr.text, re.S)
                if entries:
                    t = re.search(r"<title>(.*?)</title>", entries[0], re.S)
                    if t:
                        await _add(Paper(
                            title=re.sub(r"\s+", " ", t.group(1)).strip(),
                            authors=[], doi=f"10.48550/arxiv.{m.group(1)}",
                            url=f"https://arxiv.org/abs/{m.group(1)}",
                        ), "web-arxiv-link")
            except Exception:  # noqa: BLE001
                pass
            continue
        m = _DOI_HREF_RE.search(url)
        if m:
            try:
                rr = await client.get(
                    f"https://api.crossref.org/works/{m.group(1)}", timeout=15
                )
                msg = rr.json().get("message") or {}
                title = (msg.get("title") or [""])[0]
                if title:
                    await _add(Paper(
                        title=title, authors=[],
                        doi=m.group(1).split("?")[0],
                        citation_count=msg.get("is-referenced-by-count") or 0,
                    ), "web-doi-link")
            except Exception:  # noqa: BLE001
                pass

    # 2. Title-string anchors — candidates, similarity-gated
    for r in results:
        cand = _clean_title_candidate(r["title"])
        if len(cand.split()) < 5 and len(cand) < 20:
            continue
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
                await _add(Paper(
                    title=t, authors=[],
                    doi=(it.get("DOI") or "") or None,
                    citation_count=it.get("is-referenced-by-count") or 0,
                ), "web-title-anchor")
        except Exception:  # noqa: BLE001
            continue
    return papers


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
            QUERY_SYSTEM, f"Literature question: {question}",
            max_tokens=300, thinking=False, cheap=True,
        )
        pick = lambda v: [  # noqa: E731
            str(q).strip()[:120] for q in (v or [])
            if isinstance(q, str) and 2 <= len(q.split()) <= 12
        ][:2]
        return {"en": pick(result.get("en")), "zh": pick(result.get("zh"))}
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
        per_query = max(max_results // max(len(keywords), 1), 5)
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
