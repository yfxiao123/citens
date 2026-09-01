"""Survey-seed recall — harvest a field survey's reference list.

Bench-measured context (2026-08-31): pool golds are often niche papers the
lexical sources rank >100, and the web bridge's honest ceiling leaves ~7/15
questions unreached. A survey of the subfield is a HUMAN-CURATED canon: its
reference list contains exactly the classics-and-milestones those questions
ask about. Measured irony: citation count is an INVERTED ranking signal for
"which specific paper" questions — the most-cited pool members are the
surveys themselves, which the LLM rank correctly demotes — so surveys'
value is purely at RECALL time as seed anchors, never as ranking evidence.

Measured failure modes this module guards against (first live probe,
2026-08-31, 0/10): searching S2 with the raw QUESTION sentence returns
wrong-subfield reviews (an astronomy "Radio Survey" for a diffusion-model
question); some surveys carry no DOI. So: the search term is the PLANNER's
keyword query (not the question), a seed must share >=2 informative tokens
with the topic, and references are fetched by S2 paperId when the DOI is
missing.

Seed discovery, cheapest first:
1. scan the existing pool for review-shaped titles (zero extra calls);
2. S2 search "<keywords> survey" (free, process-wide throttled).

References come from S2's graph API. Harvested refs are ordered by citation
count so the family quota keeps the survey's own canon-head.
"""

from __future__ import annotations

import re

from citens.models import Paper

# titles that mark a survey/review; CJK covers 知乎-style 综述/教程 pages
_REVIEW_TITLE_RE = re.compile(
    r"\b(surveys?|review|tutorial|overview)\b|综述|教程",
    re.I,
)
_STOP = set(
    "a an the of for in on to with and or is are be by from as at into via "
    "using based which paper papers research study work method methods "
    "model models approach novel new".split()
)
# seed-marker words must never count as topic overlap: "survey" bridged an
# astronomy "Radio Survey" into a diffusion-model question (measured)
_MARKER_WORDS = {"survey", "surveys", "review", "tutorial", "overview"}
_S2_ID_RE = re.compile(r"^[0-9a-f]{40}$")


def _tokens(s: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
        if w not in _STOP and w not in _MARKER_WORDS and len(w) > 2
    }


def review_shaped(title: str) -> bool:
    """Title looks like a survey/review rather than a research paper."""
    return bool(_REVIEW_TITLE_RE.search(title or ""))


def pick_pool_seeds(pool: list[Paper], k: int = 2) -> list[Paper]:
    """Review-shaped papers from the pool, most-cited first.

    The pool's citation head is exactly where surveys live (measured p90
    4-11k citations vs gold's 0-78) — free seeds we already retrieved.
    """
    cand = [p for p in pool if review_shaped(p.title) and (p.doi or p.title)]
    cand.sort(key=lambda p: -(p.citation_count or 0))
    return cand[:k]


async def _s2_search_surveys(term: str, limit: int = 10) -> list[dict]:
    """S2 search "<term> survey|review", review-shaped titles, raw items."""
    from citens.search.semantic_scholar import _throttled_get
    from citens.search.snowball import _s2_headers

    import httpx

    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=30, headers=_s2_headers()) as client:
            for suffix in ("survey", "review"):
                resp = await _throttled_get(
                    client,
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": f"{term} {suffix}",
                        "fields": (
                            "paperId,title,year,citationCount,externalIds,"
                            "venue,url"
                        ),
                        "limit": limit,
                    },
                )
                if resp.status_code != 200:
                    continue
                out = [
                    it for it in resp.json().get("data", []) or []
                    if review_shaped(it.get("title") or "")
                ]
                if out:
                    break
    except Exception:  # noqa: BLE001 — optional channel
        return []
    out.sort(key=lambda it: -(it.get("citationCount") or 0))
    return out


async def survey_seed_papers(
    query: str,
    pool: list[Paper],
    *,
    planner: list[str] | None = None,
    max_surveys: int = 2,
    per_survey: int = 100,
    quota: int = 30,
) -> tuple[list[Paper], list[str]]:
    """Reference harvest from the subfield's surveys. Returns (papers, seeds).

    Seed ordering: citation count desc — the survey's canon-head first, so
    the family quota keeps the papers surveys converge on.
    """
    from citens.search.snowball import _s2_neighbors, _s2_to_paper

    terms = [t for t in (planner or []) if t and t.strip()][:2] or [query]
    topic = set().union(*(_tokens(t) for t in terms))

    def _guarded(cands: list[tuple[str, object]], want: int) -> list:
        """>=2 shared topic tokens, else DROPPED — no loose fallback. A
        wrong-subfield survey harvests ~30 off-topic refs into the pool
        (pure ranker noise), worse than abstaining. Measured leaks that
        forced this: planner phrasing junk ("What research FIRST
        proposed" -> token 'first') bridged an astronomy FIRST Radio
        Survey via a >=1 fallback. Returns surviving (title, payload)."""
        strict = [
            c for c in cands if len(topic & _tokens(c[0])) >= 2
        ]
        return strict[:want]

    # pool scan is cheapest; a raw 40-hex S2 paperId works on the graph
    # API directly, otherwise _s2_neighbors DOI-qualifies it
    seed_refs = [
        (doi, title)
        for title, doi in _guarded(
            [(s.title, s.doi) for s in pick_pool_seeds(pool, k=max_surveys * 2)
             if s.doi],
            max_surveys,
        )
    ]
    if not seed_refs:
        items = await _s2_search_surveys(terms[0])
        for title, it in _guarded(
            [(it.get("title") or "", it) for it in items], max_surveys,
        ):
            doi = (it.get("externalIds") or {}).get("DOI")
            ref = it.get("paperId") or doi
            if ref:
                seed_refs.append((ref, title))

    harvested: dict[str, Paper] = {}
    seed_titles: list[str] = []
    for ref, title in seed_refs:
        # a raw 40-hex S2 paperId passes through _s2_neighbors untouched;
        # DOIs get qualified there
        refs = await _s2_neighbors(ref, "references", per_survey)
        if not refs:
            continue
        seed_titles.append(title)
        for item in refs:
            p = _s2_to_paper(item, "survey-ref", title[:30])
            if p is None or not p.title.strip():
                continue
            key = (p.doi or p.title).lower()
            if key not in harvested:
                harvested[key] = p
    ranked = sorted(harvested.values(), key=lambda p: -(p.citation_count or 0))
    return ranked[:quota], seed_titles
