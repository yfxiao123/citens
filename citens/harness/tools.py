"""Tool surface for the agentic retrieval loop.

Each tool is a thin adapter over existing, battle-tested functions — the
loop adds judgment, not new retrieval machinery. Handlers mutate a shared
:class:`HarnessState` and return a compact string result for the model
(tool results are the model's only feedback channel, so they carry the
signal: per-query hits, zero-hit warnings, untried synonyms, pool deltas).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from citens.agents.planner import QueryPlan
from citens.models import Paper
from citens.search.base import deduplicate, search_round
from citens.search.snowball import snowball

_MAX_QUERIES_PER_CALL = 4


@dataclass
class HarnessState:
    """Everything the loop and its tools share."""

    topic: str
    plan: QueryPlan
    pool: list[Paper]
    keywords: list[str]
    query_stats: dict[str, int]
    constraints: Any = None  # RetrievalConstraints | None
    sources: list[str] | None = None
    max_results: int = 60
    facets: list[dict] = field(default_factory=list)
    search_calls: int = 0
    snowball_calls: int = 0
    # budget caps the tools enforce themselves (the loop also tracks steps)
    max_search_calls: int = 5
    # anti-loop: exact query-sets already executed (lowercased frozenset)
    searched_keys: set[frozenset] = field(default_factory=set)
    # goal shape: "survey" (default — build a balanced pool) or "find"
    # (targeted retrieval: the question names specific paper(s))
    goal: str = "survey"
    # done-gate state: anchors must run once before the first done is honored
    anchors_checked: bool = False
    done_refused: bool = False

    def note_queries(self, queries: list[str]) -> None:
        for q in queries:
            if q not in self.keywords:
                self.keywords.append(q)

    def merge(self, new: list[Paper]) -> int:
        """Merge into the pool, return how many papers were actually new."""
        before = len(self.pool)
        self.pool = deduplicate(self.pool + new)
        return len(self.pool) - before


def _fmt_stats(state: HarnessState, queries: list[str]) -> str:
    parts = [f"{q} {state.query_stats.get(q, 0)}" for q in queries]
    zero = [q for q in queries if state.query_stats.get(q, 0) == 0]
    lines = ["per-query hits: " + " · ".join(parts)]
    if zero:
        syn_hints: list[str] = []
        for q in zero:
            for syn in state.plan.synonyms_for(q)[:2]:
                if syn.lower() not in {k.lower() for k in state.keywords}:
                    syn_hints.append(syn)
        lines.append(
            f"ZERO-HIT queries: {', '.join(zero)} — the field does not use this "
            f"phrasing. Untried synonyms: {', '.join(syn_hints) or '(none left)'}"
        )
    return "\n".join(lines)


async def tool_search(state: HarnessState, queries: list[str], rationale: str = "") -> str:
    if state.search_calls >= state.max_search_calls:
        return (
            "SEARCH BUDGET EXHAUSTED — further search calls are refused. "
            "Use pool_report / read_paper to consolidate, or call done."
        )
    queries = [str(q).strip() for q in queries if str(q).strip()][:_MAX_QUERIES_PER_CALL]
    if not queries:
        return "ERROR: empty query list."
    key = frozenset(q.lower() for q in queries)
    if key in state.searched_keys:
        return (
            "DUPLICATE search — this exact query set was already executed "
            "(see history). Vary the terminology or call done."
        )
    state.searched_keys.add(key)
    state.search_calls += 1
    state.note_queries(queries)
    more, _health, more_stats = await search_round(
        queries, min(state.max_results, 30),
        sources=state.sources, constraints=state.constraints,
    )
    state.query_stats.update(more_stats)
    new_n = state.merge(more)
    return (
        f"new papers: {new_n} (pool now {len(state.pool)}).\n"
        + _fmt_stats(state, queries)
    )


async def tool_snowball(
    state: HarnessState,
    anchor_dois: list[str],
    directions: list[str] | None = None,
    rationale: str = "",
) -> str:
    dirs = [d for d in (directions or ["backward", "forward", "related"])
            if d in ("backward", "forward", "related")]
    anchors = []
    for doi in anchor_dois[:5]:
        d = str(doi).lower()
        for p in state.pool:
            if p.doi and p.doi.lower() == d:
                anchors.append(p)
                break
    if not anchors:
        return "ERROR: no pool paper matches those DOIs. See pool_report for actual DOIs."
    state.snowball_calls += 1
    found = await snowball(
        anchors, {p.id for p in state.pool},
        backward="backward" in dirs, forward="forward" in dirs,
        related="related" in dirs, limit_per_paper=6,
        relevance_terms=state.keywords,
    )
    new_n = state.merge(found)
    top = "\n".join(
        f"- {p.title[:70]} [{p.citation_count}c]" for p in found[:5]
    )
    return f"snowball({','.join(dirs)}) from {len(anchors)} anchors: {new_n} new (pool {len(state.pool)}).\n{top}"


async def tool_pool_report(state: HarnessState) -> str:
    from citens.orchestration.support import facet_coverage_report

    by_source: dict[str, int] = {}
    for p in state.pool:
        src = p.source.split(" (")[0].split("(")[0][:14]
        by_source[src] = by_source.get(src, 0) + 1
    lines = [
        f"pool: {len(state.pool)} papers | sources: "
        + " · ".join(f"{k} {v}" for k, v in sorted(by_source.items(), key=lambda kv: -kv[1])),
        "top-cited:",
    ]
    for p in sorted(state.pool, key=lambda p: p.citation_count, reverse=True)[:6]:
        lines.append(f"- [{p.citation_count}c] {p.title[:70]} (doi:{p.doi or '-'})")
    if state.facets:
        cov = facet_coverage_report(state.facets, state.pool)
        thin = [r["facet"] for r in cov if r["papers"] < 3]
        lines.append(
            "facet coverage: " + "; ".join(f"{r['facet']}={r['papers']}" for r in cov)
        )
        if thin:
            lines.append(f"THIN facets (target these): {', '.join(thin)}")
    zero = [q for q, n in state.query_stats.items() if n == 0]
    if zero:
        lines.append(f"zero-hit queries so far: {', '.join(zero)}")
    return "\n".join(lines)


async def tool_pivot(state: HarnessState) -> str:
    """Mine the subfield's vocabulary out of the pool's own abstracts.

    For vocabulary-wall topics the question's words and every synonym
    reformulation fail together — bench-measured, one-hop snowball AND
    learned recommendations from the nearest neighbors miss too (the
    neighbors sit in an adjacent subfield). But the neighbors' TEXT names
    the target cluster's tasks/benchmarks/methods; queries coined from
    that vocabulary, verbatim, are the bridge.
    """
    import asyncio as _aio

    from citens.agents.pivot import pivot_from_abstracts

    # most question-relevant pool head: matched_queries counts how many of
    # our queries found the paper — a better neighbor signal than citations
    neighbors = sorted(
        state.pool,
        key=lambda p: (len(p.matched_queries or []), p.citation_count),
        reverse=True,
    )
    known = {k.lower() for k in state.keywords}
    queries = [
        q for q in await _aio.to_thread(
            pivot_from_abstracts, state.topic, neighbors[:12]
        )
        if q.lower() not in known
    ][:_MAX_QUERIES_PER_CALL]
    if not queries:
        return (
            "PIVOT: no new queries mined from the pool's abstracts (or all "
            "duplicate past searches). The pool's text does not name the "
            "target cluster — try read_paper on the closest candidates."
        )
    result = await tool_search(state, queries)
    return (
        "PIVOT mined queries from neighbor abstracts: "
        + "; ".join(queries)
        + f"\n{result}"
    )


async def tool_read_paper(state: HarnessState, match: str) -> str:
    m = str(match).lower()
    hits = [
        p for p in state.pool
        if m in (p.title or "").lower() or m in (p.doi or "").lower()
    ][:3]
    if not hits:
        return f"no pool paper matches '{match}'."
    return "\n\n".join(
        f"{p.title} (doi:{p.doi or '-'})\n"
        f"year {p.year} | {p.venue or 'no venue'} | {p.citation_count} citations "
        f"| source {p.source[:40]}\nabstract: {(p.abstract or '(none)')[:600]}"
        for p in hits
    )


# --- anchors: the external core-coverage check ---------------------------------

def _anchor_works(queries: list[str], per_query: int = 8) -> list[dict]:
    """Field-defining works per query: OpenAlex top-cited (sync, ~1s).

    Relevance search answers "what matches these words"; this answers "what
    did the field build on" — the axis a pool built from relevance searches
    systematically misses (measured: core50_recall 6% on the acceptance run).
    """
    import re as _re

    from citens import net
    from citens.config import settings

    params: dict[str, Any] = {"per-page": per_query, "sort": "cited_by_count:desc"}
    if settings.openalex_email:
        params["mailto"] = settings.openalex_email
    out: list[dict] = []
    seen: set[str] = set()
    with net.sync_client(timeout=15) as client:
        for q in queries:
            # anchors runs right after a search wave on the same host — a
            # 429 burst is common and clears in seconds; swallow it as
            # "no works" and the agent falsely concludes core coverage is
            # unknowable-or-fine (measured live on the RAG run)
            resp = None
            for attempt in range(3):
                resp = client.get(
                    "https://api.openalex.org/works",
                    params={"search": q, **params},
                )
                if resp.status_code != 429:
                    break
                import time as _time

                _time.sleep(2 * (attempt + 1))
            if resp is None or resp.status_code != 200:
                raise RuntimeError(
                    f"openalex HTTP {getattr(resp, 'status_code', 'none')} for anchors"
                )
            for w in resp.json().get("results", []):
                doi = ((w.get("doi") or "").replace("https://doi.org/", "")
                       .replace("https://openalex.org/", "").strip().lower())
                if doi in seen or not doi:
                    continue
                seen.add(doi)
                out.append({
                    "title": (w.get("display_name") or "").strip(),
                    "doi": doi,
                    "citations": w.get("cited_by_count") or 0,
                    "tokens": {
                        t for t in _re.split(r"[^a-z0-9]+",
                                             (w.get("display_name") or "").lower())
                        if len(t) > 2
                    },
                })
    return out


def _in_pool(anchor: dict, pool: list[Paper]) -> bool:
    for p in pool:
        if p.doi and anchor["doi"] and p.doi.lower() == anchor["doi"]:
            return True
        if not p.doi:
            ptok = {
                t for t in re.split(r"[^a-z0-9]+", (p.title or "").lower())
                if len(t) > 2
            }
            if len(ptok) >= 3 and ptok == anchor["tokens"]:
                return True
    return False


async def tool_anchors(state: HarnessState) -> str:
    """External falsifiability for 'is the pool enough': the topic's most-cited
    works vs the pool. Pool size measures effort; anchor overlap measures
    coverage — the two diverged exactly in the coverage eval."""
    import asyncio as _aio

    queries = [c.get("term", "") for c in state.plan.concepts[:2] if c.get("term")]
    if not queries:
        queries = state.keywords[:2]
    try:
        works = await _aio.to_thread(_anchor_works, queries)
    except Exception as exc:  # noqa: BLE001 - degrade to a warning, not a stop
        state.anchors_checked = True  # unreachable check must not gate done
        return f"anchors unavailable ({type(exc).__name__}) — proceed on pool signals."
    if not works:
        state.anchors_checked = True
        return "no anchor works found — proceed on pool signals."
    state.anchors_checked = True
    missing = [w for w in works if not _in_pool(w, state.pool)]
    hit = len(works) - len(missing)
    lines = [f"CORE COVERAGE: {hit}/{len(works)} field-defining works in pool."]
    for w in missing[:8]:
        lines.append(f"- MISSING [{w['citations']}c] {w['title'][:75]} (doi:{w['doi']})")
    if missing:
        lines.append(
            "Search the exact missing titles (quoted) if they belong in this "
            "review; otherwise justify in done.skipped."
        )
    else:
        lines.append("Core coverage complete.")
    return "\n".join(lines)


# --- OpenAI function schemas -------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Execute web-literature search queries against arXiv / Semantic "
                "Scholar / OpenAlex / Crossref. Max 4 queries per call. English "
                "only. Use when a direction needs evidence or a query phrase "
                "needs testing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array", "items": {"type": "string"},
                        "description": "1-4 concise English queries",
                    },
                    "rationale": {"type": "string", "description": "one line: why these queries"},
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "snowball",
            "description": (
                "Expand the pool from anchor papers via the citation graph "
                "(backward=references, forward=citing) or semantic neighbors "
                "(related). Anchors must be pool papers — use their DOIs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_dois": {"type": "array", "items": {"type": "string"}},
                    "directions": {
                        "type": "array", "items": {"type": "string"},
                        "enum": ["backward", "forward", "related"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["anchor_dois"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pool_report",
            "description": (
                "Perceive the current pool: size, source composition, top-cited "
                "papers with DOIs, facet coverage, thin facets, zero-hit "
                "queries. Call before deciding what to do next."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Full title/abstract/venue of pool papers matching a substring "
                "(title text or DOI). Use to judge whether a direction's "
                "evidence is strong or a snowball anchor is worth chaining from."
            ),
            "parameters": {
                "type": "object",
                "properties": {"match": {"type": "string"}},
                "required": ["match"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "anchors",
            "description": (
                "External core-coverage check: fetches the topic's most-cited "
                "(field-defining) works from OpenAlex and reports which are "
                "MISSING from the pool. Pool size measures effort; anchor "
                "overlap measures coverage. Required once before done."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pivot",
            "description": (
                "Mine new search queries from the abstracts of the pool's "
                "most relevant papers — the SUBFIELD's own vocabulary "
                "(task/benchmark/method names, verbatim), which the topic's "
                "wording lacks. Use when searches keep returning results "
                "that don't match what you need."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "End retrieval. Only when the completion checklist is met or "
                "remaining gaps are genuinely unreachable — state which "
                "directions you could not cover and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "what was achieved"},
                    "skipped": {
                        "type": "string",
                        "description": "directions not covered + why (be honest)",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


async def dispatch(name: str, args: dict, state: HarnessState) -> str:
    """Execute one tool call; errors become messages, never exceptions."""
    try:
        if name == "search":
            return await tool_search(state, **args)
        if name == "snowball":
            return await tool_snowball(state, **args)
        if name == "pool_report":
            return await tool_pool_report(state)
        if name == "anchors":
            return await tool_anchors(state)
        if name == "pivot":
            return await tool_pivot(state)
        if name == "read_paper":
            return await tool_read_paper(state, **args)
        return f"ERROR: unknown tool {name}."
    except Exception as exc:  # noqa: BLE001 - tool errors are model feedback
        return f"ERROR: {type(exc).__name__}: {exc}"
