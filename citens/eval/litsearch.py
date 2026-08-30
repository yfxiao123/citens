"""LitSearch live-retrieval benchmark (maintainer tool, ``citens bench``).

The search stack never got an external yardstick: every number so far is
internal (pool size, yield, precision). LitSearch (Princeton NLP, EMNLP
2024) provides 597 real literature-search questions with gold papers —
this harness samples them, runs each through the PRODUCTION searchers
over live APIs, and scores gold recall@5/@20 (matched by DOI, then arXiv
id, then normalized title). Variants isolate each lever:

    single:<src>   one source alone, in its native relevance order
    union          all sources, round-robin interleave (no LLM, no scores)
    planned        planner (cheap LLM) turns the question into keyword
                   queries first — isolates query-formulation value; the
                   sources return EMPTY for long natural-language questions
                   (measured: S2/OpenAlex return 0 hits on raw LitSearch
                   questions), so this is the minimum viable config
    rank           union re-ordered by the production ranker (citations+venue,
                   the exact composite citens uses after filtering)
    llm_rerank     union top-30 listwise-reranked by the cheap model (the
                   analog of LitSearch's GPT-4 rerank row, on our tier)
    agentic        the full hybrid agent: plan -> search -> harness loop
                   (pool-hit rate instead of recall@k — the harness returns
                   a pool, not a ranking)

Protocol note: LitSearch's published BM25/dense rows run against a fixed
16k-abstract corpus; this is LIVE retrieval, so those numbers are context,
not a leaderboard. Their Google Custom Search row is the closest analog to
what we measure.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from citens import cache
from citens.events import EventBus
from citens.models import Paper, ScoredPaper
from citens.ranking import rank_papers
from citens.search.base import paper_arxiv_id, search_round

PER_SOURCE = 20  # per-source result cap == the union@20 ceiling
PLANNED_DEPTH = 40  # planned-leg per-cell depth (fused top-20 is the answer)
# each query family's guaranteed seats in the adaptive candidate pool
_ADAPTIVE_FAMILY_QUOTA = 30
# query-generation cache version — bump when a generator prompt changes, so
# frozen generations from an older prompt are not reused
_GEN_CACHE_VER = 2


def _cached_queries(
    kind: str,
    question: str,
    gen: Callable[[], list[str]],
    path: Path | None = None,
) -> list[str]:
    """Freeze LLM query generation per question (planner / hyde / pivot).

    Measured problem: every generator regenerates per run (planner combos,
    pivot-mined phrases), so between-run comparisons measured generation
    luck, not the lever under test — a hyde-caught gold vanished next run
    when the planner redraw changed the fusion mix. With generation frozen
    AND search results cached, only the ranking lever varies. The cache
    lives under bench_results/ (gitignored); bump _GEN_CACHE_VER to reseed.
    """
    p = path or Path("bench_results") / "gen_cache.json"
    key = f"{_GEN_CACHE_VER}:{kind}:{question}"
    try:
        cache_doc = (
            json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        )
    except (OSError, json.JSONDecodeError):
        cache_doc = {}
    if key in cache_doc:
        return [str(q) for q in cache_doc[key]]
    out = [str(q) for q in gen()]
    try:
        cache_doc[key] = out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(cache_doc, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except OSError:
        pass  # cache write is best-effort; the run proceeds regardless
    return out


# --- data loading ------------------------------------------------------------


def load_queries(data_dir: str | Path) -> list[dict]:
    """597 LitSearch questions from the parquet snapshot in bench_data/."""
    import pyarrow.parquet as pq

    path = Path(data_dir) / "query.parquet"
    return pq.read_table(str(path)).to_pylist()


def load_gold(data_dir: str | Path) -> dict[str, dict]:
    path = Path(data_dir) / "gold_papers.json"
    return json.load(open(path, encoding="utf-8"))


def sample_queries(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified by query_set — the four sets have very different difficulty
    (inline_nonacl is the hardest); a plain sample would swing ±10 points."""
    by_set: dict[str, list[dict]] = {}
    for r in rows:
        by_set.setdefault(r["query_set"], []).append(r)
    rng = random.Random(seed)
    take: list[dict] = []
    remaining = n
    sets = sorted(by_set)
    for i, s in enumerate(sets):
        quota = round(n * len(by_set[s]) / len(rows))
        if i == len(sets) - 1:
            quota = remaining
        take.extend(rng.sample(by_set[s], min(quota, len(by_set[s]))))
        remaining -= min(quota, len(by_set[s]))
    rng.shuffle(take)
    return take


# --- gold matching -----------------------------------------------------------


def _norm_doi(doi: str | None) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (doi or "").strip().lower())


def _title_tokens(title: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (title or "").lower()) if len(t) > 2}


def match_gold(paper: Paper, golds: list[dict]) -> str | None:
    """How this paper matches one of the query's golds: doi/arxiv/title."""
    pdoi = _norm_doi(paper.doi)
    parx = paper_arxiv_id(paper)
    ptok = _title_tokens(paper.title)
    for g in golds:
        if pdoi and g.get("doi") and pdoi == _norm_doi(g["doi"]):
            return "doi"
        if parx and g.get("arxiv") and g["arxiv"] == parx:
            return "arxiv"
        gtok = _title_tokens(g.get("title", ""))
        # >=3 informative tokens — short numbered titles collapse to a
        # 2-token skeleton that would match everything
        if len(ptok) >= 3 and len(gtok) >= 3 and ptok == gtok:
            return "title"
    return None


def gold_hits(ordered: list[Paper], golds: list[dict], k: int) -> float:
    """Recall@k: fraction of the query's golds appearing in the top-k."""
    if not golds:
        return 0.0
    seen = 0
    matched: set[int] = set()
    for p in ordered[:k]:
        for i, g in enumerate(golds):
            if i in matched:
                continue
            if match_gold(p, [g]):
                matched.add(i)
                seen += 1
    return seen / len(golds)


# --- search variants ---------------------------------------------------------


async def search_one_query(
    query: str,
    sources: list[str],
    per_source: int = PER_SOURCE,
    health_out: dict[str, list[int]] | None = None,
) -> dict[str, list[Paper]]:
    """Per-source result lists for one question (native relevance order).

    ``health_out`` accumulates per-source cell buckets [ok, empty, failed]
    across every call that passes it — an exhausted provider 429s into
    empty results that are indistinguishable from "the words didn't match"
    unless counted (measured: OpenAlex dies mid-day and silently zeros
    whole bench legs).
    """

    async def _one(s: str) -> list[Paper]:
        try:
            papers, health, _stats = await search_round(
                [query], max_results=per_source, sources=[s]
            )
            if health_out is not None:
                b = health_out.setdefault(s, [0, 0, 0])
                b[0] += 1
                if not papers:
                    h = health.get(s, "")
                    if h.startswith("failed"):
                        b[2] += 1
                        b[0] -= 1
                    else:
                        b[1] += 1
            return papers
        except Exception:  # noqa: BLE001 — a dead source must not kill the bench
            if health_out is not None:
                health_out.setdefault(s, [0, 0, 0])[2] += 1
            return []

    results = await asyncio.gather(*(_one(s) for s in sources))
    return dict(zip(sources, results, strict=True))


def _key_of(p: Paper) -> str:
    return ("d:" + _norm_doi(p.doi)) if p.doi else (
        "t:" + " ".join(sorted(_title_tokens(p.title)))
    )


def oa_title_filter(title: str) -> str:
    """OpenAlex ``filter=title.search:`` payload for one candidate title.

    The relevance endpoint ranks full-text-ish matches; title.search is the
    dedicated title-index — for HyDE-generated titles it is precisely the
    near-exact-title matching shape that our dissection found to be the
    only reliable lexical retrieval form. Informative words only (the
    filter treats stopwords loosely but short tokens add noise)."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 2]
    return " ".join(words[:10])


async def oa_title_search(
    titles: list[str], per_page: int = 20, fetch=None
) -> dict[str, list[Paper]]:
    """One OpenAlex title.search cell per candidate title (HyDE family).

    Cached like search results; empty responses are NOT cached (a throttled
    provider must not poison retries). ``fetch`` is injectable for tests.
    """
    import httpx

    from citens.config import settings

    async def _default_fetch(filt: str) -> list[dict]:
        headers = {}
        if settings.openalex_email:
            headers["User-Agent"] = f"mailto:{settings.openalex_email}"
        from citens.search.openalex import apply_api_key

        async with httpx.AsyncClient(timeout=25, headers=headers) as hc:
            r = await hc.get(
                "https://api.openalex.org/works",
                params=apply_api_key(
                    {"filter": f"title.search:{filt}", "per-page": per_page}
                ),
            )
            if r.status_code != 200:
                raise RuntimeError(f"openalex {r.status_code}")
            await asyncio.sleep(0.2)
            return r.json().get("results", [])

    fetch = fetch or _default_fetch
    out: dict[str, list[Paper]] = {}
    for t in titles:
        cache_ns, cache_key = "eval.oa-title", {"t": t}
        cached = cache.get(cache_ns, cache_key)
        if cached is None:
            try:
                works = await fetch(oa_title_filter(t))
            except Exception as e:  # noqa: BLE001 — a dead OA stays dead
                print(f"[oa-title] {t[:40]}… failed: {e}", flush=True)
                out[t] = []
                continue
            cached = [
                {
                    "title": w.get("display_name") or "",
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "year": w.get("publication_year"),
                    "cited": w.get("cited_by_count") or 0,
                    "abstract": "",
                }
                for w in works
            ]
            if any(c["title"] for c in cached):
                cache.put(cache_ns, cache_key, cached)
        out[t] = [
            Paper(
                title=c["title"],
                authors=[],
                year=c["year"],
                doi=c["doi"] or None,
                citation_count=c["cited"],
                source="oa-title",
            )
            for c in cached
        ]
    return out


def interleave(per_source: dict[str, list[Paper]], order: list[str]) -> list[Paper]:
    """Round-robin fusion: rank r from every source before rank r+1 from any.
    Score-free, deterministic, and the fairest 'no-LLM' baseline."""
    merged: list[Paper] = []
    seen: set[str] = set()
    for r in range(PER_SOURCE):
        for s in order:
            if r >= len(per_source.get(s, [])):
                continue
            p = per_source[s][r]
            key = _key_of(p)
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
    return merged


def ranked_order(union: list[Paper]) -> list[Paper]:
    """The production ranker on the union (relevance unknown at retrieval
    time, so relevance_score=0 for all — this is citations+venue exactly as
    the pipeline ranks after the filter stage)."""
    scored = [ScoredPaper(**p.model_dump()) for p in union]
    return list(rank_papers(scored))


def plan_keyword_queries(question: str, cap: int = 3) -> list[str]:
    """The production planner (cheap tier) distilled to keyword queries.

    Combination queries first: the planner assembles both single-concept
    coverage queries and multi-concept precision combos, and for a
    find-that-paper question the gold title usually IS a concept combination
    ("entity-level factual consistency abstractive summarization"). Taking
    the assembled list head-first picked only broad coverage queries and
    measured 0/15 — combo-first selection is the fix.
    """
    from citens.agents.planner import plan_queries

    plan = plan_queries(question, None)
    ranked = sorted(plan.queries, key=len, reverse=True)
    return ranked[:cap]


def fuse_multi_query(
    per_query: dict[str, dict[str, list[Paper]]], sources: list[str]
) -> list[Paper]:
    """Production-style pool fusion for the planned leg: Reciprocal Rank
    Fusion — score = sum over (query, source) cells of 1/(60 + rank), the
    standard score-free combiner. Round-robin interleave buried papers that
    ranked high in ONE cell (measured: a gold at cell-rank ~5 landed fused
    #29); RRF promotes consensus plus single-cell excellence."""
    scores: dict[str, float] = {}
    papers_by_key: dict[str, Paper] = {}
    for qlists in per_query.values():
        for s in sources:
            for r, p in enumerate(qlists.get(s, [])):
                # score by KEY, not object identity: the same work returns
                # as distinct Paper objects from different cells
                key = _key_of(p)
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + r)
                papers_by_key.setdefault(key, p)
    return sorted(papers_by_key.values(), key=lambda p: -scores[_key_of(p)])


def llm_rerank(query: str, union: list[Paper], top: int = 30) -> list[Paper]:
    """Listwise rerank of the union's top-N by the cheap model — our tier's
    version of LitSearch's GPT-4 rerank leg. Falls back to union order.
    Shared implementation: citens.agents.rerank (the same judgment also
    ranks harness find output); the adaptive leg calls it with top=100 over
    the fused pool."""
    from citens.agents.rerank import listwise_rank

    return listwise_rank(query, union, top=top)


# --- snowball observation leg (graph-bridging diagnosis) ----------------------


def pick_snowball_anchors(fused: list[Paper], k: int = 5) -> list[Paper]:
    """The k most question-relevant DOI-carrying papers from the fused order.

    Snowball resolves anchors by DOI (OpenAlex), and the fused head is the
    best deterministic relevance order the stack produces.
    """
    return [p for p in fused if p.doi][:k]


async def run_snowball_leg(
    query: str,
    golds: list[dict],
    anchors: list[Paper],
    existing_ids: set[str],
    relevance_terms: list[str],
) -> dict:
    """Is the gold REACHABLE from the question's neighbors, and which channel
    lets it through?

    Four fractions per query:
      reach  gold anywhere in the one-hop citation windows (backward/forward/
             related, 100-wide per direction — the raw graph ceiling). Low
             reach = the vocab-wall golds cite neither our anchors nor the
             reverse; one-hop citation chaining cannot bridge them.
      lex20  gold in the top-20 of today's lexical rank (relevance_terms =
             planner keywords — production behavior)
      recs   gold in S2's LEARNED recommendations fed all anchors as
             positives at once — the model pools them into a topical
             embedding, which is the channel citation edges cannot provide
      sem20  gold in the top-20 after a cheap-LLM listwise rerank against
             the question over (recs top-10 + lexical top-20) — the two-stage
             funnel a production bridge would actually run
    """
    from citens.search.snowball import (
        _s2_recommended,
        _s2_to_paper,
    )
    from citens.search.snowball import (
        snowball as snowball_expand,
    )

    cands = await snowball_expand(
        anchors, existing_ids, backward=True, forward=True, related=True,
        # observation windows are wider than production (6): the S2 limit is
        # a query param, so a wide window costs the same call count — this
        # leg measures the ceiling, not the production funnel
        limit_per_paper=100, relevance_terms=relevance_terms, top=500,
    )
    reach = gold_hits(cands, golds, len(cands))
    lex20 = gold_hits(cands, golds, 20)
    recs = [
        p for p in (
            _s2_to_paper(i, "recs", "pool")
            for i in await _s2_recommended(
                [a.doi for a in anchors if a.doi], 20
            )
        ) if p
    ]
    recs_reach = gold_hits(recs, golds, len(recs))
    sem_pool = (recs[:10] + cands[:20])[:30]
    sem = await asyncio.to_thread(llm_rerank, query, sem_pool, 30) if sem_pool else []
    sem20 = gold_hits(sem, golds, 20)
    return {
        "reach": round(reach, 3),
        "lex20": round(lex20, 3),
        "recs": round(recs_reach, 3),
        "sem20": round(sem20, 3),
        "n_cands": len(cands),
        "n_recs": len(recs),
        "anchors": [p.title[:50] for p in anchors],
    }


# --- agentic leg -------------------------------------------------------------


async def run_agentic(query: str, sources: list[str] | None,
                      goal: str = "find") -> dict:
    """The hybrid agent on one question: plan -> seed wave -> harness loop.

    Costs real LLM calls (planner + orchestrator turns). Returns pool-hit
    (gold found at all), pool size, and cost counters — the harness's pool
    IS its answer; recall@k needs a ranking it doesn't produce. The default
    goal="find" turns on targeted mode: pool size stops counting as success
    and done must name the papers it found.
    """
    from citens.agents.planner import plan_queries
    from citens.harness import HarnessBudget, run_retrieval_harness
    from citens.harness.tools import HarnessState

    t0 = time.monotonic()
    plan = plan_queries(query, None)
    papers, _h, stats = await search_round(plan.queries[:4], max_results=40,
                                           sources=sources)
    state = HarnessState(
        topic=query,
        plan=plan,
        pool=list(papers),
        keywords=list(plan.queries),
        query_stats=dict(stats),
        sources=sources,
        max_results=40,
        goal=goal,
    )
    # production defaults for the survey goal. find gets more room: every
    # agentic miss so far (seed 13 and 42) ended budget:steps mid-dig —
    # still reformulating vocabulary, not stalling — so the dig budget is
    # one notch wider than the seed-42 16/18 that cut both misses off
    budget = HarnessBudget()
    if goal == "find":
        budget = HarnessBudget(max_steps=20, max_llm_calls=22, max_search_calls=6)
    res = await run_retrieval_harness(state, bus=EventBus(), budget=budget)
    return {
        "pool": res.papers,
        "pool_unranked": res.papers_unranked,
        "queries": plan.queries,
        "steps": res.steps,
        "llm_calls": res.llm_calls,
        "search_calls": res.search_calls,
        "finish_reason": res.finish_reason,
        "wall_s": round(time.monotonic() - t0, 1),
    }


# --- runner ------------------------------------------------------------------


@dataclass
class BenchResult:
    summary: dict = field(default_factory=dict)
    details: list[dict] = field(default_factory=list)


async def run_bench(
    data_dir: str | Path = "bench_data/litsearch",
    n: int = 50,
    seed: int = 13,
    sources: list[str] | None = None,
    with_planned: bool = False,
    with_adaptive: bool = False,
    with_expand: bool = False,
    with_llm_rerank: bool = False,
    with_snowball: bool = False,
    agentic_n: int = 0,
    detail_path: str | Path | None = None,
) -> BenchResult:
    if sources is None:
        sources = ["semantic_scholar", "openalex", "arxiv", "crossref"]
    rows = sample_queries(load_queries(data_dir), n, seed)
    gold_map = load_gold(data_dir)
    result = BenchResult()
    sums: dict[str, dict[str, float]] = {}
    # incremental flush: a killed run keeps every completed query (learned
    # the hard way — a 50-query run died at 26 with everything in memory).
    # Closed explicitly after the loops; a with-block would span them all.
    detail_fh = open(detail_path, "a", encoding="utf-8") if detail_path else None  # noqa: SIM115

    def _flush(obj: dict) -> None:
        if detail_fh:
            detail_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            detail_fh.flush()

    def _acc(variant: str, r5: float, r20: float) -> None:
        bucket = sums.setdefault(variant, {"r5": 0.0, "r20": 0.0, "n": 0})
        bucket["r5"] += r5
        bucket["r20"] += r20
        bucket["n"] += 1

    for qi, row in enumerate(rows):
        query = row["query"]
        golds = [gold_map[str(c)] for c in row["corpusids"] if str(c) in gold_map]
        if not golds:
            continue
        # per-source cell health [ok, empty, failed] across ALL calls this
        # question makes — a throttled provider shows up as failures, not
        # as innocent zero-result queries
        cell_health: dict[str, list[int]] = {}

        async def _soq(
            qtext: str,
            depth: int = PER_SOURCE,
            _h: dict[str, list[int]] = cell_health,
        ) -> dict[str, list[Paper]]:
            return await search_one_query(
                qtext, sources, per_source=depth, health_out=_h
            )

        per_source = await _soq(query)
        union = interleave(per_source, sources)
        det: dict[str, list[Paper]] = {f"single:{s}": per_source[s] for s in sources}
        det["union"] = union
        det["rank"] = ranked_order(union)
        rerank_t = None
        if with_llm_rerank:
            # independent of the planned leg (needs only the union) — run
            # the blocking LLM call concurrently with the planned searches
            rerank_t = asyncio.create_task(
                asyncio.to_thread(llm_rerank, query, union)
            )
        if with_planned:
            qs = await asyncio.to_thread(
                _cached_queries, "planned", query,
                partial(plan_keyword_queries, query),
            )
            per_query = dict(zip(
                qs,
                await asyncio.gather(*(_soq(q, PLANNED_DEPTH) for q in qs)),
                strict=True,
            ))
            # one depth-40 fetch feeds both legs: RRF only ever sums ranks,
            # so truncating each cell to PER_SOURCE reproduces the depth-20
            # run exactly — the depth lever is measured at zero extra API cost
            truncated = {
                q: {s: lst[:PER_SOURCE] for s, lst in cells.items()}
                for q, cells in per_query.items()
            }
            det["planned"] = fuse_multi_query(truncated, sources)
            det["planned_d40"] = fuse_multi_query(per_query, sources)
            # the question's own phrasing joins the fusion (cells already
            # fetched for the union leg — zero extra API): crossref title
            # matching sometimes beats every keyword combo (measured seed 42:
            # union 100% while planned 0% — the question's words were IN the
            # gold title and the planner's combos were not)
            det["planned_raw"] = fuse_multi_query(
                per_query | {query: per_source}, sources
            )
            planned_qs = qs
        if with_adaptive and with_planned:
            # adaptive leg — every vocabulary-crossing query form in one
            # pool, then one semantic ranking. The two new forms attack the
            # vocab wall from opposite sides: hypothetical TITLE queries
            # (question side, HyDE — the only retrieval shape our dissection
            # found reliable is near-exact-title matching) and PRF-mined
            # entities quoted as phrases (corpus side, bench-validated
            # 13.3% vs planned 6.7%). RRF alone DILUTES such cells
            # (pivot-only 13.3% -> 6.7% once fused), so the final order is
            # the LLM listwise ranking over the fused top-100 — our tier's
            # substitute for the dense retriever (measured: llm_rerank
            # doubled r@5 whenever the gold was in the candidate pool).
            from citens.agents.hypothetical import hypothetical_queries
            from citens.agents.pivot import pivot_from_abstracts
            from citens.agents.rerank import cascade_rank

            hyde_qs = [
                q for q in await asyncio.to_thread(
                    _cached_queries, "hyde", query,
                    partial(hypothetical_queries, query),
                )
                if q and q.lower() not in {k.lower() for k in per_query}
            ]
            piv_head = det["planned_raw"][:10]
            piv_qs = await asyncio.to_thread(
                _cached_queries, "pivot", query,
                partial(pivot_from_abstracts, query, piv_head),
            )
            # mined names are exact entity strings — quoting keeps the terms
            # together; hypothetical titles are searched raw (title-shaped
            # already, and crossref title-text matching likes full strings)
            piv_qs = [
                f'"{q}"' if " " in q and '"' not in q else q
                for q in piv_qs
                if q.lower() not in {k.lower() for k in per_query}
            ]
            extra_qs = [
                q for q in dict.fromkeys(hyde_qs + piv_qs)
                if q.lower() != query.lower()
            ]
            extra_cells = dict(zip(
                extra_qs,
                await asyncio.gather(*(_soq(q) for q in extra_qs)),
                strict=True,
            )) if extra_qs else {}
            # global RRF over everything — kept as the dilution witness: a
            # gold at cell-rank 0 in ONE cell loses to consensus noise from
            # a dozen cells (live probe: union's rank-0 crossref hit fell
            # out of the fused top-100, so the semantic rank never saw it)
            det["adaptive_rrf"] = fuse_multi_query(
                per_query | {query: per_source} | extra_cells, sources
            )
            # the pool the LLM actually judges: each query FAMILY's own
            # fused head (quota), not the global fusion — every family's
            # best papers are guaranteed to reach the semantic rank
            families = [
                fuse_multi_query(per_query, sources),
                fuse_multi_query({query: per_source}, sources),
                fuse_multi_query(
                    {q: c for q, c in extra_cells.items() if q in set(hyde_qs)},
                    sources,
                ),
                fuse_multi_query(
                    {q: c for q, c in extra_cells.items() if q in set(piv_qs)},
                    sources,
                ),
            ]
            # fifth family: OpenAlex's dedicated title index over the HyDE
            # titles — title.search is the endpoint-shaped version of the
            # only retrieval form our dissection found reliable
            oa_title_cells = await oa_title_search(hyde_qs[:4])
            if any(oa_title_cells.values()):
                families.append(fuse_multi_query(
                    {f"[title] {t}": {"openalex": cs}
                     for t, cs in oa_title_cells.items()},
                    ["openalex"],
                ))
                detail_adaptive_queries_extra = {
                    "oa_title": [t for t, v in oa_title_cells.items() if v]
                }
            else:
                detail_adaptive_queries_extra = {"oa_title": []}
            pool: list[Paper] = []
            seen_keys: set[str] = set()
            for fam in families:
                for p in fam[:_ADAPTIVE_FAMILY_QUOTA]:
                    k = _key_of(p)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        pool.append(p)
            # sixth family: one-hop citation expansion from the pool's own
            # head — a live probe (seed 42) reached a never-hit gold via a
            # forward citation edge from a HyDE-family anchor that lexical
            # retrieval had placed correctly; the earlier all-zero verdict
            # came from WRONG-subfield (planned) anchors, not from the
            # channel being dead
            expanded: list[Paper] = []
            if with_expand:
                from citens.search.snowball import snowball

                anchors = [p for p in pool[:8] if p.doi][:5]
                if anchors:
                    try:
                        existing = seen_keys
                        expanded = await snowball(
                            anchors, existing_ids=existing,
                            backward=True, forward=True, related=False,
                            limit_per_paper=100, top=120,
                        )
                    except Exception as e:  # noqa: BLE001 — expansion is optional
                        print(f"[expand] failed: {e}", flush=True)
            if expanded:
                seen2 = set(seen_keys)
                for p in expanded[:_ADAPTIVE_FAMILY_QUOTA]:
                    k = _key_of(p)
                    if k not in seen2:
                        seen2.add(k)
                        pool.append(p)
            det["adaptive"] = await asyncio.to_thread(
                cascade_rank, query, pool, 50, True
            )
            detail_adaptive_queries = {
                "hyde": hyde_qs,
                "pivot": piv_qs,
                **detail_adaptive_queries_extra,
            }
        if rerank_t is not None:
            det["llm_rerank"] = await rerank_t

        detail: dict = {
            "i": qi,
            "query_set": row["query_set"],
            "query": query,
            "golds": [g.get("title", "")[:80] for g in golds],
        }
        if with_planned:
            detail["planned_queries"] = planned_qs
        if with_adaptive and with_planned:
            detail["adaptive_queries"] = detail_adaptive_queries
        detail["cells"] = {
            s: {"ok": b[0], "empty": b[1], "failed": b[2]}
            for s, b in cell_health.items()
        } if cell_health else None
        for name, ordered in det.items():
            r5 = gold_hits(ordered, golds, 5)
            r20 = gold_hits(ordered, golds, 20)
            _acc(name, r5, r20)
            detail[name] = {"r5": round(r5, 3), "r20": round(r20, 3),
                            "hits5": _match_types(ordered[:5], golds)}
        if with_snowball and with_planned:
            anchors = pick_snowball_anchors(det["planned_raw"])
            sb: dict | None = None
            if anchors:
                try:
                    sb = await run_snowball_leg(
                        query, golds, anchors,
                        {p.id for p in union} | {p.id for p in det["planned_raw"]},
                        planned_qs,
                    )
                    _acc("snowball_reach", sb["reach"], sb["reach"])
                    _acc("snowball_lex@20", sb["lex20"], sb["lex20"])
                    _acc("snowball_recs", sb["recs"], sb["recs"])
                    _acc("snowball_sem@20", sb["sem20"], sb["sem20"])
                except Exception as e:  # noqa: BLE001 — diagnosis leg, not the bench
                    sb = {"error": str(e)[:120]}
            else:
                sb = {"error": "no DOI-carrying anchors"}
            detail["snowball"] = sb
        result.details.append(detail)
        _flush(detail)
        got = {s: len(v) for s, v in per_source.items()}
        pl = detail.get("planned", {}).get("r20")
        pl_s = f" planned r@20={pl:.0%}" if pl is not None else ""
        ad = detail.get("adaptive", {})
        ad_s = (
            f" adaptive rrf@20={detail['adaptive_rrf']['r20']:.0%}"
            f" r@5={ad['r5']:.0%} r@20={ad['r20']:.0%}"
            if ad else ""
        )
        sbd = detail.get("snowball")
        sb_s = (
            f" | sb reach={sbd['reach']:.0%} lex@20={sbd['lex20']:.0%}"
            f" recs={sbd['recs']:.0%} sem@20={sbd['sem20']:.0%}"
            if sbd and "reach" in sbd else ""
        )
        dead = [s for s, b in cell_health.items() if b[2] > 0]
        dead_s = f" !! DEAD:{','.join(dead)}" if dead else ""
        print(f"[{qi + 1}/{len(rows)}] {row['query_set']:14s} "
              f"union r@5={detail['union']['r5']:.0%} r@20={detail['union']['r20']:.0%}"
              f"{pl_s}{ad_s}{sb_s}{dead_s} ({got})", flush=True)

    # agentic leg on the first agentic_n sampled questions
    for qi in range(min(agentic_n, len(result.details))):
        row = rows[qi]
        golds = [gold_map[str(c)] for c in row["corpusids"] if str(c) in gold_map]
        if not golds:
            continue
        try:
            ag = await run_agentic(row["query"], sources)
        except Exception as e:  # noqa: BLE001 — keep the bench alive
            result.details[qi]["agentic"] = {"error": str(e)[:120]}
            _flush({"i": qi, "agentic": result.details[qi]["agentic"]})
            continue
        hit = sum(1 for g in golds if any(match_gold(p, [g]) for p in ag["pool"]))
        r5 = gold_hits(ag["pool"], golds, 5)
        # find pools are LLM-ranked at harness finish now; the raw
        # insertion-order metric stays in the record to show the gain
        raw = ag.get("pool_unranked") or []
        r5_raw = gold_hits(raw, golds, 5) if raw else r5
        rec = {"pool_hit": round(hit / len(golds), 3), "pool_top5": round(r5, 3),
               "pool_top5_raw": round(r5_raw, 3),
               "pool_size": len(ag["pool"]), "queries": ag["queries"][:6],
               "steps": ag["steps"], "llm_calls": ag["llm_calls"],
               "search_calls": ag["search_calls"],
               "finish_reason": ag["finish_reason"], "wall_s": ag["wall_s"]}
        result.details[qi]["agentic"] = rec
        _flush({"i": qi, "agentic": rec})
        bucket = sums.setdefault("agentic", {"r5": 0.0, "r20": 0.0, "n": 0})
        bucket["r5"] += rec["pool_top5"]
        bucket["r20"] += rec["pool_hit"]
        bucket["n"] += 1
        print(f"[agentic {qi + 1}] pool_hit={rec['pool_hit']:.0%} "
              f"pool={rec['pool_size']} llm={rec['llm_calls']} {rec['wall_s']}s",
              flush=True)

    if detail_fh:
        detail_fh.close()
    result.summary = {
        name: {"recall@5": round(b["r5"] / b["n"], 4),
               "recall@20": round(b["r20"] / b["n"], 4),
               "n": b["n"]}
        for name, b in sums.items()
    }
    return result


def _match_types(ordered: list[Paper], golds: list[dict]) -> list[str]:
    out: list[str] = []
    for p in ordered:
        m = match_gold(p, golds)
        out.append(m or "-")
    return out
