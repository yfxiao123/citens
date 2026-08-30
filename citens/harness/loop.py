"""The budgeted agent loop for agentic retrieval.

Perceive → decide → act → repeat, with the safety rails that separate a
harness from a toy: a budget ledger (steps / LLM calls / search calls),
duplicate-call rejection (in the tools), saturation detection (consecutive
searches with no new papers), and forced convergence into the deterministic
pipeline once any budget runs out. Every decision is emitted as an event —
the web transcript shows the loop thinking and acting in real time.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from citens import llm
from citens.events import EventBus, StepProgress
from citens.harness.tools import TOOL_SCHEMAS, HarnessState, dispatch
from citens.orchestration.support import _emit


@dataclass
class HarnessBudget:
    max_steps: int = 12          # tool calls executed
    max_llm_calls: int = 14      # orchestrator turns
    max_search_calls: int = 5    # search() invocations (snowball separate)
    max_pool: int = 150          # stop adding papers beyond this


def _find_claimed_titles(summary: str) -> list[str]:
    """Title-ish chunks from a find-mode done summary, for pool-match
    verification. Heuristic: quoted strings, then lines/sentences that look
    like titles (4+ words with a capitalized word). Imperfect on purpose —
    only chunks with >=3 informative tokens get verified, so false bounces
    stay rare."""
    import re as _re

    chunks = _re.findall(r'"([^"]{10,160})"', summary)
    if not chunks:
        chunks = [
            line.strip(" -*•\t")
            for line in summary.splitlines()
            if len(line.split()) >= 4
        ]
    return chunks[:6]


@dataclass
class HarnessResult:
    papers: list = field(default_factory=list)
    # insertion order before find-mode output ranking (empty in survey mode);
    # keeps the pre-ranking order auditable next to the ranked deliverable
    papers_unranked: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    query_stats: dict = field(default_factory=dict)
    summary: str = ""
    skipped: str = ""
    steps: int = 0
    llm_calls: int = 0
    search_calls: int = 0
    finish_reason: str = ""


SYSTEM_PROMPT = """You are the retrieval director of an academic literature-review agent. \
The pipeline has already run an initial keyword search; your job is the ADAPTIVE part: \
perceive what the pool actually contains, judge where coverage is thin or phrasing is \
wrong, and act — more searches (different terminology), citation/semantic snowball, \
or reading papers to judge a direction — until the pool covers the topic.

Rules:
1. English queries only (all sources are English corpora).
2. Max 4 queries per search call; each query 2-6 words; no near-duplicates of past queries.
3. A zero-hit query means the field does not use that phrasing — switch terminology \
(synonyms were provided in the plan).
4. Call pool_report before your first action and whenever you are unsure what to do.
5. Snowball anchors must be DOIs of papers already in the pool (see pool_report).
6. Run the anchors tool once before calling done: if field-defining works are \
MISSING from the pool, search their exact quoted titles (they belong in the review) \
or justify each miss in 'skipped'. A large pool is NOT coverage — anchor overlap is.
7. Completion checklist — call done only when: every search facet has >=3 papers AND \
the anchors check passed (or misses are justified), OR you have made a genuine \
attempt (>=1 follow-up action) per thin facet and explain in 'skipped' why it stays thin.
8. Never fabricate: papers only exist if a tool returned them.
9. When searches return results but the RIGHT papers are missing (wrong \
subfield vocabulary), call pivot: it mines the field's own task/benchmark/method \
names from the pool's abstracts. Search any mined names as quoted phrases."""

_FIND_GOAL_SUFFIX = """

TARGETED MODE: the question asks for SPECIFIC paper(s), not a survey pool. Pool size \
is NOT success. Use the question's key terms as quoted exact-phrase searches; read \
the candidates you find; call done only when you can name the found paper(s) — exact \
titles — in your summary, or explain precisely what you tried and why it failed. \
If the question's phrasings keep missing, call pivot once to mine the subfield's \
vocabulary from neighbor abstracts, then search the mined names as quoted phrases."""


def _system_prompt(state: HarnessState) -> str:
    if state.goal == "find":
        return SYSTEM_PROMPT + _FIND_GOAL_SUFFIX
    return SYSTEM_PROMPT


def _initial_user_prompt(state: HarnessState, budget: HarnessBudget) -> str:
    from citens.orchestration.support import facet_coverage_report

    lines = [
        f"Topic: {state.topic}",
        f"Budget: <= {budget.max_search_calls} search calls, "
        f"{budget.max_steps} tool steps total.",
        "",
        "Query plan (concepts + untried synonyms):",
    ]
    for c in state.plan.concepts:
        lines.append(f"- {c.get('term', '')} | synonyms: {', '.join(c.get('synonyms', []))}")
    lines.append(f"\nQueries already executed (initial round): {', '.join(state.keywords[:12])}")
    if state.facets:
        cov = facet_coverage_report(state.facets, state.pool)
        lines.append(
            "Facet coverage of the current pool: "
            + "; ".join(f"{r['facet']}={r['papers']}" for r in cov)
        )
    lines.append(
        f"\nCurrent pool: {len(state.pool)} papers. Decide your first action "
        "(pool_report is recommended)."
    )
    return "\n".join(lines)


def _assistant_message(msg: Any) -> dict:
    """Serialize an SDK assistant message back into the transcript list."""
    out: dict = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


async def run_retrieval_harness(
    state: HarnessState,
    bus: EventBus | None = None,
    budget: HarnessBudget | None = None,
) -> HarnessResult:
    """Run the perceive-decide-act loop to convergence or budget exhaustion."""
    budget = budget or HarnessBudget()
    state.max_search_calls = budget.max_search_calls
    res = HarnessResult(
        papers=state.pool, keywords=state.keywords, query_stats=state.query_stats
    )
    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(state)},
        {"role": "user", "content": _initial_user_prompt(state, budget)},
    ]
    steps = 0
    llm_calls = 0
    no_tool_strikes = 0
    empty_search_streak = 0

    async def finish(
        reason: str, summary: str = "", skipped: str = ""
    ) -> HarnessResult:
        res.steps = steps
        res.llm_calls = llm_calls
        res.search_calls = state.search_calls
        res.papers = state.pool
        # find's deliverable is a ranked shortlist, not an insertion-ordered
        # pool: bench seed 42 put 3/5 golds IN the pool and 0/5 in its top-5
        # (survey mode keeps insertion order — the pipeline ranks downstream)
        if state.goal == "find" and len(state.pool) >= 2:
            from citens.agents.rerank import cascade_rank

            res.papers_unranked = list(state.pool)
            # pre-sort by query-match count then citations so the coarse
            # stage starts from a sane order (insertion order is random);
            # the cascade scores the WHOLE pool pointwise (a gold past any
            # fixed window still gets scored), then strong-listwise-orders
            # the top slice — bench seed 42: golds IN the pool 3/5, in the
            # cheap single-pass top-5 0/5
            promising = sorted(
                state.pool,
                key=lambda p: (len(p.matched_queries or []), p.citation_count),
                reverse=True,
            )
            res.papers = await asyncio.to_thread(
                cascade_rank, state.topic, promising, 50, True
            )
            _emit(bus, StepProgress(
                step="search",
                message=f"find 产出按问题相关性重排（{len(res.papers)} 篇）",
                detail=True,
            ))
        res.keywords = state.keywords
        res.query_stats = state.query_stats
        res.summary = summary
        res.skipped = skipped
        res.finish_reason = reason
        _emit(bus, StepProgress(
            step="search",
            message=f"harness 结束（{reason}）: {summary[:80]}",
            detail=True,
        ))
        return res

    while True:
        if llm_calls >= budget.max_llm_calls:
            return await finish("budget:llm_calls")
        if steps >= budget.max_steps:
            return await finish("budget:steps")

        # to_thread: chat_tool_call is a sync SDK call (10-30s on reasoning
        # models) — running it inline would block this event loop for the
        # whole orchestrator turn
        msg = await asyncio.to_thread(llm.chat_tool_call, messages, TOOL_SCHEMAS)
        llm_calls += 1
        messages.append(_assistant_message(msg))

        if not getattr(msg, "tool_calls", None):
            no_tool_strikes += 1
            if no_tool_strikes >= 2:
                return await finish("no_tool_use", summary=msg.content or "")
            messages.append({
                "role": "user",
                "content": (
                    "Use a tool (pool_report / anchors / search / snowball / "
                    "pivot / read_paper / done)."
                ),
            })
            continue
        no_tool_strikes = 0

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            steps += 1
            if name == "done":
                # done-gate: the first done without an anchors check is
                # bounced once — "enough papers" must be falsified against
                # the field's core, not felt. A second done is honored
                # (the model may legitimately judge anchors unreachable).
                if not state.anchors_checked and not state.done_refused:
                    state.done_refused = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            "NOT DONE YET: run the anchors tool first — core "
                            "coverage must be checked before completion. Then "
                            "search any MISSING field-defining titles, or "
                            "justify them in done.skipped."
                        ),
                    })
                    continue
                # find mode: the summary names specific papers — bench seed 42
                # produced confident done calls whose named papers were NOT
                # in the pool (pool_hit=0). A named paper that matches nothing
                # bounces once with instructions to actually retrieve it.
                if state.goal == "find":
                    from citens.eval.litsearch import _title_tokens

                    claimed = str(args.get("summary", ""))
                    pool_toks = [_title_tokens(p.title) for p in state.pool if p.title]
                    unmatched = []
                    for chunk in _find_claimed_titles(claimed):
                        toks = _title_tokens(chunk)
                        if len(toks) < 3:
                            continue
                        # >=half the informative tokens overlapping a pool
                        # title counts as present; one shared generic word
                        # ("paper") must not (measured false-negative)
                        best = max(
                            (len(toks & pt) / len(toks) for pt in pool_toks),
                            default=0.0,
                        )
                        if best < 0.5:
                            unmatched.append(chunk)
                    if unmatched and not state.done_refused:
                        state.done_refused = True
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": (
                                "NOT DONE YET: these papers you named are NOT "
                                "in the pool: " + "; ".join(unmatched[:3]) + ". "
                                "Search for them (exact-title or pivot-mined "
                                "queries), or remove them from the summary."
                            ),
                        })
                        continue
                return await finish(
                    "done",
                    summary=str(args.get("summary", "")),
                    skipped=str(args.get("skipped", "")),
                )
            if steps > budget.max_steps:
                return await finish("budget:steps")
            act_desc = name
            if name == "search":
                act_desc = "search: " + ", ".join(args.get("queries", []))[:120]
            _emit(bus, StepProgress(step="search", message=f"🤖 {act_desc}", detail=True))
            result = await dispatch(name, args, state)
            _emit(bus, StepProgress(
                step="search", message=result.splitlines()[0][:140], detail=True
            ))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:4000],
            })
            if name == "search":
                # refusals (duplicate / budget / errors) are not searches:
                # counting them as empty would fake a saturation stop
                if result.startswith(("DUPLICATE", "ERROR", "SEARCH BUDGET")):
                    continue
                # survey-mode economics: a big pool IS progress there. But in
                # find mode the prompt says pool size is NOT success — cutting
                # a targeted dig at a pool count contradicts the goal (bench
                # seed 42: a find miss ended budget:pool at 169 papers while
                # still reformulating)
                if len(state.pool) >= budget.max_pool and state.goal != "find":
                    return await finish("budget:pool")
                new_n = 0
                first_line = result.splitlines()[0]
                if "new papers: " in first_line:
                    try:
                        new_n = int(first_line.split("new papers: ")[1].split(" ")[0])
                    except (IndexError, ValueError):
                        new_n = 0
                empty_search_streak = empty_search_streak + 1 if new_n == 0 else 0
                if empty_search_streak >= 3:
                    return await finish(
                        "saturation",
                        summary="3 consecutive searches brought nothing new",
                    )
