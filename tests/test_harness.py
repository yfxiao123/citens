"""Agentic retrieval harness (Phase 1): tools + loop, fully mocked.

The LLM boundary is a scripted fake model emitting tool_calls; search_round
is mocked too — these tests pin the LOOP MECHANICS (budget, anti-loop,
saturation, result assembly), not the model's taste.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import citens.harness.tools as tools_mod
from citens.agents.planner import QueryPlan
from citens.harness import HarnessBudget, run_retrieval_harness
from citens.harness.tools import HarnessState, tool_pool_report, tool_search
from citens.models import Paper


def _state(pool=None, **kw):
    return HarnessState(
        topic="generative recommendation",
        plan=QueryPlan(
            queries=["generative recommendation"],
            concepts=[
                {"term": "generative recommendation", "synonyms": ["GenRec"]},
                {"term": "sequential recommendation", "synonyms": ["session-based"]},
            ],
        ),
        pool=pool or [],
        keywords=["generative recommendation"],
        query_stats={"generative recommendation": 10},
        facets=[{"name": "methods", "queries": ["diffusion recommender"]}],
        **kw,
    )


def _paper(title, doi=None, cited=0):
    return Paper(title=title, authors=["A"], year=2023, doi=doi, citation_count=cited)


# --- tools --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_search_merges_and_reports(monkeypatch):
    state = _state(pool=[_paper("Base Paper")])

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return (
            [_paper("New Paper 1"), _paper("Base Paper")],
            {"arXiv": "ok"},
            {queries[0]: 2, queries[1]: 0},
        )

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    result = await tool_search(state, ["genrec", "denoising recommender"])
    assert "new papers: 1" in result  # duplicate merged away
    assert len(state.pool) == 2
    assert "genrec" in state.keywords
    # zero-hit feedback carries the untried synonym hint
    assert "ZERO-HIT" in result and "denoising recommender" in result


@pytest.mark.asyncio
async def test_tool_search_rejects_duplicates_and_budget(monkeypatch):
    state = _state()
    state.searched_keys.add(frozenset({"genrec"}))
    out = await tool_search(state, ["GenRec"])  # case-insensitive set match
    assert out.startswith("DUPLICATE")

    state2 = _state(max_search_calls=0)
    out2 = await tool_search(state2, ["fresh query"])
    assert out2.startswith("SEARCH BUDGET EXHAUSTED")


@pytest.mark.asyncio
async def test_tool_pool_report_signals(monkeypatch):
    state = _state(pool=[_paper("Diffusion Recommender Model")])
    state.query_stats = {"generative recommendation": 10, "bad phrase": 0}
    report = await tool_pool_report(state)
    assert "pool: 1 papers" in report
    assert "THIN facets" in report  # methods facet has <3 papers
    assert "bad phrase" in report  # zero-hit surfaced


# --- loop mechanics -------------------------------------------------------------


def _tool_call(name, args, call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name, arguments=json.dumps(args)
        ),
    )


def _msg(tool_calls=None, content=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _script_model(script, captured=None):
    """Fake chat_tool_call that pops scripted messages in order."""
    queue = list(script)

    def fake(messages, tool_schemas, **kw):
        if captured is not None:
            captured.append([m.get("content", "") for m in messages])
        if not queue:
            return _msg(content="stop")
        return queue.pop(0)

    return fake


@pytest.mark.asyncio
async def test_loop_runs_script_and_finishes_done(monkeypatch):
    state = _state(pool=[_paper("Base")])
    captured = []

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([_paper("From Agent Search")], {}, {queries[0]: 3})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    script = [
        _msg(tool_calls=[_tool_call("pool_report", {})]),
        _msg(tool_calls=[_tool_call("search", {"queries": ["genrec diffusion"]})]),
        _msg(tool_calls=[_tool_call("anchors", {})]),
        _msg(tool_calls=[_tool_call(
            "done", {"summary": "covered", "skipped": "session-based thin"}
        )]),
    ]
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script, captured)
    )
    res = await run_retrieval_harness(state, bus=None)
    assert res.finish_reason == "done"
    assert res.summary == "covered"
    assert res.skipped == "session-based thin"
    assert res.llm_calls == 4 and res.steps == 4
    assert any(p.title == "From Agent Search" for p in res.papers)
    # tool results really reached the model as tool messages
    flat = "\n".join(captured[-1])
    assert "new papers: 1" in flat


@pytest.mark.asyncio
async def test_loop_budget_and_saturation(monkeypatch):
    # model keeps searching novel queries -> search budget stops it
    state = _state()

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([_paper(f"Paper {queries[0]}")], {}, {queries[0]: 1})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    counter = {"n": 0}

    def endless(messages, tool_schemas, **kw):
        counter["n"] += 1
        return _msg(tool_calls=[_tool_call("search", {"queries": [f"q{counter['n']}"]})])

    monkeypatch.setattr("citens.harness.loop.llm.chat_tool_call", endless)
    res = await run_retrieval_harness(
        state, bus=None, budget=HarnessBudget(max_steps=10, max_llm_calls=10, max_search_calls=3)
    )
    assert res.search_calls == 3  # tool refused beyond budget
    # loop kept running perceive turns but eventually hit step/llm budget
    assert res.finish_reason.startswith("budget:")


@pytest.mark.asyncio
async def test_loop_saturation_three_empty_searches(monkeypatch):
    state = _state()

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([], {}, {queries[0]: 0})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    counter = {"n": 0}

    def endless(messages, tool_schemas, **kw):
        counter["n"] += 1
        return _msg(tool_calls=[_tool_call("search", {"queries": [f"q{counter['n']}"]})])

    monkeypatch.setattr("citens.harness.loop.llm.chat_tool_call", endless)
    res = await run_retrieval_harness(state, bus=None)
    assert res.finish_reason == "saturation"
    assert res.search_calls == 3  # stopped right at the third empty search


@pytest.mark.asyncio
async def test_loop_two_prose_turns_force_finish(monkeypatch):
    state = _state()
    script = [
        _msg(content="let me think..."),
        _msg(content="still thinking"),
    ]
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script)
    )
    res = await run_retrieval_harness(state, bus=None)
    assert res.finish_reason == "no_tool_use"


# --- anchors: external core-coverage check --------------------------------------


def _anchor(title, doi, cited):
    import re as _re

    return {
        "title": title,
        "doi": doi,
        "citations": cited,
        "tokens": {t for t in _re.split(r"[^a-z0-9]+", title.lower()) if len(t) > 2},
    }


@pytest.mark.asyncio
async def test_tool_anchors_reports_missing(monkeypatch):
    state = _state(pool=[_paper("P", doi="10.1/kept")])

    def fake_anchor_works(queries, per_query=8):
        assert queries == ["generative recommendation", "sequential recommendation"]
        return [
            _anchor("Kept Classic Work Here", "10.1/kept", 900),
            _anchor("Missing Field Defining Paper", "10.1/miss", 1200),
        ]

    monkeypatch.setattr(tools_mod, "_anchor_works", fake_anchor_works)
    out = await tools_mod.tool_anchors(state)
    assert "CORE COVERAGE: 1/2" in out
    assert "MISSING [1200c] Missing Field Defining Paper" in out
    assert state.anchors_checked is True


@pytest.mark.asyncio
async def test_tool_anchors_degrades_when_unreachable(monkeypatch):
    state = _state()

    def boom(queries, per_query=8):
        raise RuntimeError("openalex down")

    monkeypatch.setattr(tools_mod, "_anchor_works", boom)
    out = await tools_mod.tool_anchors(state)
    assert "anchors unavailable" in out
    assert state.anchors_checked is True  # must not gate done forever


@pytest.mark.asyncio
async def test_done_gate_bounces_once_then_honors(monkeypatch):
    state = _state(pool=[_paper("Base")])
    captured = []

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([_paper("Found Paper")], {}, {queries[0]: 3})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)

    def fake_anchor_works(queries, per_query=8):
        return [_anchor("Base Title Words Here", "10.1/base", 10)]

    monkeypatch.setattr(tools_mod, "_anchor_works", fake_anchor_works)
    script = [
        _msg(tool_calls=[_tool_call("done", {"summary": "enough"})]),  # bounced
        _msg(tool_calls=[_tool_call("anchors", {})]),
        _msg(tool_calls=[_tool_call("done", {"summary": "done for real"})]),
    ]
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script, captured)
    )
    res = await run_retrieval_harness(state, bus=None)
    assert res.finish_reason == "done"
    assert res.summary == "done for real"
    # the bounce reason reached the model as tool feedback
    assert any("NOT DONE YET" in c for c in captured[1])


def test_find_goal_switches_prompt():
    from citens.harness.loop import _system_prompt

    assert "TARGETED MODE" not in _system_prompt(_state())
    assert "TARGETED MODE" in _system_prompt(_state(goal="find"))


@pytest.mark.asyncio
async def test_tool_pivot_mines_and_searches(monkeypatch):
    from citens.harness.tools import tool_pivot

    state = _state(pool=[_paper("Neighbor Paper")])
    monkeypatch.setattr(
        "citens.agents.pivot.pivot_from_abstracts",
        lambda q, papers, k=4: ["verbatim task name"],
    )

    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([_paper("Pivoted Paper")], {}, {queries[0]: 2})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    out = await tool_pivot(state)
    assert out.startswith("PIVOT mined queries")
    assert "verbatim task name" in out
    assert any(p.title == "Pivoted Paper" for p in state.pool)
    assert "verbatim task name" in state.keywords


@pytest.mark.asyncio
async def test_tool_pivot_skips_when_all_duplicate(monkeypatch):
    from citens.harness.tools import tool_pivot

    state = _state(pool=[_paper("Neighbor Paper")])
    state.keywords.append("already tried")
    monkeypatch.setattr(
        "citens.agents.pivot.pivot_from_abstracts",
        lambda q, papers, k=4: ["already tried"],
    )
    called: list = []

    async def fake_round(queries, max_results, sources=None, constraints=None):
        called.append(queries)
        return ([], {}, {})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    out = await tool_pivot(state)
    assert out.startswith("PIVOT: no new queries")
    assert not called  # no search burned on a duplicate formulation


@pytest.mark.asyncio
async def test_find_goal_ignores_pool_cap_but_survey_stops(monkeypatch):
    # bench seed 42: a find-mode miss ended budget:pool at 169 papers while
    # still reformulating — pool size contradicts the find goal's own prompt
    async def fake_round(queries, max_results, sources=None, constraints=None):
        return ([_paper("From Agent Search")], {}, {queries[0]: 3})

    monkeypatch.setattr(tools_mod, "search_round", fake_round)
    monkeypatch.setattr(
        tools_mod, "_anchor_works", lambda queries, per_query=8: []
    )
    # identity ranking: this test pins budget mechanics, not output order
    # (and must not make a real LLM call from the find-mode finish path)
    import citens.agents.rerank as rerank_mod

    monkeypatch.setattr(
        rerank_mod, "cascade_rank", lambda q, papers, coarse_keep=50, strong=False: list(papers)
    )
    budget = HarnessBudget(max_pool=2, max_steps=8, max_llm_calls=8)

    find_state = _state(pool=[_paper("Base")], goal="find")
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call",
        _script_model([
            _msg(tool_calls=[_tool_call("search", {"queries": ["fresh q"]})]),
            _msg(tool_calls=[_tool_call("anchors", {})]),
            _msg(tool_calls=[_tool_call("done", {"summary": "found it"})]),
        ]),
    )
    res = await run_retrieval_harness(find_state, bus=None, budget=budget)
    assert res.finish_reason == "done"  # the search past the cap happened
    assert len(res.papers) == 2  # Base + the search result (pool == max_pool)
    assert len(res.papers_unranked) == 2  # pre-ranking copy kept for audit

    survey_state = _state(pool=[_paper("Base")])
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call",
        _script_model([
            _msg(tool_calls=[_tool_call("search", {"queries": ["fresh q"]})]),
        ]),
    )
    res2 = await run_retrieval_harness(survey_state, bus=None, budget=budget)
    assert res2.finish_reason == "budget:pool"  # survey economics unchanged


@pytest.mark.asyncio
async def test_find_goal_ranks_output_survey_keeps_insertion_order(monkeypatch):
    """Find's deliverable is a ranked shortlist (bench seed 42: golds in the
    pool 3/5, in its top-5 0/5); survey keeps insertion order — the pipeline
    ranks downstream."""
    import citens.agents.rerank as rerank_mod

    calls = {}

    def fake_rank(question, papers, coarse_keep=50, strong=False):
        calls["question"] = question
        calls["n"] = len(papers)
        calls["strong"] = strong
        return list(reversed(papers))

    monkeypatch.setattr(rerank_mod, "cascade_rank", fake_rank)
    monkeypatch.setattr(
        tools_mod, "_anchor_works", lambda queries, per_query=8: []
    )
    pool = [_paper(f"P{i}") for i in range(4)]
    script = [
        _msg(tool_calls=[_tool_call("anchors", {})]),
        _msg(tool_calls=[_tool_call("done", {"summary": "found"})]),
    ]

    find_state = _state(pool=list(pool), goal="find")
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script)
    )
    res = await run_retrieval_harness(find_state, bus=None)
    assert [p.title for p in res.papers] == ["P3", "P2", "P1", "P0"]
    assert [p.title for p in res.papers_unranked] == ["P0", "P1", "P2", "P3"]
    assert calls["question"] == find_state.topic and calls["n"] == 4
    assert calls["strong"] is True  # the deliverable rank is strong-tier

    survey_state = _state(pool=list(pool))
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script)
    )
    res2 = await run_retrieval_harness(survey_state, bus=None)
    assert [p.title for p in res2.papers] == ["P0", "P1", "P2", "P3"]
    assert res2.papers_unranked == []


@pytest.mark.asyncio
async def test_find_done_bounces_when_named_paper_not_in_pool(monkeypatch):
    """Bench seed 42: confident done calls whose named papers were NOT in
    the pool (pool_hit=0). The gate verifies named titles against the pool
    and bounces once with retrieval instructions."""
    monkeypatch.setattr(
        tools_mod, "_anchor_works", lambda queries, per_query=8: []
    )
    script = [
        _msg(tool_calls=[_tool_call("anchors", {})]),
        _msg(tool_calls=[_tool_call(
            "done",
            {"summary": 'Found "Totally Unrelated Paper About Quantum Widgets" as the answer'},
        )]),
        _msg(tool_calls=[_tool_call(
            "done",
            {"summary": 'Found "Base Paper That We Truly Have"'},
        )]),
    ]
    monkeypatch.setattr(
        "citens.harness.loop.llm.chat_tool_call", _script_model(script)
    )
    state = _state(pool=[_paper("Base Paper That We Truly Have")], goal="find")
    res = await run_retrieval_harness(state, bus=None)
    assert res.finish_reason == "done"
    # the fake model saw the bounce message between the two dones
    assert res.steps == 3
