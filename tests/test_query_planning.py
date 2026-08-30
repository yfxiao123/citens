"""Concept-block query planning (v3 planner) + the facet/synonym calibration
helpers. Pure-logic tests: the LLM boundary is mocked everywhere."""

from __future__ import annotations

from citens.agents.planner import (
    QueryPlan,
    assemble_queries,
    generate_keywords,
    plan_queries,
    refine_queries,
    synonym_fallback_queries,
)
from citens.orchestration.support import facet_coverage_report, thin_facet_queries

_CONCEPTS = [
    {"term": "limit order book", "synonyms": ["LOB", "order book dynamics"]},
    {"term": "market making", "synonyms": ["dealer problem", "liquidity provision"]},
    {"term": "price impact", "synonyms": ["price formation"]},
    {"term": "order book stylized facts", "synonyms": []},
]


# --- deterministic assembly --------------------------------------------------


def test_assemble_queries_coverage_then_combos():
    qs = assemble_queries(_CONCEPTS)
    # one coverage query per concept, in order...
    assert qs[:4] == [
        "limit order book",
        "market making",
        "price impact",
        "order book stylized facts",
    ]
    # ...then precision combos of the first (most central) three concepts
    assert "limit order book market making" in qs
    assert "limit order book price impact" in qs
    assert "market making price impact" in qs
    # the 4th concept never enters a combo
    assert all("stylized" not in q or q == "order book stylized facts" for q in qs)


def test_assemble_queries_dedupes_and_caps():
    concepts = [
        {"term": "alpha", "synonyms": []},
        {"term": "Alpha", "synonyms": []},  # case duplicate
        {"term": "beta", "synonyms": []},
        {"term": "gamma", "synonyms": []},
    ]
    qs = assemble_queries(concepts, max_queries=4)
    assert len(qs) == 4
    assert len({q.lower() for q in qs}) == 4


def test_assemble_queries_skips_subsumed_combos():
    # "alpha beta" adds nothing when "alpha beta" is already a coverage term
    concepts = [
        {"term": "alpha beta", "synonyms": []},
        {"term": "alpha", "synonyms": []},
        {"term": "beta", "synonyms": []},
    ]
    qs = assemble_queries(concepts)
    assert "alpha beta" in qs
    assert qs.count("alpha beta") == 1


def test_assemble_queries_anchors_generic_terms():
    # a bare "survey" coverage query returns every discipline's mega-survey
    # (measured: SF-36, 30k citations, in a generative-recs pool) — generic
    # terms get anchored to the central concept, duplicate words collapsing
    concepts = [
        {"term": "generative recommendation", "synonyms": []},
        {"term": "survey", "synonyms": []},
        {"term": "generative models", "synonyms": []},
    ]
    qs = assemble_queries(concepts)
    assert "generative recommendation survey" in qs
    assert "survey" not in qs
    assert "generative recommendation models" in qs  # duplicate word collapsed
    assert "generative models" not in qs


# --- the LLM boundary --------------------------------------------------------


def test_plan_queries_builds_plan_from_concepts(monkeypatch):
    monkeypatch.setattr(
        "citens.agents.planner.chat_json",
        lambda *a, **k: {"concepts": _CONCEPTS, "reasoning": ""},
    )
    plan = plan_queries("限价指令簿市场微观结构")
    assert plan.queries[:4] == [
        "limit order book",
        "market making",
        "price impact",
        "order book stylized facts",
    ]
    assert plan.concepts == _CONCEPTS
    # synonyms ride along, untried, for the calibration wave
    assert plan.synonyms_for("limit order book") == ["LOB", "order book dynamics"]


def test_plan_queries_falls_back_on_malformed_output(monkeypatch):
    monkeypatch.setattr("citens.agents.planner.chat_json", lambda *a, **k: {"queries": ["x"]})
    plan = plan_queries("some topic")
    assert plan.queries == ["some topic"]
    assert plan.concepts[0]["term"] == "some topic"


def test_plan_queries_falls_back_on_exception(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr("citens.agents.planner.chat_json", _boom)
    plan = plan_queries("some topic")
    assert plan.queries == ["some topic"]


def test_generate_keywords_wrapper_still_flat(monkeypatch):
    # collect.py + older callers depend on the flat list contract
    plan = QueryPlan(queries=["a", "b"], concepts=[])
    monkeypatch.setattr(
        "citens.agents.planner.plan_queries", lambda *a, **k: plan
    )
    assert generate_keywords("t") == ["a", "b"]


# --- PRESS-style calibration ---------------------------------------------------


def test_synonym_fallback_swaps_untried_synonyms():
    plan = QueryPlan(queries=["limit order book"], concepts=_CONCEPTS)
    out = synonym_fallback_queries(
        plan,
        zero_hit_queries=["limit order book", "price impact"],
        already_searched=["limit order book", "market making"],
    )
    assert "LOB" in out
    assert "order book dynamics" in out
    assert "price formation" in out  # from the price-impact concept


def test_synonym_fallback_skips_searched_and_caps():
    plan = QueryPlan(queries=["a"], concepts=[{"term": "a", "synonyms": ["s1", "s2", "s3"]}])
    out = synonym_fallback_queries(
        plan, ["a"], already_searched=["a", "s1"], cap=2
    )
    assert out == ["s2", "s3"]
    assert synonym_fallback_queries(plan, ["a"], ["a", "s1", "s2", "s3"]) == []


def test_synonym_fallback_no_concept_match():
    plan = QueryPlan(queries=["a"], concepts=_CONCEPTS)
    assert synonym_fallback_queries(plan, ["unrelated query"], ["a"]) == []


# --- refine sees zero-hit feedback ---------------------------------------------


def test_refine_queries_includes_zero_hit_block(monkeypatch):
    captured = {}

    def _fake_chat_json(system, user, **k):
        captured["user"] = user
        return {"queries": ["new query"]}

    monkeypatch.setattr("citens.agents.planner.chat_json", _fake_chat_json)
    out = refine_queries(
        "topic",
        ["q1", "q2"],
        found_titles=["Some Paper"],
        known_gaps=[],
        zero_hit_queries=["q1"],
    )
    assert out == ["new query"]
    assert "零命中查询" in captured["user"]
    assert "- q1" in captured["user"]


# --- thin-facet second wave -----------------------------------------------------


class _P:
    def __init__(self, title: str, abstract: str = ""):
        self.title = title
        self.abstract = abstract


_FACETS = [
    {"name": "methods", "queries": ["queueing model order book", "hawkes process orders"]},
    {"name": "applications", "queries": ["market making strategy"]},
]


def test_thin_facet_queries_targets_thin_facets_only():
    # methods facet well covered, applications facet thin -> only its queries
    papers = [_P("queueing model for the order book"), _P("hawkes process order flow"), _P("order book queueing theory")]
    report = facet_coverage_report(_FACETS, papers)
    out = thin_facet_queries(_FACETS, report, already_searched=["queueing model order book"])
    assert out == ["market making strategy"]


def test_thin_facet_queries_dedupes_and_caps():
    papers = []  # everything thin
    report = facet_coverage_report(_FACETS, papers)
    out = thin_facet_queries(_FACETS, report, already_searched=[], cap=2)
    assert len(out) == 2
    # already-searched queries never repeat
    out2 = thin_facet_queries(
        _FACETS, report, already_searched=["queueing model order book", "hawkes process orders", "market making strategy"]
    )
    assert out2 == []


# --- retrieval provenance + direction yield -----------------------------------


def test_deduplicate_unions_matched_queries():
    from citens.models import Paper
    from citens.search.base import deduplicate

    a = Paper(title="Deep LOB", authors=["Wei Zhang"], year=2020,
              citation_count=10, matched_queries=["order book"])
    b = Paper(title="Deep LOB", authors=["Wei Zhang"], year=2020,
              citation_count=20, matched_queries=["market making"])
    out = deduplicate([a, b])
    assert len(out) == 1
    assert set(out[0].matched_queries) == {"order book", "market making"}

    # fuzzy preprint/published merge keeps provenance too
    pre = Paper(title="DeepLOB Order Book Networks", authors=["Wei Zhang"],
                year=2019, citation_count=5, matched_queries=["order book"])
    pub = Paper(title="Order Book DeepLOB Networks", authors=["Wei Zhang", "Li Na"],
                year=2020, citation_count=50,
                doi="10.1/x", matched_queries=["price impact"])
    out2 = deduplicate([pre, pub])
    assert len(out2) == 1
    assert set(out2[0].matched_queries) == {"order book", "price impact"}


def test_query_yield_report_per_concept():
    from citens.models import Paper
    from citens.orchestration.support import query_yield_report

    concepts = [{"term": "alpha beta"}, {"term": "gamma"}]
    junk = Paper(title="Junk Paper", authors=["A"], year=2020,
                 matched_queries=["gamma"])
    good = Paper(title="Good Paper", authors=["B"], year=2021,
                 matched_queries=["alpha beta"])
    combo = Paper(title="Combo Paper", authors=["C"], year=2022,
                  matched_queries=["alpha beta gamma"])  # credits BOTH concepts
    rows = query_yield_report(concepts, [junk, good, combo], [good])
    by_term = {r["concept"]: r for r in rows}
    assert by_term["alpha beta"] == {"concept": "alpha beta", "hits": 2, "kept": 1}
    assert by_term["gamma"] == {"concept": "gamma", "hits": 2, "kept": 0}
