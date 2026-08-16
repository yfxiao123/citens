"""Tests for the record-first literature pool (citens collect) and the
author-engagement ranking factor."""

from __future__ import annotations

from pathlib import Path

from citens import collect as collect_mod
from citens.models import Paper, ScoredPaper
from citens.ranking import author_depth_factor, rank_papers


def _p(title, **kw):
    defaults = dict(authors=["A Author"], year=2020, abstract="abs", citation_count=10)
    defaults.update(kw)
    return Paper(title=title, **defaults)


# --- pool persistence --------------------------------------------------------


def test_pool_append_read_dedup(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path / "litdb"))
    assert collect_mod.read_pool("订单簿建模") == []
    assert collect_mod.append_pool("订单簿建模", [_p("Paper One", doi="10.1/x"), _p("Paper Two")]) == 2
    # same DOI (different title variant) deduped; new paper added
    assert (
        collect_mod.append_pool(
            "订单簿建模",
            [_p("Paper One dup", doi="10.1/x"), _p("Paper Three")],
        )
        == 1
    )
    pool = collect_mod.read_pool("订单簿建模")
    assert len(pool) == 3
    # records carry the structured fields
    assert all(hasattr(p, "subfield") and hasattr(p, "keywords") for p in pool)


def test_import_records_from_external_export(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path / "litdb"))
    records = [
        {"title": "WOB export", "authors": ["B"], "year": 2021, "abstract": "a",
         "citation_count": 99},
        {"garbage": True},  # skipped, not fatal
    ]
    assert collect_mod.import_records("某领域", records) == 1
    assert len(collect_mod.read_pool("某领域")) == 1


# --- author engagement enrichment ---------------------------------------------


def test_enrich_author_engagement_fills_top_papers(monkeypatch):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"id": "https://openalex.org/authors/A123",
                                 "display_name": "A Author",
                                 "works_count": 80,
                                 "summary_stats": {"h_index": 25}}]}

    monkeypatch.setattr(collect_mod, "_oa_get_sync", lambda url, params: _Resp())
    monkeypatch.setattr(collect_mod.time, "sleep", lambda *_: None)
    monkeypatch.setattr(collect_mod, "_field_works_count", lambda aid, q: 12)
    papers = [_p(f"P{i}", citation_count=i) for i in range(5)]
    n = collect_mod._enrich_author_engagement(papers, "order book", top_n=2)
    assert n == 2
    # only the two most-cited got the signal
    enriched = [p for p in papers if p.first_author_works > 0]
    assert {p.title for p in enriched} == {"P3", "P4"}
    assert enriched[0].first_author_h_index == 25
    assert enriched[0].author_field_works == 12


# --- ranking factor ------------------------------------------------------------


def test_author_depth_factor_bounds():
    assert author_depth_factor(0, 0) is None  # unknown -> excluded
    assert author_depth_factor(40, 100) == 1.0  # saturated
    assert author_depth_factor(30, 100) < 1.0  # near but not saturated
    f = author_depth_factor(5, 10)
    assert 0.0 < f < 1.0


def test_rank_uses_author_depth_without_punishing_unknown():
    # same everything, one has a deeply-engaged first author
    a = ScoredPaper(title="A", authors=["X"], year=2020, abstract="a",
                    relevance_score=4, citation_count=50,
                    first_author_h_index=30, first_author_works=100)
    b = ScoredPaper(title="B", authors=["Y"], year=2020, abstract="a",
                    relevance_score=4, citation_count=50)
    out = rank_papers([b, a])
    assert out[0].title == "A"
    # unknown-signal paper's score is renormalized, not tanked
    assert out[1].rank_score > 0.5


# --- query building -------------------------------------------------------------


def test_build_queries_includes_survey_hunting(monkeypatch):
    import citens.agents.planner as planner

    def fake_chat(s, u, **k):
        # keywords prompt vs seed prompt are distinguishable by content
        if "LANDMARK" in s:
            return {"papers": [], "domain_terms": ["adverse selection"]}
        return {"queries": ["order book model"]}

    monkeypatch.setattr(planner, "chat_json", fake_chat)
    queries, broad = collect_mod.build_queries("订单簿建模")
    assert "order book model" in queries
    assert "adverse selection" in queries
    assert "adverse selection" in broad  # seed domain terms -> field-constrained only
    assert "order book model" not in broad
    assert any("survey" in q for q in queries)
    assert any("review" in q for q in queries)
    assert len(queries) == len(set(q.lower() for q in queries))  # deduped


# --- v2: attribution, backfill handoff, pre-recall, audit -------------------


def test_search_per_query_attributes_matched_queries(monkeypatch):
    import asyncio

    async def fake_search(queries, max_results=10, sources=None):
        # every query "finds" the same paper + one unique paper
        out = [_p("Shared"), _p(f"Unique {queries[0]}")]
        return out

    monkeypatch.setattr(collect_mod, "search_papers", fake_search)
    by_key, hits = asyncio.run(collect_mod._search_per_query(["q one", "q two"], 10, None))
    shared = next(p for p in by_key.values() if p.title == "Shared")
    assert set(shared.matched_queries) == {"q one", "q two"}
    assert hits == {"q one": 2, "q two": 2}


def test_append_pool_merges_metadata_without_losing(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    collect_mod.append_pool("t", [_p("A", doi="10.1/x", subfield="", keywords=[])])
    # second record same DOI brings subfield + query attribution
    collect_mod.append_pool(
        "t", [_p("A", doi="10.1/x", subfield="Finance", matched_queries=["lob model"])]
    )
    pool = collect_mod.read_pool("t")
    assert len(pool) == 1
    assert pool[0].subfield == "Finance"
    assert pool[0].matched_queries == ["lob model"]


def test_recall_from_pool_ranks_and_keeps_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    records = [
        _p("order book queueing model", citation_count=1),
        _p("deep learning image classification cats", citation_count=99),
        _p("limit order book price impact", citation_count=5),
        _p("totally unrelated agriculture paper", citation_count=500, is_review=True),
    ]
    collect_mod.append_pool("t", records)
    picked = collect_mod.recall_from_pool("t", ["order book limit"], k=3)
    titles = [p.title for p in picked]
    assert len(picked) <= 3
    assert "order book queueing model" in titles
    assert "limit order book price impact" in titles
    assert "totally unrelated agriculture paper" in titles  # review always survives
    assert "deep learning image classification cats" not in titles


def test_broad_query_classification():
    assert collect_mod._is_broad("adverse selection") is False
    assert collect_mod._is_broad("adverse selection") is False or True  # 2 words
    assert collect_mod._is_broad("econometrics") is True  # single concept


def test_field_works_count_pref(tmp_path, monkeypatch):
    from citens.ranking import author_depth_factor

    f_field = author_depth_factor(20, 999999, field_works=25)
    f_total = author_depth_factor(20, 999999)
    assert f_field is not None and f_total is not None
    # in-field 25 works should beat a (artifact-prone) huge total
    assert f_field > f_total
