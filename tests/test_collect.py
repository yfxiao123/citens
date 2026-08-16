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
            return {"results": [{"display_name": "A Author",
                                 "works_count": 80,
                                 "summary_stats": {"h_index": 25}}]}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, **k):
            return _Resp()

    monkeypatch.setattr(collect_mod.httpx, "Client", lambda **k: _Client())
    monkeypatch.setattr(collect_mod.time, "sleep", lambda *_: None)
    papers = [_p(f"P{i}", citation_count=i) for i in range(5)]
    n = collect_mod._enrich_author_engagement(papers, top_n=2)
    assert n == 2
    # only the two most-cited got the signal
    enriched = [p for p in papers if p.first_author_works > 0]
    assert {p.title for p in enriched} == {"P3", "P4"}
    assert enriched[0].first_author_h_index == 25


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

    monkeypatch.setattr(
        planner, "chat_json",
        lambda s, u, **k: {"queries": ["order book model"]},
    )
    # generate_seed_papers uses the same fake: papers=[] terms=[]
    queries = collect_mod.build_queries("订单簿建模")
    assert "order book model" in queries
    assert any("survey" in q for q in queries)
    assert any("review" in q for q in queries)
    assert len(queries) == len(set(q.lower() for q in queries))  # deduped
