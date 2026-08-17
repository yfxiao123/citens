"""Tests for hybrid pool recall (BM25+vector RRF) and domain profiles."""

from __future__ import annotations

from citens import collect as collect_mod
from citens.models import Paper, ScoredPaper
from citens.profiles import load_profile, merge_profile_terms
from citens.ranking import rank_papers


def _p(title, **kw):
    defaults = dict(authors=["A Author"], year=2020, abstract="abs", citation_count=1)
    defaults.update(kw)
    return Paper(title=title, **defaults)


# --- embed index + RRF recall ---------------------------------------------


def test_embed_pool_noop_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    monkeypatch.setattr(collect_mod.settings, "embedding_model", "")
    collect_mod.append_pool("t", [_p("Order Book Dynamics")])
    assert collect_mod.embed_pool("t") == 0  # skipped, no crash


def test_recall_fuses_vector_and_bm25(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    monkeypatch.setattr(collect_mod.settings, "embedding_model", "test-emb")
    pool = [
        _p("queue position dynamics in limit order books"),   # lexical #1, vec #2
        _p("deep learning for image classification"),          # noise both
        _p("priority mechanisms for order priority trading"),  # lexical #2, vec #1
    ]
    collect_mod.append_pool("t", pool)

    doc_vecs = {"queue position dynamics in limit order books. abs": [0.9],
                "deep learning for image classification. abs": [0.1],
                "priority mechanisms for order priority trading. abs": [1.0]}

    def fake_embed(texts):
        # query call: embed close to the semantic paper
        if len(texts) == 1 and texts[0] not in doc_vecs:
            return [[0.95]]
        return [doc_vecs.get(t, [0.1]) for t in texts]

    monkeypatch.setattr("citens.grounding.retrieval.embed_texts", fake_embed)
    assert collect_mod.embed_pool("t") == 3
    picked = collect_mod.recall_from_pool("t", ["limit order book queue"], k=2)
    titles = [p.title for p in picked]
    # each channel's favorite survives; the noise paper doesn't
    assert "queue position dynamics in limit order books" in titles
    assert "priority mechanisms for order priority trading" in titles
    assert "deep learning for image classification" not in titles


def test_recall_bm25_only_when_no_index(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    collect_mod.append_pool("t", [_p("alpha factor model"), _p("gardening tips")])
    picked = collect_mod.recall_from_pool("t", ["alpha factor"], k=1)
    assert [p.title for p in picked] == ["alpha factor model"]


# --- finance profile ----------------------------------------------------------


def test_load_finance_profile():
    prof = load_profile("finance")
    assert prof is not None
    assert "market microstructure" in prof.domain_terms
    assert "Journal of Finance" in prof.venue_whitelist
    assert load_profile("no-such-profile") is None


def test_merge_profile_terms_dedupes():
    prof = load_profile("finance")
    merged = merge_profile_terms(["order flow imbalance", "custom query"], prof)
    assert merged.count("order flow imbalance") == 1
    assert "custom query" in merged and "adverse selection" in merged


def test_venue_whitelist_boosts_rank():
    prof = load_profile("finance")
    boost = prof.venue_boost_set()
    a = ScoredPaper(title="A", authors=["X"], year=2020, abstract="a",
                    relevance_score=3, citation_count=10,
                    venue="Review of Financial Studies")
    b = ScoredPaper(title="B", authors=["Y"], year=2020, abstract="a",
                    relevance_score=3, citation_count=10,
                    venue="Journal of Random Results")
    out = rank_papers([b, a], venue_boost=boost)
    assert out[0].title == "A"  # flagship venue wins the tie
    out2 = rank_papers([b, a])  # without profile: neutral
    assert out2[0].rank_score < a.relevance_score  # sanity: scores exist


def test_build_queries_merges_profile_terms(monkeypatch):
    import citens.agents.planner as planner

    monkeypatch.setattr(
        planner, "chat_json",
        lambda s, u, **k: {"queries": ["limit order book"]},
    )
    queries, broad = collect_mod.build_queries("订单簿建模", profile_name="finance")
    assert "market microstructure" in queries        # profile term added
    assert "market microstructure" in broad           # ...as a constrained concept
    assert "limit order book" in queries and "limit order book" not in broad
