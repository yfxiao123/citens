"""Offline unit tests for the bench/coverage eval modules (no network, no LLM)."""


from citens.eval.coverage import coverage_metrics, parse_bib
from citens.eval.litsearch import (
    _norm_doi,
    fuse_multi_query,
    gold_hits,
    interleave,
    match_gold,
    sample_queries,
)
from citens.models import Paper

# --- litsearch pure helpers --------------------------------------------------


def _p(title: str, doi: str | None = None, url: str = "") -> Paper:
    return Paper(title=title, doi=doi, url=url, source="test")


def test_norm_doi_strips_resolver_prefix():
    assert _norm_doi("https://doi.org/10.1145/3422622") == "10.1145/3422622"
    assert _norm_doi("10.1145/3422622") == "10.1145/3422622"


def test_match_gold_by_doi_arxiv_title():
    gold = {"title": "BitFit: Simple Parameter-efficient Fine-tuning",
            "doi": "10.48550/arxiv.2106.10199", "arxiv": "2106.10199"}
    assert match_gold(_p("Whatever", doi="10.48550/ARXIV.2106.10199"), [gold]) == "doi"
    assert match_gold(
        _p("Other", url="https://arxiv.org/abs/2106.10199"), [gold]
    ) == "arxiv"
    assert match_gold(_p("BitFit Simple Parameter efficient Fine tuning"), [gold]) == "title"
    assert match_gold(_p("Completely unrelated work"), [gold]) is None


def test_gold_hits_recall_at_k():
    gold = {"title": "Paper A", "doi": "10.1/a"}
    papers = [_p("Filler 1"), _p("Paper A", doi="10.1/a"), _p("Filler 2")]
    assert gold_hits(papers, [gold], 1) == 0.0
    assert gold_hits(papers, [gold], 2) == 1.0


def test_sample_queries_stratified_and_seeded():
    rows = [
        {"query": f"q{i}", "query_set": s, "corpusids": [i]}
        for s in ("a", "a", "a", "a", "b")
        for i in range(5)
    ]
    s1 = sample_queries(rows, 10, seed=1)
    s2 = sample_queries(rows, 10, seed=1)
    assert s1 == s2  # deterministic
    assert len(s1) == 10
    from collections import Counter

    counts = Counter(r["query_set"] for r in s1)
    assert counts["a"] == 8 and counts["b"] == 2  # proportional to 4:1


def test_interleave_round_robin_and_dedup():
    a = [_p("A1", doi="10.1/a1"), _p("A2", doi="10.1/a2")]
    b = [_p("A1 dup", doi="10.1/a1"), _p("B1", doi="10.1/b1")]
    merged = interleave({"s": a, "t": b}, ["s", "t"])
    # round-robin: rank-0 of both sources first; the DOI dup is dropped
    assert [p.title for p in merged] == ["A1", "A2", "B1"]


def test_fuse_rrf_promotes_single_cell_excellence():
    # gold ranks #0 in ONE cell; every other cell's head is filler, so gold's
    # 1/60 stands alone at the top (round-robin would bury it ~#27)
    per_q = {
        "q1": {"s2": [_p("Gold One", doi="10.1/g")]
               + [_p(f"Broad {i}", doi=f"10.1/b{i}") for i in range(9)],
               "oa": [_p("Head A", doi="10.1/ha")]
               + [_p(f"Noise A{i}", doi=f"10.1/a{i}") for i in range(9)]},
        "q2": {"s2": [_p("Head B", doi="10.1/hb")]
               + [_p(f"Noise B{i}", doi=f"10.1/c{i}") for i in range(9)],
               "oa": [_p("Head C", doi="10.1/hc")]
               + [_p(f"Noise C{i}", doi=f"10.1/d{i}") for i in range(9)]},
    }
    fused = fuse_multi_query(per_q, ["s2", "oa"])
    assert fused[0].title == "Gold One"


def test_fuse_rrf_sums_across_duplicate_objects():
    # the same work (same DOI, distinct objects) at rank 1 in two cells must
    # outrank a work at rank 0 in one cell only
    per_q = {
        "q1": {"s2": [_p("Solo", doi="10.1/solo"), _p("Twin", doi="10.1/twin")],
               "oa": [_p("Other", doi="10.1/o"), _p("Twin copy", doi="10.1/twin")]},
    }
    fused = fuse_multi_query(per_q, ["s2", "oa"])
    assert fused[0].title == "Twin"


# --- coverage ----------------------------------------------------------------


BIB = """@article{wang20170,
  title = {IRGAN: A Minimax Game},
  year = {2017},
  doi = {10.48550/arxiv.1705.10513},
}

@article{no_doi,
  title = {Some Untitled Work With No DOI},
  year = {2020},
}
"""


def test_parse_bib(tmp_path):
    f = tmp_path / "references.bib"
    f.write_text(BIB, encoding="utf-8")
    entries = parse_bib(tmp_path)
    assert len(entries) == 2
    assert entries[0]["doi"] == "10.48550/arxiv.1705.10513"
    assert entries[1]["doi"] == ""


def test_coverage_metrics_overlap_and_core():
    survey = [
        {"title": f"Survey Ref {i}", "doi": f"10.1/s{i}", "citations": 100 - i}
        for i in range(60)
    ]
    ours = [
        {"title": "Survey Ref 0", "doi": "10.1/s0"},       # core hit (top-50)
        {"title": "Survey Ref 59", "doi": "10.1/s59"},     # outside core (rank 60)
        {"title": "Off-base work", "doi": "10.9/x"},       # not in survey
    ]
    m = coverage_metrics(ours, survey)
    assert m["overlap"] == 2
    assert m["survey_recall"] == round(2 / 60, 4)
    assert m["core50_hit"] == 1  # Ref 0 in top-50; Ref 59 is rank 60 -> outside
    assert m["core50_recall"] == round(1 / 50, 4)
    assert m["overlap_precision"] == round(2 / 3, 4)


def test_coverage_matches_arxiv_preprint_to_published():
    survey = [{"title": "Attention Is All You Need", "doi": "10.5555/3295222",
               "citations": 1000}]
    ours = [{"title": "attention is all you need", "doi": ""}]  # title-only match
    m = coverage_metrics(ours, survey)
    assert m["overlap"] == 1


def test_llm_rerank_falls_back_on_model_error(monkeypatch):
    from citens.eval import litsearch

    def boom(*a, **k):  # model hiccup must degrade to deterministic order
        raise RuntimeError("api down")

    monkeypatch.setattr("citens.llm.chat", boom)
    union = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(5)]
    assert litsearch.llm_rerank("q", union) == union


def test_blend_pool_reserves_citation_spine():
    from citens.search.base import blend_pool

    # 60 arXiv papers with 0 citations (arXiv records carry none) + one
    # 17k-citation classic captured via a citation-rich source + S2 filler
    arxiv = [Paper(title=f"Arxiv Paper {i}", source="arXiv") for i in range(60)]
    s2 = [Paper(title=f"S2 Paper {i}", source="Semantic Scholar",
                citation_count=5) for i in range(30)]
    classic = Paper(title="The Foundational Classic Work Here",
                    source="Semantic Scholar", citation_count=17000)
    pool = arxiv + s2 + [classic]
    out = blend_pool(pool, cap=24)
    # the spine (top-12 by citations) always survives — a classic must not
    # die to an arbitrary tie among 0-citation arXiv records
    assert any(p.title == "The Foundational Classic Work Here" for p in out)
    assert len(out) == 24
    # diversity fill still happens: arXiv records make the cut too
    assert any(p.source == "arXiv" for p in out)


# --- citation enrichment (bench-found: arXiv-only captures sort as uncited) ---

def test_paper_arxiv_id_from_url_doi_and_none():
    from citens.search.base import paper_arxiv_id

    assert paper_arxiv_id(
        _p("X", url="https://arxiv.org/abs/2005.11401v2")
    ) == "2005.11401"
    assert paper_arxiv_id(_p("X", doi="10.48550/ARXIV.2305.08596")) == "2305.08596"
    assert paper_arxiv_id(_p("X", url="https://example.com", doi="10.1/x")) is None


def test_enrich_citations_fills_counts_and_dois():
    import asyncio

    from citens.search.semantic_scholar import enrich_citations

    classic = Paper(
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        source="arXiv", url="https://arxiv.org/abs/2005.11401",
    )
    no_id = Paper(title="No Id Paper", source="arXiv", url="")
    already_cited = Paper(title="Cited Via S2", source="Semantic Scholar",
                          citation_count=99, url="https://arxiv.org/abs/1111.1111")

    async def fake_lookup(ids):
        assert "2005.11401" in ids
        return {"2005.11401": (17000, "10.5555/xyz")}

    n = asyncio.run(enrich_citations([classic, no_id, already_cited],
                                     _lookup=fake_lookup))
    assert n == 1
    assert classic.citation_count == 17000
    assert classic.doi == "10.5555/xyz"  # published DOI arrives with the join
    assert no_id.citation_count == 0     # nothing to join on — untouched
    assert already_cited.citation_count == 99


def test_pick_snowball_anchors_skips_doiless():
    from citens.eval.litsearch import pick_snowball_anchors

    papers = [
        _p("No DOI first"),           # fused head but snowball can't resolve it
        _p("Has DOI", doi="10.1/a"),
        _p("Also DOI", doi="10.1/b"),
        _p("Third DOI", doi="10.1/c"),
    ]
    anchors = pick_snowball_anchors(papers, k=2)
    assert [p.doi for p in anchors] == ["10.1/a", "10.1/b"]


def test_enrich_citations_degrades_to_noop_when_join_fails():
    import asyncio

    from citens.search.semantic_scholar import enrich_citations

    p = Paper(title="T", source="arXiv", url="https://arxiv.org/abs/2005.11401")

    async def boom(ids):
        raise RuntimeError("429 storm")

    assert asyncio.run(enrich_citations([p], _lookup=boom)) == 0
    assert p.citation_count == 0  # unchanged — enrichment is never a dependency


def test_hypothetical_queries_filters_and_caps(monkeypatch):
    import citens.agents.hypothetical as hyp

    monkeypatch.setattr(
        hyp, "chat_json",
        lambda *a, **k: {"titles": [
            "Short",  # too few words
            "A Descriptive Multi-Concept Title for External Knowledge in Dialogue",
            'Quoted Title That Is Long Enough To Keep',
            {"not": "a string"},
            "Bench Name Integration: A Study of X for Y and Z",
            "Fifth Title That Should Be Dropped By The Cap",
        ]},
    )
    out = hyp.hypothetical_queries("q", k=2)
    assert out == [
        "A Descriptive Multi-Concept Title for External Knowledge in Dialogue",
        "Quoted Title That Is Long Enough To Keep",
    ]

    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(hyp, "chat_json", boom)
    assert hyp.hypothetical_queries("q") == []  # degrade, never fail


def test_listwise_rank_reorders_head_and_keeps_tail(monkeypatch):
    from citens.agents import rerank

    papers = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(6)]
    monkeypatch.setattr(
        "citens.llm.chat", lambda *a, **k: '{"order": [3, 1, 0, 2]}'
    )
    out = rerank.listwise_rank("q", papers, top=4)
    # head reordered by the model, tail (P4, P5) follows untouched
    assert [p.title for p in out] == ["P3", "P1", "P0", "P2", "P4", "P5"]


def test_llm_rerank_delegates_to_shared_ranker(monkeypatch):
    from citens.eval import litsearch

    seen = {}

    def fake(query, papers, top=100):
        seen["top"] = top
        return list(papers)

    import citens.agents.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "listwise_rank", fake)
    union = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(5)]
    assert litsearch.llm_rerank("q", union, top=100) == union
    assert seen["top"] == 100  # the adaptive leg's pool-wide call shape


def test_cascade_rank_coarse_then_fine(monkeypatch):
    from citens.agents import rerank

    papers = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(6)]
    # coarse scores: P4 highest, P0 lowest
    monkeypatch.setattr(
        rerank, "_pointwise_scores",
        lambda q, ps, batch=40: [1, 2, 3, 4, 9, 5],
    )
    seen = {}

    def fake_listwise(question, ps, top=100, strong=False):
        seen["n"] = len(ps)
        seen["strong"] = strong
        seen["titles"] = [p.title for p in ps]
        return list(reversed(ps))

    monkeypatch.setattr(rerank, "listwise_rank", fake_listwise)
    out = rerank.cascade_rank("q", papers, coarse_keep=3)
    # coarse kept top-3 by score (P4, P5, P3); listwise reversed them;
    # the rest follow in coarse order (P2, P1, P0)
    assert [p.title for p in out] == ["P3", "P5", "P4", "P2", "P1", "P0"]
    assert seen["n"] == 3 and seen["titles"] == ["P4", "P5", "P3"]


def test_cascade_rank_degrades_when_coarse_fails(monkeypatch):
    from citens.agents import rerank

    papers = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(4)]
    monkeypatch.setattr(rerank, "_pointwise_scores", lambda q, ps, batch=40: None)
    monkeypatch.setattr(
        rerank, "listwise_rank",
        lambda q, ps, top=100, strong=False: list(ps),
    )
    out = rerank.cascade_rank("q", papers, coarse_keep=2)
    assert [p.title for p in out] == ["P0", "P1", "P2", "P3"]


def test_pointwise_scores_batches_and_validates(monkeypatch):
    from citens.agents import rerank

    papers = [_p(f"P{i}", doi=f"10.1/{i}") for i in range(3)]
    calls = []

    all_scores = ["2", "10", "0"]
    served = 0

    def fake_chat(sys_p, user_p, **kw):
        nonlocal served
        n = len([ln for ln in user_p.splitlines() if ln[:1].isdigit()])
        calls.append(n)
        got = all_scores[served:served + n]
        served += n
        return '{"scores": [' + ", ".join(got) + "]}"

    monkeypatch.setattr("citens.llm.chat", fake_chat)
    assert rerank._pointwise_scores("q", papers, batch=2) == [2, 10, 0]
    assert len(calls) == 2  # batched: 2 + 1

    # length mismatch = garbage, not partial scores
    monkeypatch.setattr("citens.llm.chat", lambda *a, **k: '{"scores": [1]}')
    assert rerank._pointwise_scores("q", papers) is None


def test_cached_queries_freezes_generation(tmp_path):
    from citens.eval.litsearch import _cached_queries

    cache_file = tmp_path / "gen_cache.json"
    calls = []

    def gen():
        calls.append(1)
        return ["a query", "another"]

    out1 = _cached_queries("hyde", "Q1?", gen, path=cache_file)
    out2 = _cached_queries("hyde", "Q1?", gen, path=cache_file)  # cache hit
    assert out1 == out2 == ["a query", "another"]
    assert len(calls) == 1  # generator ran once, second call was a hit
    # a different question is a different key
    _cached_queries("hyde", "Q2?", gen, path=cache_file)
    assert len(calls) == 2


def test_oa_title_filter_informative_words_only():
    from citens.eval.litsearch import oa_title_filter

    f = oa_title_filter("MISC: A Mixed Strategy-Aware Model integrating COMET for")
    assert f == "MISC Mixed Strategy Aware Model integrating COMET for"


def test_oa_title_search_caches_and_never_caches_failure(tmp_path, monkeypatch):
    import asyncio

    from citens.eval import litsearch as ls

    calls = []

    async def fake_fetch(filt):
        calls.append(filt)
        if len(calls) == 1:
            return [{"display_name": "MTAG: Modal-Temporal Attention Graph",
                     "doi": "https://doi.org/10.1/x", "publication_year": 2022,
                     "cited_by_count": 55}]
        raise RuntimeError("openalex 429")

    monkeypatch.chdir(tmp_path)  # cache under cwd like the rest of the bench
    papers = asyncio.run(ls.oa_title_search(["Q One", "Q Two"], fetch=fake_fetch))
    assert [p.title[:4] for p in papers["Q One"]] == ["MTAG"]
    assert papers["Q One"][0].doi == "10.1/x"
    assert papers["Q Two"] == []  # failure -> empty, not cached
    # second round: hit read from disk, only the failed one refetches
    papers2 = asyncio.run(ls.oa_title_search(["Q One", "Q Two"], fetch=fake_fetch))
    assert [p.title[:4] for p in papers2["Q One"]] == ["MTAG"]
    assert papers2["Q Two"] == []
    assert len(calls) == 3  # Q Two retried (not poisoned), Q One did not
