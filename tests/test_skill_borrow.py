"""Borrowed from the nature-* academic skills: five-grade verifier scale,
review-as-context rule, RIS export, chunk anchors, terminology ledger,
primary-source ordering, CJK lexical retrieval, FTS5 fast path, and the
self-contained HTML review browser."""

from __future__ import annotations

import json

from citens import collect as collect_mod  # noqa: F401  (kept for parity w/ other tests)
from citens.agents import rewriter as rewriter_mod
from citens.agents import verifier as verifier_mod
from citens.artifacts import write_review_browser
from citens.audit import ingest_audit
from citens.grounding import ChunkStore, CitationTable, build_provenance
from citens.grounding.retrieval import _terms, bm25_rank_texts
from citens.models import Chunk, ChunkKind, Claim, Paper, Verdict, VerificationResult
from citens.profiles import load_profile, order_sources


def _p(title, **kw):
    defaults = dict(authors=["A Author"], year=2020, abstract="abs text",
                    citation_count=1)
    defaults.update(kw)
    return Paper(title=title, **defaults)


# --- five-grade support scale -------------------------------------------


def test_verdict_enum_has_background_and_contradictory():
    assert Verdict("background") is Verdict.BACKGROUND
    assert Verdict("contradictory") is Verdict.CONTRADICTORY
    # the grounded set is unchanged: new grades count against precision
    grounded = {Verdict.SUPPORTED, Verdict.PARTIAL}
    assert Verdict.BACKGROUND not in grounded
    assert Verdict.CONTRADICTORY not in grounded


def test_verifier_context_tags_review_papers():
    papers = [_p("Primary study"), _p("A survey of the field", is_review=True)]
    table = CitationTable(papers)
    store = ChunkStore()
    store._by_paper[papers[0].id] = [
        Chunk(paper_id=papers[0].id, chunk_id="a", text="alpha", kind=ChunkKind.ABSTRACT)
    ]
    store._by_paper[papers[1].id] = [
        Chunk(paper_id=papers[1].id, chunk_id="b", text="beta", kind=ChunkKind.ABSTRACT)
    ]
    ctx = verifier_mod._build_context([0, 1], table, store, "query")
    assert "[0] " in ctx and " [REVIEW]" not in ctx.split("\n")[0]
    assert "[1] " in ctx and " [REVIEW]" in ctx


def test_rewriter_targets_all_defect_verdicts(monkeypatch):
    claims = [
        Claim(text="claim zero [0]", citation_indices=[0]),
        Claim(text="claim one [1]", citation_indices=[1]),
        Claim(text="claim two [2]", citation_indices=[2]),
    ]
    verdicts = [
        VerificationResult(claim_text=claims[0].text, verdict=Verdict.UNSUPPORTED),
        VerificationResult(claim_text=claims[1].text, verdict=Verdict.BACKGROUND),
        VerificationResult(claim_text=claims[2].text, verdict=Verdict.SUPPORTED),
    ]

    def fake_chat_json(system, user, **k):
        # must see the background defect tagged; rewrite only claim 1
        assert "[background]" in user
        return {"rewrites": [{"claim_index": 1, "new_text": "context-only [1]",
                              "note": "weakened"}]}

    monkeypatch.setattr(rewriter_mod, "chat_json", fake_chat_json)
    table = CitationTable([_p("p0"), _p("p1"), _p("p2")])
    store = ChunkStore()
    rewrites = rewriter_mod.rewrite_unsupported_claims(claims, verdicts, table, store)
    assert 1 in rewrites and rewrites[1]["new_text"].startswith("context-only")


# --- RIS export + chunk anchors ----------------------------------------


def test_to_ris_emits_standard_fields_and_omits_missing():
    table = CitationTable([_p("Alpha paper", doi="10.1/x", url="https://doi.org/10.1/x")])
    ris = table.to_ris()
    assert ris.startswith("TY  - JOUR")
    assert "AU  - A Author" in ris
    assert "TI  - Alpha paper" in ris
    assert "DO  - 10.1/x" in ris
    assert ris.rstrip().endswith("ER  -")
    assert "VL  -" not in ris  # never invent fields


def test_build_provenance_attaches_evidence_chunk_anchors():
    papers = [_p("Grounded paper")]
    table = CitationTable(papers)
    store = ChunkStore()
    store._by_paper[papers[0].id] = [
        Chunk(paper_id=papers[0].id, chunk_id="abc-ft-2",
              text="the exact evidence sentence " * 5, kind=ChunkKind.FULLTEXT),
    ]
    claims = [Claim(text="a claim about the evidence [0]", citation_indices=[0])]
    prov = build_provenance(claims, table, chunk_store=store)
    assert prov[0]["evidence_chunks"][0]["chunk_id"] == "abc-ft-2"
    assert prov[0]["evidence_chunks"][0]["kind"] == "fulltext"
    assert len(prov[0]["evidence_chunks"][0]["excerpt"]) <= 200


# --- profile: terminology ledger + primary sources ----------------------


def test_finance_profile_has_terminology_and_source_order():
    prof = load_profile("finance")
    assert prof is not None
    assert prof.terminology.get("order flow imbalance") == "订单流失衡"
    assert prof.primary_sources[0] == "openalex"

    line = prof.terminology_line()
    assert "order flow imbalance=订单流失衡" in line

    ordered = order_sources(["arxiv", "openalex", "crossref"], prof)
    assert ordered == ["openalex", "crossref", "arxiv"]
    # unknown sources keep their position at the end
    assert order_sources(["arxiv", "custom", "openalex"], prof) == [
        "openalex", "arxiv", "custom",
    ]
    # no profile -> untouched (including None)
    assert order_sources(["arxiv", "openalex"], None) == ["arxiv", "openalex"]


def test_writer_ledger_line_reaches_prompts(monkeypatch):
    from citens.agents import writer as writer_mod
    from citens.config import settings
    from citens.models import ExtractedPaper, ThemeInfo, ThemeStructure

    seen: list[str] = []

    def fake_chat(system, user, max_tokens=0, strong=False, thinking=True):
        seen.append(system)
        return "正文一句。" + "x" * 200 + "。"

    monkeypatch.setattr(writer_mod, "chat", fake_chat)
    monkeypatch.setattr(settings, "review_language", "zh")
    papers = [
        ExtractedPaper(title="t", authors=["A"], year=2020, abstract="a",
                       research_question="q", methodology="m", key_findings=["f"],
                       limitations=["l"]),
    ]
    themes = ThemeStructure(themes=[ThemeInfo(name="主题", description="d",
                                              paper_indices=[0])])
    writer_mod.write_review_body(
        papers, themes, "topic",
        terminology={"order flow imbalance": "订单流失衡"},
    )
    assert any("TERMINOLOGY LEDGER" in s and "订单流失衡" in s for s in seen)


# --- CJK lexical retrieval + FTS5 fast path ------------------------------


def test_cjk_text_produces_bigram_terms():
    terms = _terms("订单流失衡 order flow")
    assert "订单" in terms and "单流" in terms and "流失" in terms
    assert "order" in terms and "flow" in terms


def test_bm25_ranks_chinese_matches_above_noise():
    texts = [
        "机器学习在股票收益预测中的应用研究",
        "deep learning for image classification datasets",
        "基于神经网络的高频交易策略",
    ]
    order = bm25_rank_texts(texts, "机器学习 股票 收益 预测")
    assert order[0] == 0


def test_fts5_fast_path_orders_large_corpus():
    n = 500  # above _FTS5_MIN
    texts = [f"irrelevant filler document number {i}" for i in range(n)]
    texts[42] = "limit order book queue position dynamics " * 3
    order = bm25_rank_texts(texts, "limit order book queue")
    assert len(order) == n          # every index returned, FTS or not
    assert order[0] == 42


def test_fts5_unavailable_falls_back(monkeypatch):
    import citens.grounding.retrieval as ret

    def _boom(texts, query):
        raise RuntimeError("no fts5")

    monkeypatch.setattr(ret, "_fts5_rank", lambda t, q: None)
    texts = [f"doc {i} alpha" for i in range(600)]
    order = bm25_rank_texts(texts, "alpha")
    assert len(order) == 600


# --- complete APA references (technical-report house style) --------------


def test_apa_reference_includes_volume_issue_pages_doi():
    p = _p(
        "Deep order books",
        authors=["Zheng Zhao", "Wei Fan", "Bin Li"],
        venue="Journal of Finance", volume="36", issue="8",
        pages="4387-4403", doi="10.1111/jofi.12345",
    )
    label = CitationTable([p]).label(0)
    assert "Zhao, Z., Fan, W., Li, B." in label   # APA authors, not 3+et-al
    assert "(2020). Deep order books." in label
    assert "*Journal of Finance*" in label        # italic venue
    assert ", 36(8), 4387-4403." in label         # volume(issue), pages
    assert "https://doi.org/10.1111/jofi.12345" in label


def test_bib_and_ris_carry_biblio_fields():
    p = _p("Alpha paper", venue="Quantitative Finance", volume="12", issue="3",
           pages="301-319", doi="10.1080/14697688.2024.1234567")
    table = CitationTable([p])
    bib = table.to_bibtex()
    assert "volume = {12}" in bib
    assert "number = {3}" in bib
    assert "pages = {301--319}" in bib  # BibTeX page range dashes
    ris = table.to_ris()
    assert "VL  - 12" in ris and "IS  - 3" in ris
    assert "SP  - 301" in ris and "EP  - 319" in ris


def test_openalex_to_paper_parses_biblio():
    from citens.search.openalex import OpenAlexSearcher

    work = {
        "title": "T", "publication_year": 2020, "cited_by_count": 1,
        "authorships": [{"author": {"display_name": "Zheng Zhao"}}],
        "primary_location": {"source": {"display_name": "Journal of Finance"}},
        "biblio": {"volume": "36", "issue": "8",
                   "first_page": "4387", "last_page": "4403"},
    }
    p = OpenAlexSearcher.to_paper(work)
    assert (p.volume, p.issue, p.pages) == ("36", "8", "4387-4403")
    assert p.venue == "Journal of Finance"


def test_crossref_to_paper_parses_biblio():
    from citens.search.crossref import CrossrefSearcher

    item = {
        "title": ["T"], "is-referenced-by-count": 2, "DOI": "10.1/x",
        "author": [{"given": "Wei", "family": "Fan"}],
        "container-title": ["Journal of Finance"],
        "volume": "79", "issue": "4", "page": "921–955",  # en-dash
        "issued": {"date-parts": [[2014]]},
    }
    p = CrossrefSearcher._to_paper(item)
    assert (p.volume, p.issue, p.pages) == ("79", "4", "921-955")


# --- Chinese by default ---------------------------------------------------


def test_default_review_language_is_chinese():
    from citens.config import Settings

    assert Settings().review_language == "zh"


# --- audit ingest normalizes new verdicts -------------------------------


def test_audit_ingest_treats_background_as_unsupported(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "verification.json").write_text(
        json.dumps({
            "citation_precision": 0.5,
            "results": [
                {"claim_text": "c0", "verdict": "background",
                 "citation_indices": [], "note": ""},
                {"claim_text": "c1", "verdict": "supported",
                 "citation_indices": [], "note": ""},
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    sheet = tmp_path / "sheet.md"
    sheet.write_text(
        "## 论断 1 — 机器判定: background\n> c0\n- 人工判定: u\n\n"
        "## 论断 2 — 机器判定: supported\n> c1\n- 人工判定: s\n",
        encoding="utf-8",
    )
    report = ingest_audit(str(run), str(sheet))
    assert report["judged"] == 2
    assert report["agreement_rate"] == 1.0  # background≡u counted as agreement


# --- HTML review browser -------------------------------------------------


def _write_run(run, claims, verdicts, provenance=None):
    (run / "verification.json").write_text(
        json.dumps({
            "citation_precision": 0.5,
            "pre_rewrite_precision": 0.4, "post_rewrite_precision": 0.5,
            "supported": 1, "partial": 0, "background": 0, "contradictory": 0,
            "unsupported": 1, "unverifiable": 0,
            "results": [
                {"claim_text": c, "verdict": v, "citation_indices": [0],
                 "note": "note"}
                for c, v in zip(claims, verdicts, strict=True)
            ],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    if provenance is not None:
        (run / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False), encoding="utf-8")
    (run / "meta.json").write_text(
        json.dumps({"topic": "机器学习股票收益"}, ensure_ascii=False),
        encoding="utf-8")
    (run / "grounding.json").write_text(
        json.dumps({"with_fulltext": 1, "total": 2,
                    "papers": [{"index": 0, "title": "p0", "has_fulltext": True,
                                "n_chunks": 3}]},
                   ensure_ascii=False),
        encoding="utf-8")


def test_review_browser_bundles_claims_verdicts_and_downloads(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    prov = [{
        "claim": "claim one [0]", "section": "s",
        "citations": [{"index": 0, "label": "l", "paper_id": "x"}],
        "evidence_chunks": [{"index": 0, "chunk_id": "abc-ft-1",
                             "kind": "fulltext", "excerpt": "the evidence"}],
    }]
    _write_run(run, ["claim one [0]", "claim two [0]"],
               ["supported", "unsupported"], prov)
    (run / "review.md").write_text("# 综述", encoding="utf-8")
    (run / "references.bib").write_text("@article{k,\n}", encoding="utf-8")
    (run / "references.ris").write_text("TY  - JOUR\nER  - \n", encoding="utf-8")

    path = write_review_browser(str(run))
    assert path and str(path).endswith("review_browser.html")
    html = (run / "review_browser.html").read_text(encoding="utf-8")
    assert "机器学习股票收益" in html
    assert "claim one [0]" in html           # data embedded, not lost
    assert "abc-ft-1" in html                # chunk anchors embedded
    assert "references.ris" in html
    assert "<script>" in html and "__DATA__" not in html


def test_review_browser_without_verification_returns_none(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert write_review_browser(str(run)) is None


# --- formal journal over arXiv preprint (references) ---------------------


def test_best_venue_prefers_journal_over_arxiv():
    from citens.search.openalex import best_venue

    work = {
        "primary_location": {"source": {"display_name": "arXiv"}},
        "locations": [
            {"source": {"display_name": "arXiv"}},
            {"source": {"display_name": "Journal of Finance"}},
            {"source": {"display_name": "Journal of Finance"}},
        ],
    }
    assert best_venue(work) == "Journal of Finance"
    # preprint-only work keeps its host
    assert best_venue({"primary_location": {"source": {"display_name": "arXiv"}},
                       "locations": [{"source": {"display_name": "arXiv"}}]}) == "arXiv"


def test_openalex_to_paper_cites_journal_when_available():
    from citens.search.openalex import OpenAlexSearcher

    work = {
        "title": "T", "publication_year": 2020, "cited_by_count": 1,
        "authorships": [{"author": {"display_name": "Zheng Zhao"}}],
        "primary_location": {"source": {"display_name": "arXiv"}},
        "locations": [
            {"source": {"display_name": "arXiv"}},
            {"source": {"display_name": "Review of Financial Studies"}},
        ],
        "biblio": {"volume": "34", "issue": "2", "first_page": "1", "last_page": "30"},
    }
    p = OpenAlexSearcher.to_paper(work)
    assert p.venue == "Review of Financial Studies"


# --- theme names follow the review language ------------------------------


def test_organize_localizes_theme_names_for_chinese(monkeypatch):
    from citens.agents import organize as organize_mod
    from citens.config import settings

    assert "theme name 字段用中文" in organize_mod._localization_line() \
        if (monkeypatch.setattr(settings, "review_language", "zh") or True) else False
    monkeypatch.setattr(settings, "review_language", "zh")
    assert "theme name 字段用中文" in organize_mod._localization_line()
    monkeypatch.setattr(settings, "review_language", "en")
    assert "theme name 字段用中文" not in organize_mod._localization_line()


def test_default_max_papers_is_20():
    from citens.config import Settings

    # _env_file=None: ignore the user's .env so the code default is what we test
    assert Settings(_env_file=None).default_max_papers == 20


# --- supporting-reference layer ------------------------------------------


def test_supporting_block_renders_indices_and_guardrail():
    from citens.agents.writer import _supporting_block

    supp = [(7, _p("Context survey of the field", venue="Journal of Finance",
                   abstract="A broad survey " * 20))]
    block = _supporting_block(supp)
    assert "[7] Context survey of the field" in block
    assert "Journal of Finance (2020)" in block
    assert "NOT for primary claims" in block  # guardrail travels with the block
    assert _supporting_block(None) == ""


def test_writer_passes_supporting_into_prompts(monkeypatch):
    from citens.agents import writer as writer_mod
    from citens.config import settings
    from citens.models import ExtractedPaper, ThemeInfo, ThemeStructure

    seen: list[str] = []

    def fake_chat(system, user, max_tokens=0, strong=False, thinking=True):
        seen.append(user)
        return "正文一句。" + "x" * 200 + "。"

    monkeypatch.setattr(writer_mod, "chat", fake_chat)
    monkeypatch.setattr(settings, "review_language", "zh")
    papers = [
        ExtractedPaper(title="core", authors=["A"], year=2020, abstract="a",
                       research_question="q", methodology="m", key_findings=["f"],
                       limitations=["l"]),
    ]
    themes = ThemeStructure(themes=[ThemeInfo(name="主题", description="d",
                                              paper_indices=[0])])
    writer_mod.write_review_body(
        papers, themes, "topic",
        supporting=[(1, _p("Supporting context paper",
                           abstract="supporting abstract " * 10))],
    )
    # theme + conclusion prompts carry the supporting entry with its index
    assert any("[1] Supporting context paper" in u for u in seen)


def test_citation_table_includes_supporting_after_core():
    core = [_p("Core one")]
    supp = [_p("Supporting one"), _p("Supporting two")]
    table = CitationTable(core + supp)
    assert table.paper_id(0) == core[0].id
    assert table.paper_id(1) == supp[0].id
    assert len(table) == 3


def test_chunk_store_abstract_only_for_supporting():
    store = ChunkStore()
    p = _p("Supporting abstract paper", abstract="a real abstract here")
    store.build_from([p], fetch_full=False)
    chunks = store.chunks_for(p.id)
    assert len(chunks) == 1 and chunks[0].kind.value == "abstract"
    assert store.has(p.id)  # verifier can check claims against it


def test_default_support_papers_is_15():
    from citens.config import Settings

    assert Settings(_env_file=None).default_support_papers == 15


# --- clarifications reach the retrieval side -----------------------------


def test_parse_constraints_reads_timeframe_and_venue():
    from citens.search.filters import parse_constraints

    filters = {
        "focus": "深度学习模型（如LSTM、Transformer）",
        "scope": "以实证研究为主（含回测）",
        "timeframe": "近5年（2019-2024）",   # stale years; 近N年 must win
        "venue": "仅顶级金融/经济期刊",
    }
    c = parse_constraints(filters)
    import datetime

    cy = datetime.date.today().year
    assert c.year_from == cy - 4 and c.year_to == cy
    assert c.venue_strict is True

    c2 = parse_constraints({"timeframe": "2000年至今"})
    assert c2.year_from == 2000 and c2.year_to is None
    c3 = parse_constraints({"timeframe": "2014-2024"})
    assert (c3.year_from, c3.year_to) == (2014, 2024)
    c4 = parse_constraints({"venue": "所有同行评审期刊"})
    assert c4.venue_strict is False
    assert parse_constraints(None).describe() == ""


def test_constraints_matches_paper_year_window():
    from citens.search.filters import RetrievalConstraints

    c = RetrievalConstraints(year_from=2022, year_to=2026)
    assert c.matches_paper(_p("new", year=2024))
    assert not c.matches_paper(_p("old", year=2015))
    assert not c.matches_paper(_p("no-year", year=None))


def test_recall_from_pool_applies_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    collect_mod.append_pool("t", [
        _p("deep learning returns", year=2023),
        _p("classic DID paper", year=2008),
        _p("recent transformer forecasting", year=2025),
    ])
    from citens.search.filters import RetrievalConstraints

    got = collect_mod.recall_from_pool(
        "t", ["deep learning stock"], 10,
        constraints=RetrievalConstraints(year_from=2022, year_to=2026),
    )
    years = {p.year for p in got}
    assert 2008 not in years and years <= {2023, 2025}


def test_recall_venue_strict_keeps_whitelist_and_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_mod.settings, "litdb_dir", str(tmp_path))
    collect_mod.append_pool("t", [
        _p("top journal paper", year=2023, venue="Journal of Finance"),
        _p("mid journal paper", year=2023, venue="Random Journal"),
        _p("a field survey", year=2023, venue="arXiv", is_review=True),
    ])
    from citens.search.filters import RetrievalConstraints

    got = collect_mod.recall_from_pool(
        "t", ["survey"], 10,
        constraints=RetrievalConstraints(venue_strict=True),
        venue_whitelist={"journal of finance"},
    )
    titles = {p.title for p in got}
    assert "top journal paper" in titles
    assert "mid journal paper" not in titles
    assert "a field survey" in titles  # reviews stay citable as context


def test_clarify_rewrites_stale_year_ranges():
    import datetime

    from citens.agents.clarify import _fresh_years

    cy = datetime.date.today().year
    assert _fresh_years("近5年（2019-2024）") == f"近5年（{cy - 4}-{cy}）"
    assert _fresh_years("2000年至今") == "2000年至今"  # anchor year kept
    assert _fresh_years("不限时间") == "不限时间"


# --- blind-paper demotion + S2 enrichment + writer register (2026-08-19) ---


def test_demote_blind_papers_swaps_in_abstract_alternates():
    from citens.models import ScoredPaper
    from citens.orchestration.pipeline import demote_blind_papers

    def _sp(title, abstract, score=4.0):
        return ScoredPaper(title=title, abstract=abstract, rank_score=score)

    core = [_sp("blind one", ""), _sp("sighted", "real abstract " * 10)]
    supporting = [_sp("alt one", "alternate abstract " * 10, score=3.5)]
    new_core, new_supporting, log = demote_blind_papers(core, supporting)
    titles_core = {p.title for p in new_core}
    assert "blind one" not in titles_core and "alt one" in titles_core
    assert any(p.title == "blind one" for p in new_supporting)
    demoted = [s for s in log if s["action"] == "demoted_to_supporting"]
    assert len(demoted) == 1 and demoted[0]["replaced_by"] == "alt one"


def test_demote_blind_keeps_pdf_capable_papers():
    from citens.models import ScoredPaper
    from citens.orchestration.pipeline import demote_blind_papers

    core = [ScoredPaper(title="oa blind", abstract="", pdf_url="https://x/p.pdf")]
    new_core, _, log = demote_blind_papers(core, [])
    assert len(new_core) == 1 and not log  # fulltext-capable: not blind


def test_enrichment_fills_via_semantic_scholar(monkeypatch):
    from citens.grounding import enrichment as en

    monkeypatch.setattr(en, "_openalex_by_doi", lambda doi: "")
    monkeypatch.setattr(en, "_s2_by_doi", lambda doi: "S2 has the abstract")
    monkeypatch.setattr(en, "fetch_abstract_by_doi", lambda doi: "")
    paper = Paper(title="Elsevier no-abstract", abstract="", doi="10.1016/j.x")
    filled, log = en.enrich_abstracts([paper])
    assert filled == 1
    assert paper.abstract == "S2 has the abstract"
    assert log[0]["via"] == "semantic_scholar"


def test_s2_by_doi_throttles_and_retries_429(monkeypatch):
    import citens.grounding.enrichment as en

    monkeypatch.setattr(en.settings, "semantic_scholar_api_key", "k-test", raising=False)
    attempts = iter([(429, ""), (200, "recovered abstract")])
    sleeps = []
    monkeypatch.setattr(en, "_s2_get", lambda doi: next(attempts))
    monkeypatch.setattr(en.time, "sleep", lambda s: sleeps.append(s))
    got = en._s2_by_doi("10.1/x")
    assert got == "recovered abstract"  # 429 -> backoff -> success
    assert any(s >= 2.0 for s in sleeps)  # backed off before the retry


def test_s2_by_doi_gives_up_on_404(monkeypatch):
    import citens.grounding.enrichment as en

    monkeypatch.setattr(en.settings, "semantic_scholar_api_key", "k-test", raising=False)
    calls = []
    monkeypatch.setattr(en, "_s2_get", lambda doi: (calls.append(doi), (404, ""))[1])
    assert en._s2_by_doi("10.1/y") == ""
    assert len(calls) == 1  # a 404 is not retried


def test_writer_formality_and_stacking_rules_wired(monkeypatch):
    from citens.agents import writer as w

    # hard numeric stacking cap in the section prompt
    assert "AT MOST 3 citation markers" in w.SECTION_PROMPT
    assert "NEVER 5+" in w.SECTION_PROMPT
    # nature-writing register rules, localized per output language
    monkeypatch.setattr(w.settings, "review_language", "zh", raising=False)
    assert "一句只承载一个主要命题" in w.formality_instruction()
    assert "耐人寻味" in w.formality_instruction()  # banned patterns listed
    monkeypatch.setattr(w.settings, "review_language", "en", raising=False)
    assert "one main proposition per sentence" in w.formality_instruction()


def test_claim_stack_stats_flags_stacked_claims():
    from citens.agents.verifier import claim_stack_stats

    claims = [
        Claim(text="a [0]", citation_indices=[0]),
        Claim(text="b [0][1][2][3][4][5]", citation_indices=[0, 1, 2, 3, 4, 5]),
        Claim(text="c [1][2][3][4][5][6][7][8][9][10][11][12]",
              citation_indices=list(range(13))),
    ]
    stats = claim_stack_stats(claims)
    assert stats == {"max_citations_per_claim": 13, "stacked_claims": 2}


def test_organize_degrades_to_fallback_grouping(monkeypatch):
    from citens.agents import organize as org
    from citens.models import ExtractedPaper

    papers = [ExtractedPaper(title=f"p{i}", abstract="abs", year=2020)
              for i in range(9)]
    # a truncated/garbled LLM response must not kill the run
    monkeypatch.setattr(org, "chat_json",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("bad json")))
    structure = org.organize_themes(papers, "topic")
    assert structure.themes  # deterministic rank-order grouping kicked in
    assert all("自动分组" in t.name for t in structure.themes)
    covered = [i for t in structure.themes for i in t.paper_indices]
    assert sorted(covered) == list(range(9))  # every paper assigned


# --- thinking-budget control (deepseek-v4-flash shares completion tokens
# between thinking and the visible body; long deliberation starves the body) ---


def test_build_completion_kwargs_thinking_toggle():
    from citens.llm import build_completion_kwargs

    base = build_completion_kwargs(
        "m", system_prompt="s", user_prompt="u", temperature=0.3,
        max_tokens=100, response_json=True,
    )
    assert base["response_format"] == {"type": "json_object"}
    assert "extra_body" not in base  # thinking on: no interference

    off = build_completion_kwargs(
        "m", system_prompt="s", user_prompt="u", temperature=0.3,
        max_tokens=100, response_json=False, thinking=False,
    )
    assert off["extra_body"] == {"reasoning_effort": "none"}


def test_writer_last_resort_attempt_disables_thinking(monkeypatch):
    from citens.agents import writer as w

    calls = []

    def fake_chat(system, user, *, max_tokens=0, strong=False, thinking=True, **kw):
        calls.append({"budget": max_tokens, "thinking": thinking})
        if len(calls) < 3:
            return ""  # provider spell: two empty bodies
        return "一个完整的段落，以句号结尾。" * 20  # attempt 3 succeeds

    monkeypatch.setattr(w, "chat", fake_chat)
    monkeypatch.setattr(w.time, "sleep", lambda s: None)
    text = w._chat_section("sys", "user", 4096, "test")
    assert len(text) > 200  # recovered
    # flipped ladder: no-thinking first (fast + immune to the budget-eating
    # thinking prefix), thinking only as the double-budget last resort
    assert [c["thinking"] for c in calls] == [False, False, True]
    assert calls[1]["budget"] == 8192 and calls[2]["budget"] == 8192


# --- facet coverage / stacking lint / verdict cache / supplement gate (08-19) ---


def test_facet_coverage_report_and_note():
    from citens.models import ThemeInfo
    from citens.orchestration.pipeline import coverage_note_text, facet_coverage_report

    facets = [{"name": "Transformer", "queries": ["transformer attention forecasting"]},
              {"name": "GNN", "queries": ["graph neural network relational"]}]
    papers = [
        _p("Transformer models for stock prediction", abstract="transformer attention"),
        _p("Graph neural networks in finance", abstract="gnn graphs"),
        _p("Unrelated econometrics paper", abstract="instrumental variables"),
    ]
    report = facet_coverage_report(facets, papers)
    assert report == [
        {"facet": "Transformer", "papers": 1},
        {"facet": "GNN", "papers": 1},
    ]
    themes = [ThemeInfo(name="thin theme", description="", paper_indices=[0, 1])]
    note = coverage_note_text(report, themes, n_blind=2)
    assert "Transformer(1篇)" in note and "GNN(1篇)" in note
    assert "thin theme" in note and "2 篇论文无摘要" in note


def test_prune_citation_stacking_enforces_cap():
    from citens.orchestration.pipeline import prune_citation_stacking

    papers = [
        _p(f"paper {i}", abstract=f"alpha topic {i} " * 3) for i in range(8)
    ]
    # paper 3's abstract shares terms with the sentence -> must survive;
    # the decorative cites get stripped
    papers[3] = _p("niche market making paper", abstract="market making spread alpha")
    sent = "Market making and the spread mechanism are central[0][1][2][3][4][5]。后续讨论展开。"
    review, log = prune_citation_stacking(sent, papers, max_cites=2)
    assert len(log) == 1
    assert 3 in log[0]["kept"] and len(log[0]["kept"]) == 2
    assert review.count("[") == 2  # only the two keepers remain
    assert "后续讨论展开。" in review  # untouched sentence intact


def test_verdict_cache_skips_unchanged_claims(monkeypatch):
    from citens.agents import verifier as V

    papers = [_p("Grounded one"), _p("Grounded two")]
    table = CitationTable(papers)
    store = ChunkStore()
    for i, p in enumerate(papers):
        store._by_paper[p.id] = [
            Chunk(paper_id=p.id, chunk_id=f"c{i}", text=f"abstract {i}",
                  kind=ChunkKind.ABSTRACT)
        ]
    calls = []

    def fake_chat_json(system, user, **k):
        calls.append(1)
        return {"results": [
            {"claim_index": j, "verdict": "supported", "note": "ok"}
            for j in range(10)
        ]}

    monkeypatch.setattr(V, "chat_json", fake_chat_json)
    claims = [Claim(text=f"claim {i} [0]", citation_indices=[0]) for i in range(3)]
    cache: dict = {}
    r1, p1 = V.verify_claims(claims, table, store, verdict_cache=cache)
    assert p1 == 1.0 and len(cache) == 3
    n_first = len(calls)
    r2, p2 = V.verify_claims(claims, table, store, verdict_cache=cache)
    assert len(calls) == n_first  # second round: all cache hits, zero judge calls
    assert p2 == 1.0
    assert [r.verdict for r in r2] == [r.verdict for r in r1]


def test_gate_supplement_papers_demotes_blind():
    from citens.orchestration.pipeline import _gate_supplement_papers

    fresh = [_p("sighted supplement", abstract="real abstract"),
             _p("blind supplement", abstract="")]
    kept, supporting, blind = _gate_supplement_papers(fresh, [])
    assert [p.title for p in kept] == ["sighted supplement"]
    assert [p.title for p in blind] == ["blind supplement"]
    assert supporting == blind


def test_generate_facets_parses_and_is_resilient(monkeypatch):
    from citens.agents import planner as pl

    monkeypatch.setattr(
        pl, "chat_json",
        lambda *a, **k: {"facets": [{"name": "Classics", "queries": ["survey lob"]},
                                    {"name": "", "queries": ["x"]}]},
    )
    facets = pl.generate_facets("order book")
    assert facets == [{"name": "Classics", "queries": ["survey lob"]}]

    monkeypatch.setattr(
        pl, "chat_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("backend down")),
    )
    assert pl.generate_facets("order book") == []  # accelerator, never a pillar


# --- density & unfolding package (08-19b: length, names, numbers, abstract) ---


def test_extract_carries_system_name_and_keeps_numbers():
    from citens.agents.extract import _build_extracted
    from citens.models import ScoredPaper

    p = ScoredPaper(title="TALLRec: effective tuning", abstract="abs")
    result = {
        "system_name": "TALLRec",
        "research_question": "align LLM with rec",
        "key_findings": ["Recall@1 improved by 12.3% over base LLaMA"],
    }
    ep = _build_extracted(p, result, assess_quality=False)
    assert ep.system_name == "TALLRec"
    assert "12.3%" in ep.key_findings[0]
    # prompt contract: system_name in schema, numbers mandatory
    from citens.agents.extract import SYSTEM_PROMPT
    assert '"system_name"' in SYSTEM_PROMPT
    assert "NUMBERS ARE MANDATORY CARGO" in SYSTEM_PROMPT


def test_papers_block_shows_system_name():
    from citens.agents import writer as writer_mod
    from citens.models import ExtractedPaper

    ep = ExtractedPaper(title="Named paper", abstract="a", year=2020,
                        system_name="FactorVAE")
    block = writer_mod._papers_block([(5, ep)])
    assert "系统名: FactorVAE" in block


def test_section_prompt_has_unfolding_template_and_length():
    from citens.agents import writer as writer_mod

    assert "1500-2200" in writer_mod.SECTION_PROMPT  # zh length target
    assert "定位段" in writer_mod.SECTION_PROMPT and "收束段" in writer_mod.SECTION_PROMPT
    assert "NAMED SYSTEMS" in writer_mod.SECTION_PROMPT
    assert "DENSITY" in writer_mod.SECTION_PROMPT
    assert "作者等人[n]提出的" in writer_mod.SECTION_PROMPT


def test_abstract_prompt_contract():
    from citens.agents import writer as writer_mod

    assert "摘要：" in writer_mod.ABSTRACT_PROMPT
    assert "关键词：" in writer_mod.ABSTRACT_PROMPT
    assert "NO [n] citation markers" in writer_mod.ABSTRACT_PROMPT


def test_search_summary_text_reports_funnel():
    from citens.orchestration.pipeline import search_summary_text

    s = search_summary_text(["arxiv", "semantic_scholar"], 40, 26, 8, "2026-08-19")
    assert "40 篇候选" in s and "26 篇" in s and "8 篇支持文献" in s
    assert "2026-08-19" in s


def test_claim_parser_splits_chinese_sentences_without_spaces():
    # regression: the old `\s+` requirement merged whole paragraphs into one
    # claim (no spaces after 。), inflating per-claim citation counts
    from citens.grounding.citations import parse_claims_from_review

    md = "# t\n\n## 引言\n\n第一句引用了[0]。第二句引用了[1][2]。\n"
    claims = parse_claims_from_review(md)
    assert len(claims) == 2
    assert claims[0].citation_indices == [0]
    assert claims[1].citation_indices == [1, 2]


def test_claim_parser_never_splits_on_decimal_points():
    # regression: zero-width split after every `.` shattered numbers into
    # context-free fragments (`0.` + `92[15]。`) — the verifier then judged
    # claim-text like "92[15]。" with no meaning
    from citens.grounding.citations import parse_claims_from_review

    md = (
        "# t\n\n## 实证\n\n"
        "未经微调的LLM表现接近随机猜测（AUC≈0.5），微调后显著改善[3]。"
        "系统级排序与人类判断高度一致，Kendall's τ最高可达0.92[15]。"
        "English too: the model attains 0.879 NDCG on 1.5M interactions[7]. "
        "Next sentence[8].\n"
    )
    claims = parse_claims_from_review(md)
    assert len(claims) == 4
    assert "AUC≈0.5" in claims[0].text and "0.92" in claims[1].text
    assert "0.879" in claims[2].text and "1.5M" in claims[2].text
    assert claims[3].citation_indices == [8]


def test_number_dense_excerpts_prefers_effect_size_chunks():
    # BM25 order alone favors intro/method prose; the boost must hoist the
    # results-table chunk (denser in numbers) above it for the same paper
    from citens.grounding.chunkstore import ChunkStore
    from citens.models import Chunk, ChunkKind, ExtractedPaper
    from citens.orchestration.pipeline import number_dense_excerpts

    paper = ExtractedPaper(
        id="ignored", title="TALLRec-like study", abstract="abs", year=2024
    )
    store = ChunkStore()
    pid = paper.id  # ExtractedPaper regenerates its id — key the store by it
    store._by_paper[pid] = [
        Chunk(paper_id=pid, chunk_id=f"{pid}-ft-0", kind=ChunkKind.FULLTEXT, text=(
            "We introduce the framework and discuss related work on "
            "recommendation with large language models in detail."
        )),
        Chunk(paper_id=pid, chunk_id=f"{pid}-ft-1", kind=ChunkKind.FULLTEXT, text=(
            "Fine-tuning yields 37.2% Hit@5 versus 24.8% for the zero-shot "
            "baseline; NDCG@10 improves from 0.31 to 0.44 (p<0.05)."
        )),
    ]
    out = number_dense_excerpts([0], [paper], store, query="recommendation")
    assert "37.2%" in out
    assert out.index("37.2%") < out.index("We introduce")


def test_number_dense_excerpts_skips_abstract_only_papers():
    from citens.grounding.chunkstore import ChunkStore
    from citens.models import Chunk, ChunkKind, ExtractedPaper
    from citens.orchestration.pipeline import number_dense_excerpts

    store = ChunkStore()
    store._by_paper["p2"] = [
        Chunk(paper_id="p2", chunk_id="p2-abs", kind=ChunkKind.ABSTRACT, text="abstract only 12.3%")
    ]
    paper = ExtractedPaper(id="p2", title="No fulltext", abstract="abs", year=2024)
    store._by_paper[paper.id] = [
        Chunk(paper_id=paper.id, chunk_id="abs", kind=ChunkKind.ABSTRACT,
              text="abstract only 12.3%")
    ]
    assert number_dense_excerpts([0], [paper], store, query="q") == ""
