"""Verifier-calibration regression suite.

Born from the 2026-08-18 human audit of run 机器学习股票收益预测-20260818_201038:
the machine self-reported 100% citation precision while a strict audit put the
grounded rate at 68% (12 lenient verdicts, 1 strict). Two systematic biases
were isolated — no-abstract papers judged too leniently, and interpretive
framing credited as "supported". These tests pin the fixes so they cannot
silently regress.
"""

from __future__ import annotations

import json
from pathlib import Path

from citens.agents import verifier as verifier_mod
from citens.agents import writer as writer_mod
from citens.agents.health import check_health
from citens.grounding import ChunkStore, CitationTable
from citens.models import (
    Chunk,
    ChunkKind,
    Claim,
    ExtractedPaper,
    Paper,
    SynthesisResult,
    Verdict,
    VerificationResult,
)

GOLDEN = Path(__file__).parent / "golden" / "verifier_calibration_201038.json"


def _p(title, **kw):
    defaults = dict(authors=["A Author"], year=2020, abstract="abs text",
                    citation_count=1)
    defaults.update(kw)
    return Paper(title=title, **defaults)


def _store_with(papers, which):
    store = ChunkStore()
    for i in which:
        pid = papers[i].id
        store._by_paper[pid] = [
            Chunk(paper_id=pid, chunk_id=f"c{i}", text=f"paper {i} abstract",
                  kind=ChunkKind.ABSTRACT)
        ]
    return store


# --- prompt contract: the calibration rules stay in, the leniency tie-break stays out


def test_verifier_prompt_has_calibration_rules():
    for rule in ("NO-GROUND-TEXT RULE", "INTERPRETIVE FRAMING RULE",
                 "MULTI-CITATION RULE", "Output JSON only"):
        assert rule in verifier_mod.SYSTEM_PROMPT, rule


def test_verifier_prompt_leniency_tiebreak_removed():
    # the audited lenient bias was traced to this explicit instruction
    assert 'prefer "supported"' not in verifier_mod.SYSTEM_PROMPT
    assert 'prefer "partial"' not in verifier_mod.SYSTEM_PROMPT
    assert "prefer" not in verifier_mod.SYSTEM_PROMPT  # no leniency directive may remain


def test_no_ground_text_marker_names_the_rule():
    papers = [_p("Invisible paper")]
    table = CitationTable(papers)
    store = ChunkStore()  # no chunks at all
    ctx = verifier_mod._build_context([0], table, store, "query")
    assert "NO GROUND TEXT" in ctx


# --- verdict fallback: a missing judge response is never a free pass


def test_missing_judge_verdict_defaults_to_unverifiable(monkeypatch):
    papers = [_p("Grounded")]
    table = CitationTable(papers)
    store = _store_with(papers, [0])

    def fake_chat_json(system, user, **k):
        return {"results": []}  # judge returned nothing usable

    monkeypatch.setattr(verifier_mod, "chat_json", fake_chat_json)
    claims = [Claim(text="a claim [0]", citation_indices=[0])]
    results, precision = verifier_mod.verify_claims(claims, table, store)
    assert results[0].verdict is Verdict.UNVERIFIABLE
    assert precision == 0.0  # excluded from the denominator, not counted grounded

    def weird_chat_json(system, user, **k):
        return {"results": [{"claim_index": 0, "verdict": "mostly fine",
                             "note": "malformed grade"}]}

    monkeypatch.setattr(verifier_mod, "chat_json", weird_chat_json)
    results, _ = verifier_mod.verify_claims(claims, table, store)
    assert results[0].verdict is Verdict.UNVERIFIABLE


# --- canary honeypot: measures the judge's false-accept rate


def test_canary_check_measures_false_accept_rate(monkeypatch):
    papers = [_p("Real paper"), _p("Another real paper")]
    table = CitationTable(papers)
    store = _store_with(papers, [0, 1])

    monkeypatch.setattr(
        verifier_mod, "chat_json",
        lambda system, user, **k: {"results": [
            {"claim_index": j, "verdict": "supported", "note": "lenient"}
            for j in range(3)
        ]},
    )
    report = verifier_mod.canary_check(table, store)
    assert report["injected"] == 2  # min(papers with ground text, 3 canaries)
    assert report["caught"] == 0
    assert report["false_accept_rate"] == 1.0

    monkeypatch.setattr(
        verifier_mod, "chat_json",
        lambda system, user, **k: {"results": [
            {"claim_index": j, "verdict": "unsupported", "note": "caught"}
            for j in range(3)
        ]},
    )
    report = verifier_mod.canary_check(table, store)
    assert report["caught"] == 2
    assert report["false_accept_rate"] == 0.0


def test_canary_claims_are_never_in_the_real_results(monkeypatch):
    # canaries run in a SEPARATE call: the real verify path must stay clean
    seen_user_prompts = []

    def fake_chat_json(system, user, **k):
        seen_user_prompts.append(user)
        return {"results": [{"claim_index": 0, "verdict": "supported", "note": ""}]}

    papers = [_p("Grounded")]
    table = CitationTable(papers)
    store = _store_with(papers, [0])
    monkeypatch.setattr(verifier_mod, "chat_json", fake_chat_json)

    claims = [Claim(text="real claim [0]", citation_indices=[0])]
    results, _ = verifier_mod.verify_claims(claims, table, store)
    verifier_mod.canary_check(table, store)
    assert len(results) == 1  # only the real claim
    assert any("patient mortality" in u for u in seen_user_prompts[1:])


# --- spot check: per-claim results so the pipeline can ADOPT downgrades


def test_spot_check_returns_per_claim_results(monkeypatch):
    papers = [_p("Grounded")]
    table = CitationTable(papers)
    store = _store_with(papers, [0])
    claims = [Claim(text="overstated claim [0]", citation_indices=[0])]
    ver = [VerificationResult(claim_text=claims[0].text, verdict=Verdict.SUPPORTED)]

    monkeypatch.setattr(
        verifier_mod, "chat_json",
        lambda system, user, **k: {"results": [
            {"claim_index": 0, "verdict": "downgrade", "note": "overstates"}]},
    )
    report = verifier_mod.spot_check_supported(claims, ver, table, store)
    assert report["downgraded"] == 1
    assert report["results"] == [
        {"claim_index": 0, "verdict": "downgrade", "note": "overstates"}
    ]


# --- health: measured leniency surfaces as an issue


def test_health_flags_canary_false_accepts():
    synth = SynthesisResult(consensus=["c"], contradictions=[], gaps=[])
    results = [VerificationResult(claim_text=f"c{i}", verdict=Verdict.SUPPORTED)
               for i in range(5)]
    report = check_health(synth, results, {"absent_canonical_papers": ["x"]}, {},
                          canary={"injected": 3, "caught": 1, "false_accept_rate": 0.667})
    assert "verifier_false_accept" in report["issues"]
    assert report["metrics"]["canary_false_accept_rate"] == 0.667
    assert "calibration is broken" in report["recommendation"]

    clean = check_health(synth, results, {"absent_canonical_papers": ["x"]}, {},
                         canary={"injected": 3, "caught": 3, "false_accept_rate": 0.0})
    assert "verifier_false_accept" not in clean["issues"]


def test_health_reports_unverifiable_rate():
    synth = SynthesisResult(consensus=[], contradictions=[], gaps=[])
    results = [
        VerificationResult(claim_text="a", verdict=Verdict.SUPPORTED),
        VerificationResult(claim_text="b", verdict=Verdict.UNVERIFIABLE),
        VerificationResult(claim_text="c", verdict=Verdict.UNVERIFIABLE),
    ]
    report = check_health(synth, results, {"absent_canonical_papers": []}, {})
    assert report["metrics"]["unverifiable_rate"] == round(2 / 3, 3)


# --- writer: no-abstract papers are marked and fenced off


def test_papers_block_marks_no_abstract_papers():
    ep = ExtractedPaper(title="No abstract paper", abstract="", year=2021)
    block = writer_mod._papers_block([(4, ep)])
    assert "NO ABSTRACT" in block
    ep2 = ExtractedPaper(title="Has abstract", abstract="real abstract", year=2021)
    assert "NO ABSTRACT" not in writer_mod._papers_block([(0, ep2)])


def test_supporting_block_marks_no_abstract_papers():
    papers = [(3, _p("Supporting no-abstract", abstract=""))]
    block = writer_mod._supporting_block(papers)
    assert "NO ABSTRACT" in block and "do not cite" in block


def test_section_prompt_carries_the_rules():
    assert "NO-ABSTRACT papers" in writer_mod.SECTION_PROMPT
    assert "outside the citation brackets" in writer_mod.SECTION_PROMPT


# --- golden set integrity: the audit that motivated all of the above


def test_golden_calibration_set_is_intact():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert data["summary"]["judged"] == 22
    assert data["summary"]["machine_lenient"] == 12
    assert data["summary"]["human_grounded_rate"] == 0.682
    assert len(data["claims"]) == 22
    valid = {"supported", "partial", "unsupported"}
    assert all(c["human"] in valid for c in data["claims"])
    assert all(c["machine"] in {"supported", "partial"} for c in data["claims"])
    # the seven no-abstract downgrades all involve the ground-text-less paper 4
    downgrades = [c for c in data["claims"] if c["machine"] == "partial"
                  and c["human"] == "unsupported"]
    assert len(downgrades) == 7


def test_fuzzy_verdict_reuse_and_grounding_invalidation():
    # a reworded restatement with the same citations reuses the earlier
    # verdict; the same restatement against CHANGED ground text (the paper
    # gained fulltext between rounds) must NOT reuse the stale verdict
    from citens.agents.verifier import _fuzzy_lookup, _norm_claim_text
    from citens.grounding.chunkstore import ChunkStore
    from citens.grounding.citations import CitationTable
    from citens.models import Chunk, ChunkKind, Claim, Paper, Verdict, VerificationResult

    paper = Paper(id="p1", title="T", abstract="abs", year=2024)
    table = CitationTable([paper])
    store = ChunkStore()
    pid = table.paper_id(0)  # Paper regenerates its id — key by the table's
    store._by_paper[pid] = [
        Chunk(paper_id=pid, chunk_id="abs", kind=ChunkKind.ABSTRACT, text="abs")
    ]
    claim = Claim(text="TALLRec仅需128条样本即可改善推荐性能", citation_indices=[0])
    result = VerificationResult(
        claim_text=claim.text, verdict=Verdict.SUPPORTED,
        citation_indices=[0], note="",
    )

    def sig():
        from citens.agents.verifier import _grounding_sig
        return _grounding_sig(claim, table, store)

    fuzzy = {(_norm_claim_text(claim.text), (0,)): (sig(), result)}

    # exact normalized restatement (whitespace/punct noise) -> reuse
    hit = _fuzzy_lookup(
        Claim(text="TALLRec 仅需 128 条样本，即可改善推荐性能。", citation_indices=[0]),
        table, store, fuzzy,
    )
    assert hit is not None and hit.verdict == Verdict.SUPPORTED

    # different citations -> no reuse
    miss = _fuzzy_lookup(
        Claim(text=claim.text, citation_indices=[0, 1]), table, store, fuzzy
    )
    assert miss is None

    # same claim, but the paper's ground text changed -> no reuse
    store._by_paper[pid] = [
        Chunk(paper_id=pid, chunk_id="abs", kind=ChunkKind.ABSTRACT, text="abs"),
        Chunk(paper_id=pid, chunk_id="ft", kind=ChunkKind.FULLTEXT,
              text="full text arrived between rounds " * 5),
    ]
    miss2 = _fuzzy_lookup(claim, table, store, fuzzy)
    assert miss2 is None


def test_reasoning_effort_levels_map_to_payload():
    # "low" keeps a short deliberation; False/"none" kills it entirely;
    # True leaves the provider default (no extra_body)
    from citens.llm import build_completion_kwargs

    kw = dict(system_prompt="s", user_prompt="u", temperature=0.3,
              max_tokens=128, response_json=False)
    low = build_completion_kwargs("m", thinking="low", **kw)
    assert low.get("extra_body") == {"reasoning_effort": "low"}
    none_off = build_completion_kwargs("m", thinking=False, **kw)
    assert none_off.get("extra_body") == {"reasoning_effort": "none"}
    default = build_completion_kwargs("m", thinking=True, **kw)
    assert "extra_body" not in default


def test_align_near_duplicates_downgrades_conflicting_twins():
    """Audit 2026-08-30: one DCMAB restatement "supported", its near-duplicate
    twin "unsupported" — judge noise presented as a hard failure."""
    from citens.agents.verifier import Verdict, VerificationResult, _align_near_duplicates
    from citens.models import Claim

    twins = [
        VerificationResult(
            claim_text="[4]提出了分布式协调多代理出价方法，实验显示其整体目标优于基线",
            verdict=Verdict.SUPPORTED, citation_indices=[4],
        ),
        VerificationResult(
            claim_text="[4]提出了分布式协调多代理出价方法，实验显示整体目标优于基线方法",
            verdict=Verdict.UNSUPPORTED, citation_indices=[4],
        ),
        VerificationResult(  # different citations — must NOT be grouped
            claim_text="[4]提出了分布式协调多代理出价方法，实验显示整体目标优于基线方法",
            verdict=Verdict.SUPPORTED, citation_indices=[5],
        ),
        VerificationResult(  # contradictory stays untouched even in a twin group
            claim_text="[4]提出了分布式协调多代理出价方法，实验显示整体目标优于基线",
            verdict=Verdict.CONTRADICTORY, citation_indices=[4],
        ),
    ]
    claims = [Claim(text=r.claim_text, citation_indices=r.citation_indices,
                    section="s") for r in twins]
    out = _align_near_duplicates(twins, claims)
    by_text = {r.claim_text[:12]: r for r in out}
    aligned = [r for r in out if "aligned" in (r.note or "")]
    assert len(aligned) == 2  # the two same-citation twins aligned...
    assert all(r.verdict == Verdict.PARTIAL for r in aligned)
    # the different-citation claim keeps its supported verdict untouched
    lone = [r for r in out if r not in aligned and r.verdict == Verdict.SUPPORTED]
    assert len(lone) == 1
    # contradictory verdicts are never aligned away
    assert any(r.verdict == Verdict.CONTRADICTORY for r in out)
