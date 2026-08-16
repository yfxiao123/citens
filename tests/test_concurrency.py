"""Concurrency + merged-extract behavior for the perf rework.

run_concurrent returns results in input order — the filter log and the
defense claim_idx mapping depend on it. The merged extract must still
produce a validated quality dict from ONE call.
"""

from __future__ import annotations

import threading

from citens.agents import defense, extract
from citens.agents import filter as filter_mod
from citens.llm import run_concurrent
from citens.models import (
    Claim,
    Paper,
    ScoredPaper,
    Verdict,
    VerificationResult,
)


def _papers(n=6):
    return [
        Paper(title=f"Paper {i} on order books", authors=["A Author"],
              year=2020 + i, abstract=f"Abstract {i}")
        for i in range(n)
    ]


def test_filter_concurrent_preserves_order(monkeypatch):
    import re
    import time as _t

    seen_threads = set()

    def fake_chat_json(system, user, **k):
        _t.sleep(0.02)  # force the pool to actually schedule in parallel
        seen_threads.add(threading.current_thread().ident)
        idxs = [int(m) for m in re.findall(r"--- Paper (\d+) ---", user)]
        return {
            "results": [
                {"paper_index": i, "score": 5 - (i % 3), "reason": f"reason {i}"}
                for i in idxs
            ]
        }

    monkeypatch.setattr(filter_mod, "chat_json", fake_chat_json)
    papers = _papers(18)  # 3 batches at the default batch size of 8
    passed, log = filter_mod.filter_papers(papers, "order books", return_log=True)

    assert log[0]["title"] == "Paper 0 on order books"  # order preserved
    assert [e["title"] for e in log] == [p.title for p in papers]
    assert log[0]["score"] == 5 and log[2]["score"] == 3
    assert len(passed) == sum(1 for e in log if e["passed"])
    assert len(seen_threads) > 1 or len(papers) == 1  # actually ran parallel


def test_filter_progress_reports_done_counts(monkeypatch):
    monkeypatch.setattr(
        filter_mod, "chat_json",
        lambda s, u, **k: {"results": [{"paper_index": 0, "score": 4, "reason": "ok"}]},
    )
    calls = []
    filter_mod.filter_papers(
        _papers(4), "t", on_progress=lambda done, total, title: calls.append((done, total))
    )
    # progress fires once per BATCH, monotonically reaching the total
    dones = [d for d, _ in calls]
    assert dones == sorted(dones) and dones[-1] == 4
    assert all(t == 4 for _, t in calls)


def test_filter_small_budget_passed_through(monkeypatch):
    budgets = []
    monkeypatch.setattr(
        filter_mod, "chat_json",
        lambda s, u, **k: budgets.append(k.get("max_tokens"))
        or {"results": [{"paper_index": 0, "score": 3, "reason": ""}]},
    )
    filter_mod.filter_papers(_papers(2), "t")
    # 4096 for the batch call; 1024 when a paper falls back to per-paper
    assert budgets and all(b in (1024, 4096) for b in budgets)
    assert 4096 in budgets


def test_extract_merged_quality_single_call(monkeypatch):
    calls = []

    def fake_chat_json(system, user, **k):
        calls.append(system)
        return {
            "papers": [
                {
                    "paper_index": 0,
                    "research_question": "q", "methodology": "m",
                    "key_findings": ["f"], "limitations": [],
                    "relevance_to_topic": "r",
                    "study_type": "empirical", "evidence_level": 3,
                    "method_rigor": 4, "sample_or_data": "5 stocks",
                    "effect_direction": "positive",
                    "temporal_scope": "2019-2021", "quality_note": "fine",
                }
            ]
        }

    monkeypatch.setattr(extract, "chat_json", fake_chat_json)
    scored = [ScoredPaper(**p.model_dump(exclude={"id"}), relevance_score=4) for p in _papers(1)]
    out = extract.extract_papers(scored, "t")
    assert len(calls) == 1  # one call per batch, quality folded in
    assert out[0].quality["evidence_level"] == 3
    assert out[0].quality["study_type"] == "empirical"
    assert out[0].quality["method_rigor"] == 4


def test_extract_quality_clamps_bad_values(monkeypatch):
    monkeypatch.setattr(
        extract, "chat_json",
        lambda s, u, **k: {
            "research_question": "q", "methodology": "m", "key_findings": [],
            "study_type": "nonsense", "evidence_level": 99, "method_rigor": "high",
        },
    )
    scored = [ScoredPaper(**p.model_dump(exclude={"id"}), relevance_score=4) for p in _papers(1)]
    out = extract.extract_papers(scored, "t")
    assert out[0].quality["study_type"] == "other"
    assert out[0].quality["evidence_level"] == 4
    assert out[0].quality["method_rigor"] == 3


def test_defense_concurrent_preserves_claim_idx(monkeypatch):
    def fake_challenge(claim, verdict, context):
        return {"score": 5 if "win" in claim.text else 1,
                "rebuttal": "r" if "win" in claim.text else "", "concede": "win" not in claim.text}

    monkeypatch.setattr(defense, "challenge_verdict", fake_challenge)
    claims = [Claim(text=f"claim {i} {'win' if i % 2 else 'lose'}", citation_indices=[0])
              for i in range(6)]
    ver_results = [
        VerificationResult(claim_text=c.text, verdict=Verdict.UNSUPPORTED)
        for c in claims
    ]
    reviews = defense.review_unsupported_claims(claims, ver_results, {})
    # claim_idx maps back to the ORIGINAL claim list positions
    assert [r["claim_idx"] for r in reviews] == list(range(6))
    overturned = {r["claim_idx"] for r in reviews if r["overturned"]}
    assert overturned == {1, 3, 5}


def test_run_concurrent_order_under_real_parallelism():
    import time as _t

    def fn(i, item):
        _t.sleep((len(items) - i) * 0.01)  # later items finish first
        return item * 10

    items = list(range(12))
    assert run_concurrent(fn, items, max_workers=6) == [i * 10 for i in items]
