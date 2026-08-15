"""Resume + reverify wiring: interrupted runs continue; new PDFs re-verify.

Both are the "trust deepening" paths — they must never re-run retrieval or
writing, and must fail loudly when the run dir lacks the required artifacts.
"""

from __future__ import annotations

import json

import pytest

from citens.models import ExtractedPaper, RunMode
from citens.orchestration.pipeline import (
    RunOptions,
    _load_extracted_for_resume,
    _mode_from_run_dir,
    run_pipeline_async,
)
from citens.orchestration.reverify import reverify


def _extracted(n=2):
    return [
        ExtractedPaper(
            title=f"Paper {i}", authors=["A Author"], year=2020 + i,
            abstract=f"Abstract {i} with findings.", relevance_score=4,
        )
        for i in range(n)
    ]


def _make_run(tmp_path, *, with_extracted=True):
    d = tmp_path / "topic-20260101_000000"
    (d / "steps").mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"topic": "test topic"}), encoding="utf-8")
    if with_extracted:
        (d / "steps" / "04_extracted.json").write_text(
            json.dumps([p.model_dump() for p in _extracted()]), encoding="utf-8"
        )
    return d


# --- resume: state loading ---------------------------------------------------


def test_load_extracted_for_resume(tmp_path):
    d = _make_run(tmp_path)
    papers = _load_extracted_for_resume(str(d))
    assert len(papers) == 2
    assert papers[0].title == "Paper 0"


def test_load_extracted_missing_raises(tmp_path):
    d = _make_run(tmp_path, with_extracted=False)
    with pytest.raises(FileNotFoundError, match="04_extracted"):
        _load_extracted_for_resume(str(d))


def test_mode_from_run_dir_defaults_deep(tmp_path):
    d = _make_run(tmp_path)
    assert _mode_from_run_dir(str(d)) == RunMode.DEEP_REVIEW
    (d / "steps" / "00_intent.json").write_text(
        json.dumps({"mode": "quick_scan"}), encoding="utf-8"
    )
    assert _mode_from_run_dir(str(d)) == RunMode.QUICK_SCAN


# --- resume: pipeline skips retrieval ----------------------------------------


@pytest.mark.asyncio
async def test_resume_skips_retrieval_and_composes(tmp_path, monkeypatch):
    """If retrieval so much as touches the network, the test fails."""
    d = _make_run(tmp_path)

    async def _no_network(*a, **k):  # pragma: no cover - must not be called
        raise AssertionError("resume must not search")

    monkeypatch.setattr("citens.orchestration.pipeline.search_papers", _no_network)
    monkeypatch.setattr(
        "citens.orchestration.pipeline.generate_keywords",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no planner on resume")),
    )

    from citens.grounding import CitationTable
    from citens.models import SynthesisResult
    from citens.orchestration import pipeline as pl

    def _fake_compose(extracted, topic, run_dir, bus, **k):
        return pl.ComposeResult(
            themes=type("T", (), {"themes": []})(),
            review_path=f"{run_dir}/review.md",
            table=CitationTable(extracted),
            claims=[],
            precision=0.5,
            synthesis=SynthesisResult(),
        )

    monkeypatch.setattr(pl, "_compose", _fake_compose)
    meta = await run_pipeline_async(
        "test topic", RunOptions(resume_dir=str(d), allow_supplement=False)
    )
    assert meta.run_dir == str(d)
    assert meta.citation_precision == 0.5


# --- reverify ----------------------------------------------------------------


def test_reverify_updates_artifacts(tmp_path, monkeypatch):
    d = _make_run(tmp_path)
    (d / "review.md").write_text(
        "# Topic\n\n## Introduction\n\nA claim citing paper [0].\n",
        encoding="utf-8",
    )
    (d / "verification.json").write_text(
        json.dumps({"citation_precision": 0.25, "results": []}), encoding="utf-8"
    )
    (d / "meta.json").write_text(
        json.dumps({"topic": "test topic", "citation_precision": 0.25}),
        encoding="utf-8",
    )

    from citens.models import Verdict, VerificationResult

    def _fake_verify(claims, table, chunk_store, on_progress=None):
        results = [
            VerificationResult(
                claim_text=c.text, verdict=Verdict.SUPPORTED, note="ok"
            )
            for c in claims
        ]
        return results, 1.0

    monkeypatch.setattr(
        "citens.orchestration.reverify.verify_claims", _fake_verify
    )

    summary = reverify(str(d))
    assert summary["claims"] == 1
    assert summary["precision"] == 1.0
    assert summary["previous_precision"] == 0.25

    ver = json.loads((d / "verification.json").read_text(encoding="utf-8"))
    assert ver["citation_precision"] == 1.0
    assert ver["previous_precision"] == 0.25
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    assert prov[0]["verdict"] == "supported"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    assert meta["citation_precision"] == 1.0


def test_reverify_missing_review_raises(tmp_path):
    d = _make_run(tmp_path)  # has 04_extracted but no review.md
    with pytest.raises(FileNotFoundError, match="review.md"):
        reverify(str(d))
