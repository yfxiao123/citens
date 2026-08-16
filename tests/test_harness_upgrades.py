"""Tests for the harness-style upgrades: run log + usage attribution,
pluggable retrievers, write-time evidence grounding, and the audit loop."""

from __future__ import annotations

import json
import time
from pathlib import Path

from citens.agents import writer as writer_mod
from citens.audit import generate_audit_sheet, ingest_audit
from citens.grounding.retrieval import BM25Retriever, KeywordRetriever
from citens.llm import record_usage
from citens.models import Chunk, ChunkKind, ExtractedPaper, ThemeInfo, ThemeStructure
from citens.runlog import RunLog

# --- RunLog ----------------------------------------------------------------


def test_runlog_append_and_finalize(tmp_path: Path):
    log = RunLog(str(tmp_path))
    log.append("run_start", topic="t")
    time.sleep(0.01)
    log.mark("stage_a")
    time.sleep(0.01)
    record_usage("m1", 100, 50)  # ts falls inside stage_a
    log.mark("stage_b")
    payload = log.finalize()

    events = RunLog.read(log.path)
    kinds = [e["kind"] for e in events]
    assert kinds == ["run_start", "stage", "stage", "run_end"]

    usage = payload["token_usage_by_stage"]
    assert usage["stage_a"] == {"calls": 1, "prompt": 100, "completion": 50}
    assert payload["total_tokens"] == 150


# --- retrievers -------------------------------------------------------------


def _chunks(*texts: str) -> list[Chunk]:
    return [
        Chunk(paper_id="p", chunk_id=f"c{i}", text=t, kind=ChunkKind.FULLTEXT)
        for i, t in enumerate(texts)
    ]


def test_bm25_ranks_relevant_chunk_first():
    chunks = _chunks(
        "We study the mating habits of Antarctic penguins in detail.",
        "The gradient boosting model improves out-of-sample R2 by 15% versus OLS.",
        "Cooking techniques for sourdough bread hydration levels.",
    )
    out = BM25Retriever().rank(chunks, "gradient boosting out-of-sample prediction", k=2)
    assert out[0].text.startswith("The gradient")


def test_keyword_retriever_still_available():
    chunks = _chunks("alpha beta gamma", "delta epsilon zeta alpha")
    out = KeywordRetriever().rank(chunks, "zeta alpha", k=1)
    assert "zeta" in out[0].text


def test_embedding_retriever_falls_back_without_model(monkeypatch):
    from citens.grounding.retrieval import EmbeddingRetriever

    monkeypatch.setattr("citens.grounding.retrieval.settings.embedding_model", "")
    chunks = _chunks("penguins antarctica", "stock return prediction machine learning")
    out = EmbeddingRetriever().rank(chunks, "machine learning stock returns", k=1)
    assert "stock return" in out[0].text  # bm25 fallback did the ranking


# --- write-time evidence ------------------------------------------------------


def test_write_review_body_includes_evidence_block(monkeypatch):
    prompts = []

    def fake_chat(system, user, max_tokens=None, strong=False, **k):
        prompts.append((system, user))
        return "Section body text with a claim [0]. It ends properly."

    monkeypatch.setattr(writer_mod, "chat", fake_chat)
    papers = [
        ExtractedPaper(
            title="Paper A", authors=["A"], year=2020, abstract="abs",
            relevance_score=4,
        )
    ]
    themes = ThemeStructure(
        themes=[ThemeInfo(name="T1", description="d", paper_indices=[0])]
    )

    def evidence_for(theme):
        return "[0] Paper A — the model improves R2 by 15 percent"

    writer_mod.write_review_body(
        papers, themes, "topic", evidence_for=evidence_for
    )
    assert any("全文证据摘录" in u and "15 percent" in u for _s, u in prompts)
    assert any("EVIDENCE RULE" in s for s, _u in prompts)


# --- audit loop ---------------------------------------------------------------


def _make_run(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "references.bib").write_text(
        "@article{a,\n  title = {Paper One},\n  journal = {J},\n  year = {2020}\n}\n"
        "@article{b,\n  title = {Paper Two},\n  journal = {J},\n  year = {2021}\n}\n",
        encoding="utf-8",
    )
    (run / "verification.json").write_text(
        json.dumps(
            {
                "citation_precision": 0.8,
                "total_claims": 3,
                "supported": 2,
                "partial": 1,
                "unsupported": 0,
                "unverifiable": 0,
                "results": [
                    {"claim_text": "c1 [0]", "verdict": "supported", "citation_indices": [0]},
                    {"claim_text": "c2 [1]", "verdict": "supported", "citation_indices": [1]},
                    {"claim_text": "c3 [0][1]", "verdict": "partial", "citation_indices": [0, 1]},
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_audit_generate_and_ingest_roundtrip(tmp_path: Path):
    run = _make_run(tmp_path)
    sheet = generate_audit_sheet(str(run))
    text = Path(sheet).read_text(encoding="utf-8")
    assert "论断 1" in text and "[0] Paper One" in text
    assert "人工判定: ____" in text

    filled = text.replace("论断 1 — 机器判定: supported", "论断 1 — 机器判定: supported", 1)
    # judge: claim1 supported (agree), claim2 partial (machine lenient), claim3 partial (agree)
    filled = filled.replace("人工判定: ____", "人工判定: s", 1)
    filled = filled.replace("人工判定: ____", "人工判定: p", 1)
    filled = filled.replace("人工判定: ____", "人工判定: p", 1)
    assert filled.count("人工判定: ____") == 0
    Path(sheet).write_text(filled, encoding="utf-8")

    report = ingest_audit(str(run), sheet)
    assert report["judged"] == 3
    assert report["agreement_rate"] == round(2 / 3, 3)
    assert report["machine_lenient"] == 1  # supported judged partial by human
    assert report["human_grounded_rate"] == 1.0
    assert (run / "audit_result.json").is_file()


def test_audit_ingest_rejects_empty_sheet(tmp_path: Path):
    run = _make_run(tmp_path)
    sheet = generate_audit_sheet(str(run))
    import pytest

    with pytest.raises(ValueError):
        ingest_audit(str(run), sheet)
