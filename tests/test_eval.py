"""Eval harness offline parts: metric collection + table rendering."""

from __future__ import annotations

import json

from citens.eval import collect_metrics, render_table


def _make_run(tmp_path, *, with_verification=True, with_meta=True):
    d = tmp_path / "topic-20260101_000000"
    d.mkdir()
    if with_meta:
        (d / "meta.json").write_text(
            json.dumps({"topic": "order books", "filtered_papers": 13}), encoding="utf-8"
        )
    if with_verification:
        (d / "verification.json").write_text(
            json.dumps(
                {
                    "citation_precision": 0.937,
                    "total_claims": 65,
                    "verifiable_claims": 63,
                    "supported": 36,
                    "partial": 23,
                    "unsupported": 4,
                    "unverifiable": 2,
                }
            ),
            encoding="utf-8",
        )
    (d / "grounding.json").write_text(
        json.dumps({"with_fulltext": 3, "total": 13}), encoding="utf-8"
    )
    return d


def test_collect_metrics_full_run(tmp_path):
    m = collect_metrics(_make_run(tmp_path))
    assert m["topic"] == "order books"
    assert m["papers"] == 13
    assert m["claims"] == 65
    assert m["supported"] == 36
    assert m["precision"] == 0.937
    assert m["fulltext_papers"] == 3
    assert m["run_id"].startswith("topic-")


def test_collect_metrics_survives_partial_run(tmp_path):
    """A run that died before verification must not crash the sweep."""
    m = collect_metrics(_make_run(tmp_path, with_verification=False))
    assert m["precision"] is None
    assert m["papers"] == 13
    assert m["claims"] == 0


def test_collect_metrics_empty_dir(tmp_path):
    m = collect_metrics(tmp_path / "nope")
    assert m["precision"] is None
    assert m["topic"] == ""


def test_render_table_format():
    rows = [
        {
            "topic": "order books", "papers": 13, "claims": 65, "supported": 36,
            "partial": 23, "unsupported": 4, "unverifiable": 2,
            "fulltext_papers": 3, "precision": 0.937,
        },
        {
            "topic": "rag", "papers": 10, "claims": 40, "supported": 20,
            "partial": 10, "unsupported": 5, "unverifiable": 5,
            "fulltext_papers": 0, "precision": None,
        },
    ]
    md = render_table(rows)
    assert md.count("\n") == 3  # header + rule + 2 rows... (2 rows -> 3 newlines)
    assert "93.7%" in md
    assert "—" in md  # missing precision rendered as dash
    assert md.splitlines()[0].startswith("| topic")
