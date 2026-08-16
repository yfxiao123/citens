"""Metric collection + rendering for the eval harness."""

from __future__ import annotations

import json
from pathlib import Path


def collect_metrics(run_dir: str | Path) -> dict:
    """Pull the headline numbers out of a run directory.

    Reads verification.json (verdict counts + precision) and meta.json
    (topic, paper counts). Missing files yield zeroed metrics — a run that
    died before verification should not crash the whole eval sweep.
    """
    d = Path(run_dir)
    metrics: dict = {
        "run_id": d.name,
        "topic": "",
        "papers": 0,
        "claims": 0,
        "verifiable": 0,
        "supported": 0,
        "partial": 0,
        "unsupported": 0,
        "unverifiable": 0,
        "fulltext_papers": 0,
        "precision": None,
    }

    meta_file = d / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        metrics["topic"] = meta.get("topic", "")
        metrics["papers"] = meta.get("filtered_papers", 0)

    if not metrics["papers"] and (d / "references.bib").exists():
        # meta.json is written at finalize; a run that died late (or whose
        # meta wasn't persisted) still has the bibliography to count.
        bib = d / "references.bib"
        metrics["papers"] = sum(
            1 for ln in bib.read_text(encoding="utf-8").splitlines()
            if ln.startswith("@article{")
        )
        if not metrics["topic"]:
            # run dirs are <topic>-<timestamp>; strip the trailing timestamp
            metrics["topic"] = d.name.rsplit("-", 1)[0] or d.name

    ver_file = d / "verification.json"
    if ver_file.exists():
        ver = json.loads(ver_file.read_text(encoding="utf-8"))
        for k in ("total_claims", "verifiable_claims", "supported", "partial",
                  "unsupported", "unverifiable"):
            metrics[k.replace("_claims", "").replace("total", "claims")] = ver.get(k, 0)
        metrics["precision"] = ver.get("citation_precision")

    ground_file = d / "grounding.json"
    if ground_file.exists():
        ground = json.loads(ground_file.read_text(encoding="utf-8"))
        metrics["fulltext_papers"] = ground.get("with_fulltext", 0)

    return metrics


def render_table(rows: list[dict]) -> str:
    """Render collected metrics as a GitHub-flavored markdown table."""
    header = [
        "topic", "papers", "claims", "supported", "partial",
        "unsupported", "unverifiable", "fulltext", "precision",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        cells = [
            str(r.get("topic") or r.get("run_id", "?")),
            str(r.get("papers", 0)),
            str(r.get("claims", 0)),
            str(r.get("supported", 0)),
            str(r.get("partial", 0)),
            str(r.get("unsupported", 0)),
            str(r.get("unverifiable", 0)),
            str(r.get("fulltext_papers", 0)),
            f"{r['precision']:.1%}" if r.get("precision") is not None else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep(run_dirs: list[str | Path]) -> list[dict]:
    """Collect metrics for many run directories, sorted by name."""
    return [collect_metrics(d) for d in sorted(run_dirs, key=str)]
