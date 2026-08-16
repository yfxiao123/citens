"""Append-only run log: one JSONL stream per run, the single substrate for
observability, resume, replay, and audit.

Design borrowed from agent-harness event-sourcing: instead of per-stage files
that overwrite each other (which lost supplement rounds and the final paper
set), every stage boundary, data snapshot, and LLM-usage attribution is
appended to ``run.log``. Consumers derive views from the log; the log itself
is never rewritten.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    """Append-only JSONL event log for one pipeline run."""

    def __init__(self, run_dir: str) -> None:
        self.path = Path(run_dir) / "run.log"
        self._marks: list[tuple[str, float]] = [("run_start", time.time())]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing -------------------------------------------------------------

    def append(self, kind: str, **data) -> None:
        """Append one event line. Never raises — logging must not kill a run."""
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"ts": time.time(), "kind": kind, **data}, ensure_ascii=False)
                    + "\n"
                )
        except Exception:  # noqa: BLE001
            pass

    def mark(self, stage: str) -> None:
        """Record a stage boundary (durations are deltas between marks)."""
        self._marks.append((stage, time.time()))
        self.append("stage", stage=stage)

    def snapshot(self, what: str, **data) -> None:
        self.append("snapshot", what=what, **data)

    # -- reading / derivation --------------------------------------------------

    def stage_windows(self) -> list[tuple[str, float, float]]:
        """[(stage, start_ts, end_ts)] in mark order."""
        return [
            (name, t0, t1)
            for (name, t0), (_, t1) in zip(self._marks, self._marks[1:], strict=False)
        ]

    def token_usage_by_stage(self) -> dict[str, dict[str, int]]:
        """Attribute LLM token usage to stages by call timestamp.

        Timestamp-based attribution: correct for the (typical) one-run-per-
        process case; interleaved runs in one process may cross-attribute a
        few calls — acceptable for cost telemetry, documented tradeoff.
        """
        from citens.llm import usage_records

        windows = self.stage_windows()
        out: dict[str, dict[str, int]] = {}
        for rec in usage_records():
            ts = rec["ts"]
            # inclusive bounds: a call and the next stage mark can share one
            # clock tick; the earliest matching window wins (deterministic)
            for name, t0, t1 in windows:
                if t0 <= ts <= t1:
                    bucket = out.setdefault(name, {"calls": 0, "prompt": 0, "completion": 0})
                    bucket["calls"] += 1
                    bucket["prompt"] += rec.get("prompt", 0)
                    bucket["completion"] += rec.get("completion", 0)
                    break
        return out

    def finalize(self) -> dict:
        """Summarize durations + token usage; append the summary event."""
        now = time.time()
        self._marks.append(("run_end", now))
        usage = self.token_usage_by_stage()
        total_tokens = sum(b["prompt"] + b["completion"] for b in usage.values())
        payload = {
            "total_seconds": round(now - self._marks[0][1], 1),
            "token_usage_by_stage": usage,
            "total_tokens": total_tokens,
        }
        self.append("run_end", **payload)
        return payload

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        """Read a run.log into events (for replay / audit tooling)."""
        events = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events
