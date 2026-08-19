"""Usage records tag with run scope; RunLog reads only its own run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens import llm  # noqa: E402


def test_run_scope_tags_records():
    llm.record_usage("m", 10, 5)  # untagged
    with llm.run_scope("runA"):
        llm.record_usage("m", 1, 1)
    with llm.run_scope("runB"):
        llm.record_usage("m", 2, 2)
    a = llm.usage_records("runA")
    assert all(r.get("run") in (None, "runA") for r in a)
    assert all(r.get("run") in (None, "runB") for r in llm.usage_records("runB"))
    assert "runA" not in {r.get("run") for r in llm.usage_records("runB")}


def test_run_scope_propagates_into_thread_pool():
    def job(i, item):
        llm.record_usage("m", i, 0)

    with llm.run_scope("poolRun"):
        llm.run_concurrent(job, list(range(4)), max_workers=4)
    recs = llm.usage_records("poolRun")
    tagged = [r for r in recs if r.get("run") == "poolRun"]
    assert len(tagged) == 4
