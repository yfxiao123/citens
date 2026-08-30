"""The polling event log: /run/start + /run/events (the UI's transport).

The web console polls instead of holding an SSE stream because proxies and
AV software buffer text/event-stream (observed zero frames for minutes while
the run progressed server-side). These tests pin the contract the UI leans
on: start returns an id immediately, events carry a seq cursor, after=0
replays the whole transcript, done flips on completion, and the buffer GC
keeps the dict bounded.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from citens.api import app as app_mod  # noqa: E402
from citens.config import settings  # noqa: E402
from citens.events import RunCompleted, RunFailed, RunStarted, StepProgress  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "api_token", "")
    from citens.api.app import app

    with TestClient(app) as c:
        yield c


def _fake_pipeline(events):
    """run_pipeline_async replacement that replays fixed events."""

    async def fake(topic, options, bus=None):
        for ev in events:
            if bus is not None:
                bus.emit(ev)
        return None

    return fake


def test_start_returns_id_and_events_replay(client, monkeypatch):
    events = [
        RunStarted(topic="t"),
        StepProgress(step="search", message="42 篇候选", detail=True),
        RunCompleted(run_dir="runs/x", summary={"topic": "t"}),
    ]
    monkeypatch.setattr(app_mod, "run_pipeline_async", _fake_pipeline(events))
    r = client.post("/run/start", json={"topic": "测试主题"})
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert len(run_id) == 12

    # wait for the background thread to finish appending
    for _ in range(100):
        j = client.get(f"/run/events/{run_id}").json()
        if j["done"]:
            break
        time.sleep(0.05)
    assert j["done"] is True
    kinds = [rec["event"]["type"] for rec in j["events"]]
    assert kinds == ["RunStarted", "StepProgress", "RunCompleted"]
    assert [rec["seq"] for rec in j["events"]] == [0, 1, 2]
    assert j["events"][1]["event"]["detail"] is True


def test_after_cursor_returns_only_increment(client, monkeypatch):
    events = [StepProgress(step="s", message=f"m{i}") for i in range(4)] + [
        RunCompleted(run_dir="d")
    ]
    monkeypatch.setattr(app_mod, "run_pipeline_async", _fake_pipeline(events))
    run_id = client.post("/run/start", json={"topic": "tt"}).json()["run_id"]
    for _ in range(100):
        j = client.get(f"/run/events/{run_id}").json()
        if j["done"]:
            break
        time.sleep(0.05)

    tail = client.get(f"/run/events/{run_id}?after=4").json()
    assert [rec["seq"] for rec in tail["events"]] == [4]
    assert tail["events"][0]["event"]["type"] == "RunCompleted"


def test_failed_run_marks_done_with_error(client, monkeypatch):
    async def boom(topic, options, bus=None):
        bus.emit(RunStarted(topic=topic))
        bus.emit(RunFailed(message="exploded", step="verify"))
        raise RuntimeError("exploded")

    monkeypatch.setattr(app_mod, "run_pipeline_async", boom)
    run_id = client.post("/run/start", json={"topic": "tt"}).json()["run_id"]
    for _ in range(100):
        j = client.get(f"/run/events/{run_id}").json()
        if j["done"]:
            break
        time.sleep(0.05)
    assert j["done"] is True
    assert j["error"] == "exploded"
    # the pre-exception events are still in the log (replayable)
    assert j["events"][0]["event"]["type"] == "RunStarted"


def test_unknown_run_id_is_404(client):
    assert client.get("/run/events/deadbeef0000").status_code == 404


def test_bad_mode_rejected(client):
    r = client.post("/run/start", json={"topic": "tt", "mode": "bogus"})
    assert r.status_code == 400


def test_buffer_gc_bounded(client, monkeypatch):
    monkeypatch.setattr(app_mod, "run_pipeline_async", _fake_pipeline([RunCompleted()]))
    monkeypatch.setattr(app_mod, "_RUN_BUFFER_MAX", 2)
    ids = [
        client.post("/run/start", json={"topic": f"t{i}"}).json()["run_id"]
        for i in range(4)
    ]
    # oldest logs evicted once the cap is exceeded
    assert client.get(f"/run/events/{ids[0]}").status_code == 404
    assert client.get(f"/run/events/{ids[-1]}").status_code == 200
    assert len(app_mod._RUN_BUFFERS) <= 2


def test_cancel_stops_run_and_marks_done(client, monkeypatch):
    """The cancel contract: cancel flips the flag; the pipeline's next event
    emission raises, and the log closes with a visible cancelled RunFailed —
    never a silent hang (the v1.3.3 exe's failure mode)."""
    started = threading.Event()

    def endless_pipeline(topic, options, bus=None):
        bus.emit(RunStarted(topic=topic))
        started.set()
        for _ in range(200):  # a long stage: no events for a while
            time.sleep(0.05)
        bus.emit(RunCompleted(run_dir="runs/x", summary={}))

    monkeypatch.setattr(app_mod, "run_pipeline_async", endless_pipeline)
    run_id = client.post("/run/start", json={"topic": "长任务"}).json()["run_id"]
    assert started.wait(timeout=5)

    r = client.post(f"/run/cancel/{run_id}")
    assert r.status_code == 200 and r.json()["cancelled"] is True

    # cooperative: lands at the NEXT event boundary (the fake's next emit)
    j = client.get(f"/run/events/{run_id}").json()
    for _ in range(400):
        j = client.get(f"/run/events/{run_id}").json()
        if j["done"]:
            break
        time.sleep(0.05)
    assert j["done"] is True
    assert j["error"] == "已被用户中断 / cancelled"
    # RunStarted is still in the log (the cancel only ends future stages)
    assert j["events"][0]["event"]["type"] == "RunStarted"


def test_cancel_unknown_run_404_and_finished_run_noop(client, monkeypatch):
    assert client.post("/run/cancel/deadbeef0000").status_code == 404
    monkeypatch.setattr(app_mod, "run_pipeline_async", _fake_pipeline([RunCompleted()]))
    run_id = client.post("/run/start", json={"topic": "秒完"}).json()["run_id"]
    for _ in range(50):
        if client.get(f"/run/events/{run_id}").json()["done"]:
            break
        time.sleep(0.05)
    r = client.post(f"/run/cancel/{run_id}").json()
    assert r["cancelled"] is False and "finished" in r["reason"]
