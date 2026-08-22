"""The agent-transcript trace: every chat() call reports itself to the hook.

Covers the contract the web console's live feed depends on: start/end/cached
phases, purpose derivation, reasoning capture from hybrid models, stage
tagging, and run filtering (concurrent runs must not cross-stream).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens import llm  # noqa: E402
from citens.events import LLMTrace, StepProgress  # noqa: E402


class _FakeBackend:
    """Records calls; returns a fixed body, optionally with reasoning."""

    def __init__(self, body="ok", reasoning=""):
        self.body = body
        self.reasoning = reasoning
        self.calls = 0

    def chat(self, system_prompt, user_prompt, **kw):
        self.calls += 1
        return self.body


def _resp(body="ok", reasoning=None, prompt=10, completion=5):
    msg = SimpleNamespace(content=body)
    if reasoning is not None:
        msg.reasoning_content = reasoning
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(message=msg)],
    )


def _no_cache(monkeypatch):
    monkeypatch.setattr(llm.cache, "get", lambda ns, key: None)
    monkeypatch.setattr(llm.cache, "put", lambda ns, key, val: None)


def test_chat_fires_start_and_end(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(llm, "get_backend", lambda model=None: _FakeBackend("hello"))
    records: list[dict] = []
    llm.set_trace(records.append)
    try:
        out = llm.chat("SYS FIRST LINE.\nrest", "user prompt", thinking=False)
    finally:
        llm.set_trace(None)
    assert out == "hello"
    phases = [r["phase"] for r in records]
    assert phases == ["start", "end"]
    assert records[0]["model"]
    assert "SYS FIRST LINE" in records[0]["purpose"]
    assert records[0]["thinking"] is False
    assert records[1]["chars_out"] == len("hello")
    assert records[1]["ms"] >= 0


def test_chat_cache_hit_fires_cached(monkeypatch):
    monkeypatch.setattr(llm.cache, "get", lambda ns, key: "cached-text")
    records: list[dict] = []
    llm.set_trace(records.append)
    try:
        assert llm.chat("s", "u") == "cached-text"
    finally:
        llm.set_trace(None)
    assert [r["phase"] for r in records] == ["start", "cached"]


def test_reasoning_captured_from_message(monkeypatch):
    # _chat_with_retry reads reasoning_content off the response message and
    # stashes it in the thread-local the trace hook reads
    out = llm._chat_with_retry(
        lambda: _resp(body="answer", reasoning="thinking hard about this"), "m"
    )
    assert out == "answer"
    assert getattr(llm._tls, "last_reasoning", "") == "thinking hard about this"

    # plain models have no reasoning field -> empty string, never an error
    llm._chat_with_retry(lambda: _resp(body="answer"), "m")
    assert llm._tls.last_reasoning == ""

    # and the end-phase record carries the excerpt through chat(): the
    # backend stashes the reasoning in the thread-local, chat() picks it up
    _no_cache(monkeypatch)

    class _ReasoningBackend:
        def chat(self, *a, **kw):
            llm._tls.last_reasoning = "reasoning excerpt"
            return "answer"

    monkeypatch.setattr(llm, "get_backend", lambda model=None: _ReasoningBackend())
    records: list[dict] = []
    llm.set_trace(records.append)
    try:
        llm.chat("s", "u")
    finally:
        llm.set_trace(None)
    assert records[-1]["reasoning"] == "reasoning excerpt"


def test_stage_tagged_on_records(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(llm, "get_backend", lambda model=None: _FakeBackend())
    records: list[dict] = []
    llm.set_trace(records.append)
    try:
        llm.set_trace_stage("planner")
        llm.chat("s", "u")
    finally:
        llm.set_trace(None)
        llm.set_trace_stage("")
    assert records[0]["stage"] == "planner"
    assert records[1]["stage"] == "planner"


def test_stage_propagates_into_thread_pool(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(llm, "get_backend", lambda model=None: _FakeBackend())
    records: list[dict] = []
    llm.set_trace(records.append)
    try:
        llm.set_trace_stage("extract")
        with llm.run_scope("tracePool"):
            llm.run_concurrent(lambda i, x: llm.chat("s", f"u{i}"), list(range(3)),
                               max_workers=3)
    finally:
        llm.set_trace(None)
        llm.set_trace_stage("")
    ends = [r for r in records if r["phase"] == "end"]
    assert len(ends) == 3
    assert all(r["stage"] == "extract" for r in ends)


def test_llmtrace_event_serializes():
    from citens.api.app import _event_to_dict

    ev = LLMTrace(phase="end", call_id="ab12", model="m", purpose="p",
                  chars_out=10, ms=1500, reasoning="r", stage="verify")
    d = _event_to_dict(ev)
    # the wire contract the UI dispatches on: PascalCase class name (the raw
    # model_dump's lowercase Literal must NOT clobber it — the 1.2.x bug)
    assert d["type"] == "LLMTrace"
    assert d["phase"] == "end"
    assert d["ms"] == 1500


def test_detail_flag_on_step_progress():
    ev = StepProgress(step="filter", message="[1] A paper", detail=True)
    assert ev.model_dump()["detail"] is True


def test_trace_hook_exception_never_breaks_chat(monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(llm, "get_backend", lambda model=None: _FakeBackend("fine"))

    def bad_hook(record):
        raise RuntimeError("hook blew up")

    llm.set_trace(bad_hook)
    try:
        assert llm.chat("s", "u") == "fine"
    finally:
        llm.set_trace(None)
