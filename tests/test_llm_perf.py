"""Tests for the speed layer: LLM response cache, two-tier model routing,
run_concurrent ordering. All offline — the backend is faked."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import citens.llm as llm  # noqa: E402
from citens.config import settings  # noqa: E402


class FakeBackend:
    def __init__(self) -> None:
        self.model = ""
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def chat(self, system, user, *, temperature=0.3, max_tokens=None, response_json=False, thinking=True):
        with self.lock:
            self.calls.append((self.model, user[:20]))
        return f"ok:{user}"


def _fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cache_enabled", True)


def test_chat_caches_repeated_calls(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    fake = FakeBackend()
    monkeypatch.setattr(llm, "get_backend", lambda model=None: fake)

    a = llm.chat("sys", "user-prompt-1")
    b = llm.chat("sys", "user-prompt-1")  # served from cache
    assert a == b == "ok:user-prompt-1"
    assert len(fake.calls) == 1  # second call never hit the backend

    # different params -> different cache key -> real call
    llm.chat("sys", "user-prompt-1", temperature=0.7)
    assert len(fake.calls) == 2


def test_empty_output_not_cached(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)

    class EmptyBackend(FakeBackend):
        def chat(self, *a, **k):
            super().chat(*a, **k)
            return ""

    fake = EmptyBackend()
    monkeypatch.setattr(llm, "get_backend", lambda model=None: fake)
    assert llm.chat("s", "u") == ""
    assert llm.chat("s", "u") == ""  # not cached; backend called again
    assert len(fake.calls) == 2


def test_strong_tier_routing(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_model_strong", "big-model")

    seen_models: list[str] = []

    def fake_backend(model=None):
        b = FakeBackend()
        b.model = model or settings.llm_model
        seen_models.append(b.model)
        return b

    monkeypatch.setattr(llm, "get_backend", fake_backend)
    llm.chat("s", "cheap-call")
    llm.chat("s", "strong-call", strong=True)
    assert seen_models == [settings.llm_model, "big-model"]
    # unset -> strong falls back to the base model
    monkeypatch.setattr(settings, "llm_model_strong", "")
    llm.chat("s", "strong-fallback", strong=True)
    assert seen_models[-1] == settings.llm_model


def test_run_concurrent_order_and_progress(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_concurrency", 4)

    def job(i, item):
        # deliberately out-of-order completion
        delay = (len(items) - i) * 0.01
        threading.Event().wait(delay)
        return item * 2

    items = list(range(10))
    done_order: list[int] = []
    out = llm.run_concurrent(job, items, on_done=lambda i, it, r: done_order.append(i))
    assert out == [x * 2 for x in items]  # results in input order
    assert sorted(done_order) == list(range(10))  # every job reported


def test_run_concurrent_sequential_when_disabled(tmp_path, monkeypatch):
    _fresh_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "llm_concurrency", 1)
    out = llm.run_concurrent(lambda i, x: x + 1, [1, 2, 3])
    assert out == [2, 3, 4]


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:

                class _MP:
                    def setattr(self, obj, attr, value):
                        setattr(obj, attr, value)

                fn(Path(td), _MP())
            print(f"PASS {name}")
    print("all llm-perf tests passed")
