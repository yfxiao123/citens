"""TTL + sweep behavior for the disk cache (offline)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens import cache  # noqa: E402
from citens.config import settings  # noqa: E402


def _fresh(tmp_path, monkeypatch, ttl=30, sweep=1):
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cache_ttl_days", ttl)
    monkeypatch.setattr(settings, "cache_sweep_interval_days", sweep)


def test_fresh_entry_hits(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    cache.put("search", {"q": "x"}, [1])
    assert cache.get("search", {"q": "x"}) == [1]


def test_expired_entry_is_a_miss(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch, sweep=0)
    cache.put("search", {"q": "x"}, [1])
    p = cache._path(cache._key("search", {"q": "x"}))
    old = time.time() - 40 * 86400
    os.utime(p, (old, old))
    assert cache.get("search", {"q": "x"}) is None


def test_llm_namespace_never_expires(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch, ttl=30, sweep=0)
    cache.put("llm", {"q": "x"}, "answer")
    p = cache._path(cache._key("llm", {"q": "x"}))
    old = time.time() - 400 * 86400
    os.utime(p, (old, old))
    assert cache.get("llm", {"q": "x"}) == "answer"


def test_sweep_removes_expired_files(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch, ttl=30, sweep=1)
    cache.put("search", {"q": "stale"}, [1])
    cache.put("search", {"q": "fresh"}, [2])
    stale = cache._path(cache._key("search", {"q": "stale"}))
    old = time.time() - 40 * 86400
    os.utime(stale, (old, old))
    # force the sweep marker to be old so put()'s throttled sweep fires
    marker = cache._path(cache._SWEEP_MARKER)
    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"ts": time.time() - 2 * 86400}, f)
    os.utime(marker, (old, old))
    cache.put("search", {"q": "trigger"}, [3])
    assert not os.path.exists(stale)
    assert cache.get("search", {"q": "fresh"}) == [2]
