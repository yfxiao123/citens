"""Windowed (no-console) startup: the null stream must satisfy every library
that inspects stdout — uvicorn's logging formatter calls isatty() at boot and
crashed the v1.3.0 exe on every double-click launch (console-inherited test
launches never hit the None-stream path, which is why CI missed it)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens import desktop  # noqa: E402


def test_nullo_implements_stream_protocol():
    n = desktop._NullIO()
    assert n.isatty() is False            # uvicorn ColourizedFormatter
    assert n.writable() is True
    assert n.write("hello") == 5
    assert n.encoding == "utf-8"
    assert n.fileno is not None           # IOBase raises OSError on call, ok
    n.flush()                             # inherited no-op


def test_console_safe_installs_null_and_survives_uvicorn_logging(monkeypatch):
    # simulate the double-clicked windowed exe: no console, streams are None
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    desktop._console_safe()
    assert isinstance(sys.stdout, desktop._NullIO)
    assert isinstance(sys.stderr, desktop._NullIO)

    # the exact v1.3.0 crash: uvicorn boots its logging config on startup and
    # the colorized formatter probes the stream — this must configure cleanly
    from logging.config import dictConfig

    from uvicorn.config import LOGGING_CONFIG

    dictConfig(LOGGING_CONFIG)


def test_probe_existing_returns_none_when_no_server(monkeypatch):
    # deterministic: nothing answers any port (a real console may legitimately
    # be running on the dev machine — the probe finding it is correct there)
    import urllib.request

    def _fail(url, timeout=None):
        raise OSError("no server")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    assert desktop._probe_existing() is None
