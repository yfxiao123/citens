"""Harvest audit trail + the two 0/16-run fixes:

- arXiv papers whose only arXiv marker is the DOI (10.48550/arXiv.<id>)
  now hit the direct-pdf fast path instead of the throttled title lookup
  (S2's openAccessPdf is EMPTY for exactly these papers — measured on the
  2026-08-23 generative-recommendation run that landed 0/16 fulltext).
- the S2 leg sends the configured SEMANTIC_SCHOLAR_API_KEY.
- fetch_fulltext records a per-paper outcome ("dl.acm.org HTTP 403",
  "arxiv.org ok") that the transcript's ground lines surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens.config import settings  # noqa: E402
from citens.grounding import fulltext  # noqa: E402
from citens.models import Paper  # noqa: E402


def _paper(**kw) -> Paper:
    base = {"title": "Masked Diffusion for Generative Recommendation",
            "authors": ["A"], "year": 2025, "url": "", "doi": ""}
    base.update(kw)
    return Paper(**base)


# --- fix 1: arXiv DOI fast path ---------------------------------------------

def test_arxiv_doi_in_doi_field_is_direct_pdf():
    p = _paper(doi="10.48550/arXiv.2511.23021",
               url="https://doi.org/10.48550/arXiv.2511.23021")
    assert fulltext._arxiv_pdf_url(p) == "https://arxiv.org/pdf/2511.23021.pdf"


def test_arxiv_doi_only_in_url_field():
    p = _paper(url="https://doi.org/10.48550/arXiv.2409.16674")
    assert fulltext._arxiv_pdf_url(p) == "https://arxiv.org/pdf/2409.16674.pdf"


def test_arxiv_doi_old_style_id(monkeypatch):
    monkeypatch.setattr(fulltext, "_arxiv_title_lookup", lambda p: None)
    p = _paper(doi="10.48550/arXiv.cs/0301012")
    assert fulltext._arxiv_pdf_url(p) == "https://arxiv.org/pdf/cs/0301012.pdf"
    p2 = _paper(doi="10.48550/arXiv.math.GT/0309136")
    assert fulltext._arxiv_pdf_url(p2) == "https://arxiv.org/pdf/math.GT/0309136.pdf"


def test_plain_doi_still_falls_to_title_lookup(monkeypatch):
    monkeypatch.setattr(fulltext, "_arxiv_title_lookup", lambda p: "LOOKED_UP")
    p = _paper(doi="10.1145/3539618.3591663")
    assert fulltext._arxiv_pdf_url(p) == "LOOKED_UP"


# --- fix 2: S2 key -----------------------------------------------------------

def test_s2_sends_configured_key(monkeypatch):
    seen: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"openAccessPdf": {"url": "https://x.org/p.pdf"}}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp()

    def factory(*a, headers=None, **k):
        seen["headers"] = headers
        return _Client()

    monkeypatch.setattr(fulltext, "sync_client", factory)
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "s2k-test")
    assert fulltext._s2_pdf_url("10.1/x") == ["https://x.org/p.pdf"]
    assert seen["headers"].get("x-api-key") == "s2k-test"

    monkeypatch.setattr(settings, "semantic_scholar_api_key", "")
    fulltext._s2_pdf_url("10.1/x")
    assert "x-api-key" not in (seen["headers"] or {})


# --- fix 3: the audit trail ---------------------------------------------------

def test_fetch_report_records_failure_reasons(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(fulltext, "_local_pdf", lambda p: None)
    monkeypatch.setattr(fulltext, "_arxiv_pdf_url",
                        lambda p: "https://arxiv.org/pdf/2501.00001.pdf")
    monkeypatch.setattr(fulltext, "rewrite_url", lambda u: u)

    class _Resp:
        def __init__(self, code=200, payload=None, content=b""):
            self.status_code = code
            self.content = content
            self.headers = {}
            self._payload = payload

        def json(self):
            return self._payload or {}

    class _Client:  # dispatch by URL: arXiv throttled, S2 lists a paywalled ACM pdf
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            if "api.semanticscholar.org" in url:
                return _Resp(payload={"openAccessPdf":
                                      {"url": "https://dl.acm.org/doi/pdf/10.1/x"}})
            if "arxiv.org/pdf" in url:
                return _Resp(code=429)
            return _Resp(code=403)

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    p = _paper(doi="10.1/x", title="Audit Failure Paper")
    assert fulltext.fetch_fulltext(p) is None
    report = fulltext.fetch_report(p.id)
    assert "arxiv:arxiv.org HTTP 429" in report
    assert "s2:dl.acm.org HTTP 403" in report  # S2 listed it; ACM refused us


def test_fetch_report_records_success_leg(monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    monkeypatch.setattr(fulltext, "_local_pdf", lambda p: None)

    class _Resp:
        status_code = 200
        content = b"%PDF-" + b"x" * 3000
        headers = {"content-type": "application/pdf"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _Resp()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    monkeypatch.setattr(fulltext, "_pdf_bytes_to_text", lambda c, p=None: "body " * 500)
    p = _paper(url="https://arxiv.org/abs/2501.00001", title="Audit Success Paper")
    assert fulltext.fetch_fulltext(p)
    report = fulltext.fetch_report(p.id)
    assert report.startswith("arxiv") and "✓" in report


def test_download_and_convert_outcome_on_non_pdf(monkeypatch):
    class _Resp:
        status_code = 200
        content = b"<html>" + b"x" * 3000
        headers = {"content-type": "text/html"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            return _Resp()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    monkeypatch.setattr(fulltext, "rewrite_url", lambda u: u)
    text, outcome = fulltext._download_and_convert("https://pubs.org/xxx")
    assert text is None
    assert "非PDF" in outcome
