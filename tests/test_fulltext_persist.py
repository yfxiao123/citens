"""Fetched-PDF persistence + Unpaywall strictness + local-file caching."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from citens.config import settings  # noqa: E402
from citens.grounding import fulltext  # noqa: E402
from citens.models import Paper  # noqa: E402
from tests.test_pdf_smoke import _make_pdf  # noqa: E402


def _paper(doi="10.1234/test.2026", title="A test paper about order flow imbalance"):
    return Paper(
        title=title, authors=["A B"], year=2026, abstract="abs",
        source="test", citation_count=0, url="", doi=doi,
    )


@pytest.fixture
def papers_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "papers_dir", str(tmp_path / "papers"))
    return tmp_path / "papers"


def test_fetched_pdf_is_persisted_and_locally_matched(papers_dir, monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    pdf = _make_pdf("Sample methods text. " * 40)
    text = fulltext._pdf_bytes_to_text(pdf, _paper())
    assert text and "Sample methods" in text
    kept = list(Path(papers_dir).glob("auto-*.pdf"))
    assert len(kept) == 1, "fetched PDF must survive conversion in papers/"
    # and the next run's local-first scan finds it by slug
    assert fulltext._local_pdf(_paper()) == kept[0]


def test_unconvertible_pdf_is_not_kept(papers_dir, monkeypatch):
    monkeypatch.setattr(settings, "cache_enabled", False)
    # valid PDF, but < 500 chars of text -> _convert_pdf_file returns None
    pdf = _make_pdf("Short body.")
    text = fulltext._pdf_bytes_to_text(pdf, _paper())
    assert text is None
    assert not list(Path(papers_dir).glob("auto-*.pdf")), (
        "an unconvertible PDF must not poison later runs' local scan"
    )


def test_no_paper_argument_still_uses_temp_file(papers_dir, monkeypatch):
    import glob
    import tempfile

    monkeypatch.setattr(settings, "cache_enabled", False)
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "*.pdf")))
    pdf = _make_pdf("Temporary conversion only. " * 30)
    text = fulltext._pdf_bytes_to_text(pdf)
    assert text
    assert not list(Path(papers_dir).glob("auto-*.pdf"))
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "*.pdf")))
    assert not (after - before), "temp PDFs must still be cleaned up"


def test_unpaywall_harvests_all_locations_pdf_urls_only(monkeypatch):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "best_oa_location": {
                    "url_for_pdf": "https://publisher.org/landing/page.pdf",
                    "url": "https://publisher.org/landing/page",
                },
                "oa_locations": [
                    {"url_for_pdf": "https://repo.org/copy.pdf", "url": "https://repo.org/"},
                    {"url_for_pdf": "https://repo.org/copy.pdf"},  # dup -> once
                    {"url_for_pdf": "", "url": "https://x.org/a"},  # empty -> dropped
                    {"url": "https://x.org/landing"},  # landing only -> dropped
                ],
            }

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    urls = fulltext._unpaywall_pdf_url("10.1/x")
    assert urls == ["https://repo.org/copy.pdf"], (
        "all oa_locations' url_for_pdf, deduped, no landing pages"
    )


def test_openalex_harvests_all_location_pdf_urls(monkeypatch):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"locations": [
                {"pdf_url": "https://ut-capitole.fr/1422/1/microstructure.pdf"},
                {"pdf_url": ""},
                {"pdf_url": "https://dl.acm.org/doi/pdf/10.1145/3604237.3626881"},
                {"landing_page_url": "https://x.org/"},
            ]}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    urls = fulltext._openalex_pdf_urls("10.1/x")
    assert urls == [
        "https://ut-capitole.fr/1422/1/microstructure.pdf",
        "https://dl.acm.org/doi/pdf/10.1145/3604237.3626881",
    ]


def test_openalex_no_doi_returns_empty():
    assert fulltext._openalex_pdf_urls("") == []


def test_local_conversion_cached_by_mtime(papers_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "papers_dir", str(papers_dir))
    monkeypatch.setattr(settings, "cache_enabled", True)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "cache_ttl_days", 0)

    pdf_path = Path(papers_dir) / "10-1234-test-2026.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(_make_pdf("Cached conversion text. " * 30))

    calls = {"n": 0}
    orig = fulltext._convert_pdf_file

    def counting(path):
        calls["n"] += 1
        return orig(path)

    monkeypatch.setattr(fulltext, "_convert_pdf_file", counting)
    p = _paper()
    assert fulltext.fetch_fulltext(p)  # first call converts
    assert calls["n"] == 1
    assert fulltext.fetch_fulltext(p)  # second call hits the mtime cache
    assert calls["n"] == 1
    # replacing the file invalidates the cache (mtime changes)
    pdf_path.write_bytes(_make_pdf("Replaced file content. " * 30))
    import os
    import time as _t

    os.utime(pdf_path, (_t.time() + 10, _t.time() + 10))
    assert fulltext.fetch_fulltext(p)
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_arxiv_source_timeout_returns_empty(monkeypatch):
    """A hanging arXiv backend must not stall the search stage: the source
    returns [] after its budget instead of crawling for minutes."""

    from citens.search.arxiv import ArxivSearcher

    def _hang(self, keywords, max_results):
        import time as _t
        _t.sleep(5)
        return ["never"]

    monkeypatch.setattr(ArxivSearcher, "SOURCE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(ArxivSearcher, "_search_sync", _hang)
    import time as _t2

    t0 = _t2.monotonic()
    out = await ArxivSearcher().search(["kw"], 10)
    dt = _t2.monotonic() - t0
    assert out == []
    assert dt < 2.0, f"timeout must bound the source (took {dt:.1f}s)"


def test_s2_doi_lookup_harvests_openaccess_pdf(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"openAccessPdf": {"url": "http://people.orie.cornell.edu/sfs33/x.pdf"}}

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return _Resp()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _Client())
    assert fulltext._s2_pdf_url("10.1/x") == [
        "http://people.orie.cornell.edu/sfs33/x.pdf"
    ]

    class _RespNone(_Resp):
        def json(self):
            return {"openAccessPdf": None}

    class _ClientN(_Client):
        def get(self, url, params=None):
            return _RespNone()

    monkeypatch.setattr(fulltext, "sync_client", lambda *a, **k: _ClientN())
    assert fulltext._s2_pdf_url("10.1/x") == []
    assert fulltext._s2_pdf_url("") == []
