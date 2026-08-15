"""Minimal tests for the access layer (Phase A): EZproxy rewrite, local PDF
pickup, fetch list. Pure-logic tests — no network, no LLM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens.config import settings  # noqa: E402
from citens.grounding.fetchlist import suggested_filename, write_fetch_list  # noqa: E402
from citens.grounding.fulltext import _local_pdf, pdf_slugs, slugify  # noqa: E402
from citens.models import Paper  # noqa: E402
from citens.net import rewrite_url  # noqa: E402


def make_paper(**kw) -> Paper:
    base = dict(
        title="Agent-based models of financial markets",
        authors=["E. Samanidou"],
        year=2007,
        abstract="A survey of agent-based models.",
        source="openalex",
        url="",
        pdf_url="",
        citation_count=0,
        doi="",
    )
    base.update(kw)
    return Paper(**base)


def test_slugify():
    assert slugify("Limit Order Book: A Survey!") == "limit-order-book-a-survey"
    assert slugify("10.1016/j.physa.2015.03.043") == "10-1016-j-physa-2015-03-043"


def test_rewrite_url_off_by_default():
    settings.ezproxy_prefix = ""
    url = "https://www.sciencedirect.com/science/article/pii/X"
    assert rewrite_url(url) == url


def test_rewrite_url_free_hosts_untouched():
    settings.ezproxy_prefix = "https://lib.univ.edu.cn/login?url="
    assert (
        rewrite_url("https://arxiv.org/pdf/1407.5684.pdf")
        == "https://arxiv.org/pdf/1407.5684.pdf"
    )
    assert (
        rewrite_url("https://api.openalex.org/works?foo=1")
        == "https://api.openalex.org/works?foo=1"
    )


def test_rewrite_url_publisher():
    settings.ezproxy_prefix = "https://lib.univ.edu.cn/login?url="
    url = "https://www.sciencedirect.com/science/article/pii/X?a=1"
    out = rewrite_url(url)
    assert out.startswith("https://lib.univ.edu.cn/login?url=https%3A%2F%2F")
    assert out.endswith(quote_all(url))


def quote_all(u: str) -> str:
    from urllib.parse import quote

    return quote(u, safe="")


def test_rewrite_url_respects_accessible_domains():
    settings.ezproxy_prefix = "https://lib.univ.edu.cn/login?url="
    settings.accessible_domains = "springer.com"
    url = "https://link.springer.com/article/10.1007/x"
    assert rewrite_url(url).startswith("https://lib.univ.edu.cn/login?url=")
    other = "https://www.tandfonline.com/doi/x"
    assert rewrite_url(other) == other  # not declared -> untouched
    settings.accessible_domains = ""


def test_rewrite_url_prefix_without_param():
    settings.ezproxy_prefix = "https://lib.univ.edu.cn/login"
    out = rewrite_url("https://www.sciencedirect.com/x")
    assert "?url=" in out
    settings.ezproxy_prefix = ""


def test_pdf_slugs():
    p = make_paper(
        doi="10.1016/j.physa.2015.03.043", url="https://arxiv.org/abs/1407.5684"
    )
    slugs = pdf_slugs(p)
    assert "10-1016-j-physa-2015-03-043" in slugs
    assert "1407-5684" in slugs
    short = make_paper(title="LOB")  # too short to title-match
    assert pdf_slugs(short) == []


def test_local_pdf_pickup(tmp_path, monkeypatch):
    d = tmp_path / "papers"
    d.mkdir()
    monkeypatch.setattr(settings, "papers_dir", str(d))

    p = make_paper(doi="10.1016/j.physa.2015.03.043")
    assert _local_pdf(p) is None
    f = d / "10-1016-j-physa-2015-03-043.pdf"
    f.write_bytes(b"%PDF-1.4 dummy")
    assert _local_pdf(p) == f

    # title-based name matches too
    p2 = make_paper()
    f2 = d / "agent-based-models-of-financial-markets-scan.pdf"
    f2.write_bytes(b"%PDF-1.4")
    assert _local_pdf(p2) == f2


def test_suggested_filename():
    assert (
        suggested_filename(make_paper(doi="10.1016/j.physa.2015.03.043"))
        == "10-1016-j-physa-2015-03-043.pdf"
    )
    name = suggested_filename(make_paper(url="https://arxiv.org/abs/2501.12345"))
    assert name == "arxiv-2501-12345.pdf"
    assert suggested_filename(make_paper()).endswith(".pdf")


def test_fetch_list(tmp_path, monkeypatch):
    pd = tmp_path / "papers"
    monkeypatch.setattr(settings, "papers_dir", str(pd))
    papers = [
        make_paper(doi="10.1016/x", url="https://www.sciencedirect.com/a"),
        make_paper(title="Another Paper", url="https://arxiv.org/abs/2501.1"),
    ]
    out = write_fetch_list(str(tmp_path), papers)
    text = Path(out).read_text(encoding="utf-8")
    assert "fetch_list" in text
    assert "https://doi.org/10.1016/x" in text
    assert (pd / "README.md").exists()
    assert write_fetch_list(str(tmp_path), []) is None


if __name__ == "__main__":
    import tempfile

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            import inspect

            if "tmp_path" in inspect.signature(fn).parameters or "monkeypatch" in (
                inspect.signature(fn).parameters
            ):
                with tempfile.TemporaryDirectory() as td:

                    if "monkeypatch" in inspect.signature(fn).parameters:
                        class _MP:  # minimal shim
                            def setattr(self, obj, attr, value):
                                setattr(obj, attr, value)

                        fn(Path(td), _MP())
                    else:
                        fn(Path(td))
            else:
                fn()
            print(f"PASS {name}")
    print("all access-layer tests passed")
