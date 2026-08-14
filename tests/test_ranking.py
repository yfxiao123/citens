"""Minimal tests for venue-aware ranking (Phase B). No network, no LLM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from litreview.config import settings  # noqa: E402
from litreview.models import ScoredPaper  # noqa: E402
from litreview.ranking import (  # noqa: E402
    SJRIndex,
    citation_factor,
    quartile_histogram,
    rank_papers,
    venue_score,
)

SAMPLE_CSV = """Rank;Title;Type;Issn;SJR;SJR Best Quartile;H index
1;Nature;journal;0028-0836;14.928;Q1;1011
2;Journal of Finance;journal;0022-1082;8.123;Q1;233
3;Physics Reports;journal;0370-1573;3.412;Q1;290
4;Quantitative Finance;journal;1469-7688;1.211;Q2;62
5;Journal of Applied Econometrics;journal;0883-7252;1.002;Q2;77
6;Some Regional Studies Journal;journal;0000-0000;0.402;Q3;30
"""

# Michael-E-Rose mirror format (comma-delimited, no quartile column)
SAMPLE_MIRROR = """Title,field,year,SJR,h-index,avg_citations,Issn,Sourceid
Nature,1000,2022,14.9,1011,42.0,0028-0836,24327
Nature,1000,2023,15.1,1011,44.0,0028-0836,24327
Journal of Finance,2000,2023,8.1,233,7.0,0022-1082,18448
Tiny Journal,1000,2023,0.01,5,0.1,0000-0001,99999
"""


def make_p(**kw) -> ScoredPaper:
    base = dict(
        title="A paper",
        authors=["A"],
        year=2020,
        abstract="abs",
        source="OpenAlex (X)",
        url="",
        citation_count=0,
        doi="",
        relevance_score=3,
    )
    base.update(kw)
    return ScoredPaper(**base)


def _load_index(tmp_path: Path) -> SJRIndex:
    f = tmp_path / "sjr.csv"
    f.write_text(SAMPLE_CSV, encoding="utf-8")
    return SJRIndex.load(f)


def test_sjr_load_and_lookup(tmp_path):
    idx = _load_index(tmp_path)
    assert len(idx) == 6
    hit = idx.lookup("Journal of Finance")
    assert hit is not None and hit.quartile == "Q1"
    # normalization: case/punctuation insensitive
    assert idx.lookup("journal-of finance  ").quartile == "Q1"
    # containment fallback
    hit2 = idx.lookup("Some Regional Studies Journal — Special Issue")
    assert hit2 is not None and hit2.quartile == "Q3"
    assert idx.lookup("") is None
    assert idx.lookup("No Such Journal Exists At All") is None


def test_sjr_mirror_format(tmp_path):
    f = tmp_path / "mirror.csv"
    f.write_text(SAMPLE_MIRROR, encoding="utf-8")
    idx = SJRIndex.load(f)  # auto-detects the comma/mirror format
    assert len(idx) == 3
    nature = idx.lookup("Nature")
    assert nature is not None and nature.quartile == "Q1"  # top of distribution
    assert nature.sjr == 15.1  # latest-year row wins over 2022
    jf = idx.lookup("Journal of Finance")
    assert jf is not None and jf.quartile in {"Q2", "Q3"}  # mid-distribution
    tiny = idx.lookup("Tiny Journal")
    assert tiny is not None and tiny.quartile == "Q4"


def test_scores():
    assert venue_score("Q1") == 1.0
    assert venue_score("q2") == 0.75
    assert venue_score("") == 0.5  # neutral when unknown
    assert citation_factor(0) == 0.0
    assert citation_factor(999) > 0.99
    assert citation_factor(10_000) == 1.0  # capped


def test_rank_papers_without_sjr_data(tmp_path, monkeypatch):
    # no SJR file -> venue factor neutral, ranking still works
    monkeypatch.setattr(settings, "sjr_csv_path", str(tmp_path / "missing.csv"))
    import litreview.ranking as rk

    rk._index, rk._index_loaded = None, False  # reset singleton

    hi_cited_low_rel = make_p(relevance_score=3, citation_count=2000)
    hi_rel_low_cited = make_p(relevance_score=5, citation_count=1, title="B paper", doi="10.1/b")
    out = rank_papers([hi_cited_low_rel, hi_rel_low_cited])
    assert out[0].title == "B paper"  # relevance dominates by default weights
    assert out[0].rank_score > out[1].rank_score
    assert out[0].venue_quartile == ""

    rk._index_loaded = False  # let later tests reload


def test_rank_papers_venue_boost(tmp_path, monkeypatch):
    idx = _load_index(tmp_path)
    import litreview.ranking as rk

    monkeypatch.setattr(rk, "get_sjr_index", lambda: idx)
    preprint = make_p(relevance_score=5, citation_count=500, title="arXiv preprint", doi="10.1/p")
    journal = make_p(
        relevance_score=4,
        citation_count=50,
        venue="Journal of Finance",
        title="JF paper",
        doi="10.1/j",
    )
    out = rk.rank_papers([preprint, journal])
    scores = {p.title: p.rank_score for p in out}
    # preprint: .6*1 + .2*~0.9 + .2*0.5 = 0.88 ; JF: .6*.8 + .2*~0.48 + .2*1.0 = 0.78
    assert scores["arXiv preprint"] > scores["JF paper"]
    assert next(p for p in out if p.title == "JF paper").venue_quartile == "Q1"
    # but a high-relevance Q1 journal beats the preprint
    strong = make_p(
        relevance_score=5,
        citation_count=500,
        venue="Nature",
        title="Nature paper",
        doi="10.1/n",
    )
    out2 = rk.rank_papers([preprint, strong])
    assert out2[0].title == "Nature paper"


def test_histogram():
    ps = [
        make_p(venue_quartile="Q1"),
        make_p(venue_quartile="Q1", title="b", doi="10.1/b"),
        make_p(),
    ]
    hist = quartile_histogram(ps)
    assert hist == {"Q1": 2, "unranked": 1}


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)

                class _MP:
                    def setattr(self, obj, attr, value):
                        setattr(obj, attr, value)

                fn(td_path, _MP())
            print(f"PASS {name}")
    print("all ranking tests passed")
