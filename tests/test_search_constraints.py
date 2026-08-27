"""Clarification constraints compile into each source's NATIVE filter syntax
(no more "2023..2026" text hacks inside query strings), and per-query hit
stats flow back for the zero-hit calibration wave."""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import httpx
import pytest
import respx

from citens.models import Paper
from citens.search.base import search_round
from citens.search.crossref import CrossrefSearcher
from citens.search.filters import RetrievalConstraints
from citens.search.openalex import OpenAlexSearcher
from citens.search.semantic_scholar import SemanticScholarSearcher

OPENALEX_WORK = {
    "id": "https://openalex.org/W1",
    "title": "A Paper",
    "authorships": [{"author": {"display_name": "A Author"}}],
    "publication_year": 2021,
    "cited_by_count": 5,
    "doi": None,
    "primary_location": {},
    "open_access": {},
    "relevance_score": 1.0,
}

S2_ENVELOPE = {"data": [{"title": "S2 Paper", "year": 2020, "citationCount": 3,
                         "externalIds": {}, "url": "u", "authors": [{"name": "B"}]}]}


@pytest.mark.asyncio
@respx.mock
async def test_openalex_native_year_filters_and_stats():
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [OPENALEX_WORK]})
    )
    src = OpenAlexSearcher()
    src.set_constraints(RetrievalConstraints(year_from=2020, year_to=2024))
    papers = await src.search(["order book"], max_results=10)
    flt = route.calls.last.request.url.params["filter"]
    assert flt.startswith("default.search:order book")
    assert "from_publication_date:2020-01-01" in flt
    assert "to_publication_date:2024-12-31" in flt
    assert len(papers) == 1
    assert src.query_stats == {"order book": 1}
    assert papers[0].matched_queries == ["order book"]  # retrieval provenance


@pytest.mark.asyncio
@respx.mock
async def test_openalex_no_constraints_keeps_plain_filter():
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": []})
    )
    src = OpenAlexSearcher()
    await src.search(["order book"], max_results=10)
    flt = route.calls.last.request.url.params["filter"]
    assert flt == "default.search:order book"


@pytest.mark.asyncio
@respx.mock
async def test_s2_year_param_and_stats():
    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(200, json=S2_ENVELOPE)
    )
    src = SemanticScholarSearcher()
    src.set_constraints(RetrievalConstraints(year_from=2019))
    papers = await src.search(["order book"], max_results=10)
    assert route.calls.last.request.url.params["year"].startswith("2019-")
    assert len(papers) == 1
    assert src.query_stats == {"order book": 1}


@pytest.mark.asyncio
@respx.mock
async def test_crossref_pub_date_filters_and_stats():
    route = respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(200, json={"message": {"items": []}})
    )
    src = CrossrefSearcher()
    src.set_constraints(RetrievalConstraints(year_from=2018, year_to=2022))
    await src.search(["market making"], max_results=10)
    flt = route.calls.last.request.url.params["filter"]
    assert "from-pub-date:2018-01-01" in flt
    assert "until-pub-date:2022-12-31" in flt
    assert src.query_stats == {"market making": 0}


def _fake_arxiv_results():
    def result(title, year):
        return SimpleNamespace(
            title=title,
            authors=[SimpleNamespace(name="C Author")],
            published=datetime.datetime(year, 6, 1),
            summary="abstract text",
            entry_id="http://arxiv.org/abs/2401.00001",
            doi=None,
            journal_ref="",
        )

    return [result("old paper", 2015), result("new paper", 2023)]


@pytest.mark.asyncio
async def test_arxiv_post_filters_by_year_window(monkeypatch):
    import citens.search.arxiv as arxiv_mod

    fake_results = _fake_arxiv_results()

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_mod.arxiv, "Client", _FakeClient)
    src = arxiv_mod.ArxivSearcher()
    src.set_constraints(RetrievalConstraints(year_from=2020))
    papers = await src.search(["order book"], max_results=10)
    assert [p.title for p in papers] == ["new paper"]
    assert src.query_stats == {"order book": 1}


def test_search_round_aggregates_stats_and_survives_failures(monkeypatch):
    import asyncio

    from citens.search import base as search_base

    class _Counting:
        name = "counting"

        def __init__(self):
            self.constraints = None
            self.query_stats = {}

        def set_constraints(self, c):
            self.constraints = c

        async def search(self, keywords, max_results):
            for q in keywords:
                self.query_stats[q] = 0 if q == "miss" else 7
            return [Paper(title="A", authors=["X"], year=2020)]

    class _Broken:
        name = "broken"

        def __init__(self):
            self.query_stats = {"miss": 99}  # must NOT count: source fails

        async def search(self, keywords, max_results):
            raise RuntimeError("429")

    monkeypatch.setattr(search_base, "REGISTRY", {"counting": _Counting, "broken": _Broken})

    async def _nosleep(_):
        return None

    monkeypatch.setattr("asyncio.sleep", _nosleep)
    papers, health, stats = asyncio.run(
        search_round(["hit", "miss"], 10, sources=["counting", "broken"],
                     constraints=RetrievalConstraints(year_from=2020))
    )
    assert health["broken"].startswith("failed:")
    assert stats == {"hit": 7, "miss": 0}  # broken source's 99 discarded
    assert len(papers) == 1
