"""Search-adapter contract tests against mocked HTTP (respx).

If a source changes its payload shape (OpenAlex renaming a field, S2 altering
the bulk envelope), these fail loudly instead of silently returning zero
papers mid-run.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from citens.models import Paper
from citens.search.crossref import CrossrefSearcher
from citens.search.openalex import OpenAlexSearcher
from citens.search.semantic_scholar import SemanticScholarSearcher

OPENALEX_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Statistical properties of stock order books",
    "authorships": [
        # same author, three affiliations -> must collapse to one entry
        {"author": {"display_name": "Bence Toth"}},
        {"author": {"display_name": "bence toth"}},
        {"author": {"display_name": "Jean-Philippe Bouchaud"}},
    ],
    "publication_year": 2002,
    "abstract_inverted_index": {
        "we": [0], "study": [1], "order": [2], "books": [3],
    },
    "cited_by_count": 512,
    "doi": "https://doi.org/10.1140/epjb/e2002-00035-9",
    "primary_location": {
        "pdf_url": "https://arxiv.org/pdf/cond-mat/0012345",
        "source": {"display_name": "The European Physical Journal B"},
    },
    "open_access": {"oa_url": ""},
    "relevance_score": 42.0,
}

S2_ITEM = {
    "title": "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books",
    "authors": [{"name": "Zihao Zhang"}, {"name": "Stefan Zohren"}, {"name": "Stephen J. Roberts"}],
    "year": 2018,
    "abstract": "We propose a CNN for limit order book data.",
    "citationCount": 600,
    "externalIds": {"DOI": "10.1109/TSP.2019.2907260", "ArXiv": "1808.03668"},
    "url": "https://www.semanticscholar.org/paper/x",
    "venue": "IEEE Transactions on Signal Processing",
    "openAccessPdf": {"url": "https://arxiv.org/pdf/1808.03668"},
}

CROSSREF_ITEM = {
    "DOI": "10.1137/090762786",
    "title": ["Optimal Trade Execution and Absence of Price Manipulations in Limit Order Book Models"],
    "author": [{"given": "Aurélien", "family": "Alfonsi"}, {"given": "Alexander", "family": "Schied"}],
    "published-print": {"date-parts": [[2010, 1]]},
    "abstract": "<jats:p>We use limit order book shape to derive optimal execution.</jats:p>",
    "is-referenced-by-count": 311,
    "URL": "https://doi.org/10.1137/090762786",
    "container-title": ["SIAM Journal on Financial Mathematics"],
    "link": [{"URL": "https://pubs.siam.org/pdf/x", "content-type": "application/pdf"}],
}


@pytest.mark.asyncio
@respx.mock
async def test_openalex_end_to_end_contract():
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [OPENALEX_WORK]})
    )
    papers = await OpenAlexSearcher().search(["order book stylized facts"], max_results=10)
    # title+abstract search (default.search), not title-only — title-only
    # starved recall on the largest metadata source
    assert route.calls.last.request.url.params["filter"].startswith("default.search:")
    assert len(papers) == 1
    p = papers[0]
    assert isinstance(p, Paper)
    # the bug that motivated the authors dedup validator
    assert p.authors == ["Bence Toth", "Jean-Philippe Bouchaud"]
    # inverted-index abstract decode restores sentence order
    assert p.abstract == "we study order books"
    # DOI prefix stripped on the adapter path too
    assert p.doi == "10.1140/epjb/e2002-00035-9"
    assert p.venue == "The European Physical Journal B"
    assert p.pdf_url == "https://arxiv.org/pdf/cond-mat/0012345"
    assert p.citation_count == 512


@pytest.mark.asyncio
@respx.mock
async def test_openalex_error_is_gracefully_skipped():
    """A failing source must not kill the run: search() swallows per-source
    errors and returns whatever it has (here: nothing)."""
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    papers = await OpenAlexSearcher().search(["anything"], max_results=10)
    assert papers == []


@pytest.mark.asyncio
@respx.mock
async def test_semantic_scholar_bulk_contract():
    route = respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(200, json={"data": [S2_ITEM], "token": None})
    )
    papers = await SemanticScholarSearcher().search(["deep learning limit order book"], max_results=10)
    assert len(papers) == 1
    p = papers[0]
    assert p.authors == ["Zihao Zhang", "Stefan Zohren", "Stephen J. Roberts"]
    assert p.doi == "10.1109/TSP.2019.2907260"
    assert p.venue == "IEEE Transactions on Signal Processing"
    # openAccessPdf is requested and harvested — S2's free OA link feeds
    # fulltext grounding directly
    assert p.pdf_url == "https://arxiv.org/pdf/1808.03668"
    assert "openAccessPdf" in route.calls.last.request.url.params["fields"]
    # bulk endpoint gets sliced client-side to the per-keyword limit
    assert route.calls.last.request.url.params["query"] == "deep learning limit order book"


@pytest.mark.asyncio
@respx.mock
async def test_crossref_contract():
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(200, json={"message": {"items": [CROSSREF_ITEM]}})
    )
    papers = await CrossrefSearcher().search(["optimal execution limit order book"], max_results=10)
    assert len(papers) == 1
    p = papers[0]
    assert p.year == 2010
    # JATS tags cleaned out of the abstract
    assert p.abstract == "We use limit order book shape to derive optimal execution."
    assert "<jats" not in p.abstract
    assert p.pdf_url == "https://pubs.siam.org/pdf/x"
    assert p.venue == "SIAM Journal on Financial Mathematics"
    assert p.citation_count == 311


@pytest.mark.asyncio
@respx.mock
async def test_multi_source_search_gathers_and_dedups():
    """search_papers() merges sources and dedups by DOI, keeping the
    highest-cited variant of each."""
    from citens.search.base import search_papers

    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [OPENALEX_WORK]})
    )
    same_paper_crossref_shape = {
        "DOI": "10.1140/epjb/e2002-00035-9",  # same work, lower citation count
        "title": ["Statistical properties of stock order books"],
        "author": [{"given": "Bence", "family": "Toth"}],
        "issued": {"date-parts": [[2002]]},
        "is-referenced-by-count": 1,
        "URL": "https://doi.org/10.1140/epjb/e2002-00035-9",
        "container-title": ["The European Physical Journal B"],
    }
    respx.get("https://api.crossref.org/works").mock(
        return_value=httpx.Response(
            200, json={"message": {"items": [same_paper_crossref_shape, CROSSREF_ITEM]}}
        )
    )
    # only two sources enabled -> S2 endpoint not mocked and not called
    papers = await search_papers(["q"], max_results=10, sources=["openalex", "crossref"])
    by_doi = {p.doi: p for p in papers}
    assert len(by_doi) == 2  # deduped across sources
    assert by_doi["10.1140/epjb/e2002-00035-9"].citation_count == 512
