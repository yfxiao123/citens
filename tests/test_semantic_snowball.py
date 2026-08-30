"""The semantic (related_works) snowball direction + query-aware ranking."""

from __future__ import annotations

import httpx
import pytest
import respx

from citens.models import Paper
from citens.search.snowball import snowball


@pytest.fixture(autouse=True)
def _no_snowball_cache(monkeypatch):
    """The snowball disk cache would leak results ACROSS tests (same anchor
    DOI -> same cache key) — disable it for determinism."""
    monkeypatch.setattr("citens.search.snowball.cache.get", lambda *a, **k: None)
    monkeypatch.setattr("citens.search.snowball.cache.put", lambda *a, **k: None)


_RELATED_LOOKUP = "https://api.openalex.org/works/https://doi.org/10.1/anchor"


def _work(wid: str, title: str, cited: int, year: int = 2023) -> dict:
    return {
        "id": f"https://openalex.org/{wid}",
        "title": title,
        "authorships": [{"author": {"display_name": "A Author"}}],
        "publication_year": year,
        "cited_by_count": cited,
        "doi": None,
        "primary_location": {},
        "open_access": {},
    }


@pytest.mark.asyncio
@respx.mock
async def test_related_direction_fetches_semantic_neighbors():
    respx.get(_RELATED_LOOKUP).mock(
        return_value=httpx.Response(200, json={
            "related_works": [
                "https://openalex.org/W1",
                "https://openalex.org/W2",
            ]
        })
    )
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [
            _work("W1", "Generative Recommendation with Diffusion Models", 8),
            _work("W2", "Totally Unrelated Soil Chemistry Study", 500),
        ]})
    )
    anchor = Paper(title="Anchor", authors=["X"], year=2022, doi="10.1/anchor")
    out = await snowball(
        [anchor], set(), backward=False, forward=False, related=True,
        relevance_terms=["generative recommendation", "diffusion model"],
    )
    assert route.calls.last.request.url.params["filter"] == "openalex_id:W1|W2"
    titles = [p.title for p in out]
    assert "Generative Recommendation with Diffusion Models" in titles
    # the 500-cited off-topic paper is admitted (>=3 citations) but must NOT
    # outrank the topical one under query-aware ranking
    assert out[0].title.startswith("Generative Recommendation")
    assert out[0].source.startswith("snowball-related(")


@pytest.mark.asyncio
@respx.mock
async def test_citation_only_ranking_without_terms():
    """No relevance_terms -> legacy behavior: pure citation order."""
    respx.get(_RELATED_LOOKUP).mock(
        return_value=httpx.Response(200, json={
            "related_works": ["https://openalex.org/W1", "https://openalex.org/W2"]
        })
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [
            _work("W1", "Topical But Obscure", 4),
            _work("W2", "Famous Off Topic", 900),
        ]})
    )
    anchor = Paper(title="Anchor", authors=["X"], year=2022, doi="10.1/anchor")
    out = await snowball(
        [anchor], set(), backward=False, forward=False, related=True,
    )
    assert out[0].title == "Famous Off Topic"  # legacy popularity order


@pytest.mark.asyncio
@respx.mock
async def test_related_respects_existing_ids_and_gates():
    respx.get(_RELATED_LOOKUP).mock(
        return_value=httpx.Response(200, json={
            "related_works": ["https://openalex.org/W1", "https://openalex.org/W2"]
        })
    )
    respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(200, json={"results": [
            _work("W1", "Low Cited Paper", 1),   # below the >=3 gate
            _work("W2", "Good Paper", 25),
        ]})
    )
    anchor = Paper(title="Anchor", authors=["X"], year=2022, doi="10.1/anchor")
    existing = {Paper(title="Good Paper", authors=["Y"], year=2023).id}
    out = await snowball(
        [anchor], existing, backward=False, forward=False, related=True,
    )
    assert out == []  # one gated out, one already in the pool


_S2_ITEM = {
    "title": "Fallback Found Paper",
    "authors": [{"name": "A Author"}],
    "year": 2023,
    "abstract": "",
    "citationCount": 10,
    "externalIds": {"DOI": "10.9/x"},
    "venue": "",
    "url": "",
}


@pytest.mark.asyncio
@respx.mock
async def test_backward_falls_back_to_s2_when_openalex_dead():
    """OpenAlex's daily budget runs out (429 on every call) — the direction
    must fall back to S2 instead of silently returning zero (measured live
    failure: snowball returned n_cands=0 for a whole bench run)."""
    respx.get(_RELATED_LOOKUP).mock(return_value=httpx.Response(429, json={}))
    respx.get(
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1/anchor/references"
    ).mock(return_value=httpx.Response(
        200, json={"data": [{"citedPaper": _S2_ITEM}]}
    ))
    anchor = Paper(title="Anchor", authors=["X"], year=2022, doi="10.1/anchor")
    out = await snowball(
        [anchor], set(), backward=True, forward=False, related=False,
    )
    assert [p.title for p in out] == ["Fallback Found Paper"]
    assert out[0].source.startswith("snowball-backward·s2(")


@pytest.mark.asyncio
@respx.mock
async def test_related_falls_back_to_s2_recommendations():
    respx.get(_RELATED_LOOKUP).mock(return_value=httpx.Response(429, json={}))
    respx.get(
        "https://api.semanticscholar.org/recommendations/v1/papers"
        "/forpaper/DOI:10.1/anchor"
    ).mock(return_value=httpx.Response(
        200, json={"recommendedPapers": [_S2_ITEM]}
    ))
    anchor = Paper(title="Anchor", authors=["X"], year=2022, doi="10.1/anchor")
    out = await snowball(
        [anchor], set(), backward=False, forward=False, related=True,
    )
    assert out and out[0].title == "Fallback Found Paper"
    assert out[0].source.startswith("snowball-related·s2(")
