"""Model-layer regression tests (dedup, normalization)."""

from citens.models import Paper


def test_authors_dedup_preserves_order():
    # OpenAlex emits one authorships entry per (author, affiliation)
    p = Paper(
        title="Studies of the limit order book around large price changes",
        authors=["Bence Toth", "Bence Toth ", "bence toth", "Jean-Philippe Bouchaud"],
    )
    assert p.authors == ["Bence Toth", "Jean-Philippe Bouchaud"]


def test_authors_dedup_case_and_whitespace_insensitive():
    p = Paper(title="t", authors=["  Alice   Smith ", "alice smith", "Bob Jones"])
    assert p.authors == ["Alice Smith", "Bob Jones"]


def test_authors_empty_and_none_entries():
    assert Paper(title="t", authors=[]).authors == []
    assert Paper(title="t", authors=["", "  ", None]).authors == []  # type: ignore[list-item]


def test_doi_prefixes_stripped():
    for raw, want in [
        ("https://doi.org/10.1109/TSP.2019.2907260", "10.1109/TSP.2019.2907260"),
        ("http://dx.doi.org/10.1/x", "10.1/x"),
        ("doi:10.1/y", "10.1/y"),
        ("10.1/z", "10.1/z"),
        ("", ""),
        (None, None),
    ]:
        assert Paper(title="t", doi=raw).doi == want


def test_brief_truncates_author_list():
    p = Paper(title="t", authors=[f"Author {i}" for i in range(8)])
    assert "et al." in p.brief()
    assert "Author 0" in p.brief()
