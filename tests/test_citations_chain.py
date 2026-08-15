"""Citation identity chain: review.md [n] markers -> claims -> table -> provenance.

The [n] in prose, the references list, the BibTeX and the verifier verdicts
all key off CitationTable; drift here silently corrupts the trust story.
"""

from citens.grounding.citations import (
    CitationTable,
    build_provenance,
    parse_claims_from_review,
)
from citens.models import Paper, Verdict, VerificationResult

REVIEW = """# Topic

## Introduction

First claim cites one paper [0]. Second claim cites two at once [2][1].

A claim in the next paragraph repeats a citation [0] and adds one [2].

## References

[0] Ignored Author (2001). Never parsed as a claim [3].
[1] Another One (2002). Also not a claim [9].
[2] Third Author (2003). Not a claim either [0].
"""


def _papers() -> list[Paper]:
    return [
        Paper(title="DeepLOB", authors=["Zihao Zhang", "Stefan Zohren"], year=2018, doi="10.1109/tsp.2019.2907260"),
        Paper(title="Optimal Execution", authors=["Aurélien Alfonsi"], year=2010, source="OpenAlex (SIAM)"),
        Paper(title="Queue Valuation", authors=["Ciamac Moallemi", "Kai Yuan"], year=2016, venue="SSRN Electronic Journal"),
    ]


class TestParseClaims:
    def test_extracts_cited_sentences(self):
        claims = parse_claims_from_review(REVIEW)
        assert len(claims) == 3

    def test_multi_marker_claim_collects_all_indices(self):
        claims = parse_claims_from_review(REVIEW)
        assert claims[1].citation_indices == [1, 2]  # deduped + sorted

    def test_section_attribution(self):
        claims = parse_claims_from_review(REVIEW)
        assert all(c.section == "Introduction" for c in claims)

    def test_references_section_is_skipped(self):
        claims = parse_claims_from_review(REVIEW)
        assert not any("Never parsed" in c.text for c in claims)

    def test_marker_survives_in_claim_text(self):
        claims = parse_claims_from_review(REVIEW)
        assert "[0]" in claims[0].text

    def test_empty_and_markerless_input(self):
        assert parse_claims_from_review("") == []
        assert parse_claims_from_review("## A\n\nNo markers here.") == []


class TestCitationTable:
    def test_index_paper_id_roundtrip(self):
        t = CitationTable(_papers())
        for i, p in enumerate(_papers()):
            assert t.paper_id(i) == p.id

    def test_unknown_index_returns_empty(self):
        t = CitationTable(_papers())
        assert t.paper_id(99) == ""
        assert t.label(99) == ""

    def test_len(self):
        assert len(CitationTable(_papers())) == 3

    def test_references_md_lists_all_indices(self):
        md = CitationTable(_papers()).references_md()
        assert md.count("\n") == 2
        assert md.startswith("[0] ")
        assert "[2] " in md

    def test_bibtex_key_is_ascii(self):
        bib = CitationTable(_papers()).to_bibtex()
        assert "@article{alfonsi2010" in bib  # accented author -> ascii key
        assert "doi = {10.1109/tsp.2019.2907260}" in bib

    def test_bibtex_prefers_venue_over_source_parens(self):
        bib = CitationTable(_papers()).to_bibtex()
        assert "journal = {SSRN Electronic Journal}" in bib
        assert "journal = {SIAM}" in bib  # extracted from "OpenAlex (SIAM)"


class TestProvenance:
    def test_claims_get_verdicts_merged(self):
        papers = _papers()
        table = CitationTable(papers)
        claims = parse_claims_from_review(REVIEW)
        ver = [
            VerificationResult(
                claim_text=claims[0].text,
                verdict=Verdict.SUPPORTED,
                note="abstract states it",
            ),
            VerificationResult(
                claim_text=claims[1].text,
                verdict=Verdict.UNSUPPORTED,
                note="not in abstract",
            ),
        ]
        prov = build_provenance(claims, table, ver)
        assert prov[0]["verdict"] == "supported"
        assert prov[1]["verdict"] == "unsupported"
        assert prov[0]["citations"][0]["paper_id"] == papers[0].id

    def test_no_verdicts_leaves_entries_clean(self):
        table = CitationTable(_papers())
        claims = parse_claims_from_review(REVIEW)
        prov = build_provenance(claims, table)
        assert all("verdict" not in e for e in prov)
