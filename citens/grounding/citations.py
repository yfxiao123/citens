"""Citation table, BibTeX export, claim parsing, and provenance.

The :class:`CitationTable` is the single source of truth for citation identity:
the same index that appears as ``[n]`` in the prose is what the references list,
the BibTeX export, the claim-citation links, and (later) the Verifier all use.
Keeping everything derived from one table prevents the classic drift between
inline markers and the bibliography.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from citens.models import Citation, Claim, Paper

_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[。.!?！？])\s+")
_REF_HEADER_RE = re.compile(r"参考文献|references", re.IGNORECASE)


_APA_MAX_AUTHORS = 10  # APA-7 lists up to 20; 10 keeps entries readable


def _apa_authors(authors: Sequence[str]) -> str:
    """OpenAlex display names ("Zheng Zhao") -> APA "Zhao, Z., Fan, W."

    Heuristic: last whitespace token is the surname, the rest becomes
    initials. Wrong for some name orders, but consistent and clearly better
    than truncating at three names with "et al." in the reference list.
    """
    parts: list[str] = []
    for name in authors[:_APA_MAX_AUTHORS]:
        tokens = [t for t in str(name).replace(",", " ").split() if t]
        if not tokens:
            continue
        surname = tokens[-1]
        initials = ".".join(w[0].upper() for w in tokens[:-1] if w[0].isalpha())
        parts.append(f"{surname}, {initials}." if initials else surname)
    if not parts:
        return ""
    out = ", ".join(parts)
    if len(authors) > _APA_MAX_AUTHORS:
        out += ", et al."
    return out


def _format_label(p: Paper) -> str:
    """Complete APA-7-style reference entry: authors (year). title. *venue*,
    volume(issue), pages. DOI — journal/volume/issue/pages included, per the
    technical report's house style, instead of the old title-only sketch."""
    authors = _apa_authors(p.authors) or "Anonymous"
    year = p.year or "n.d."
    venue = _bib_venue(p.source, getattr(p, "venue", ""))
    entry = f"{authors} ({year}). {p.title}. *{venue}*"
    vol, iss, pages = p.volume, p.issue, getattr(p, "pages", "")
    if vol:
        entry += f", {vol}"
        if iss:
            entry += f"({iss})"
    if pages:
        entry += f", {pages}"
    if p.doi:
        entry += f". https://doi.org/{p.doi}"
    elif p.url:
        entry += f". {p.url}"
    return entry


def _bibtex_key(p: Paper, index: int) -> str:
    """authorlastname + year + index, ASCII-safe and unique."""
    last = re.split(r"[\s,]+", p.authors[0])[-1] if p.authors else "anon"
    last = unicodedata.normalize("NFKD", last).encode("ascii", "ignore").decode("ascii")
    last = re.sub(r"[^A-Za-z]", "", last).lower() or "anon"
    return f"{last}{p.year or 'nd'}{index}"


def _bib_escape(value: object) -> str:
    return str(value).replace("{", "").replace("}", "").strip()


def _bib_venue(source: str, venue: str = "") -> str:
    """Prefer the explicit venue field, then the venue inside parens (e.g.
    'OpenAlex (Journal of Finance)'), otherwise the whole source string."""
    if venue.strip():
        return _bib_escape(venue)
    s = (source or "").strip()
    if "(" in s and s.endswith(")"):
        inner = s[s.find("(") + 1 : -1].strip()
        if inner:
            return _bib_escape(inner)
    return _bib_escape(s) or "Unknown"


class CitationTable:
    """Index <-> paper_id <-> label, for a fixed ordering of papers."""

    def __init__(self, papers: Sequence[Paper]) -> None:
        self.papers = list(papers)
        self._by_index: dict[int, Citation] = {}
        for i, p in enumerate(papers):
            self._by_index[i] = Citation(index=i, paper_id=p.id, label=_format_label(p))

    def label(self, index: int) -> str:
        return self._by_index[index].label if index in self._by_index else ""

    def paper_id(self, index: int) -> str:
        return self._by_index[index].paper_id if index in self._by_index else ""

    def index_of(self, paper_id: str) -> int | None:
        for i, c in self._by_index.items():
            if c.paper_id == paper_id:
                return i
        return None

    def __len__(self) -> int:
        return len(self._by_index)

    def references_md(self) -> str:
        return "\n".join(f"[{i}] {self._by_index[i].label}" for i in sorted(self._by_index))

    def to_bibtex(self) -> str:
        entries = []
        for i, p in enumerate(self.papers):
            key = _bibtex_key(p, i)
            fields = [
                ("title", _bib_escape(p.title)),
                ("author", " and ".join(_bib_escape(a) for a in p.authors)),
                ("year", p.year or ""),
            ]
            if p.doi:
                fields.append(("doi", _bib_escape(p.doi)))
            if p.url:
                fields.append(("url", _bib_escape(p.url)))
            if p.volume:
                fields.append(("volume", _bib_escape(p.volume)))
            if p.issue:
                fields.append(("number", _bib_escape(p.issue)))
            if p.pages:
                fields.append(("pages", _bib_escape(p.pages).replace("-", "--")))
            source = _bib_venue(p.source, getattr(p, "venue", ""))
            fields.append(("journal", source))
            body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields)
            entries.append(f"@article{{{key},\n  {body},\n}}")
        return "\n\n".join(entries) + "\n"

    def to_ris(self) -> str:
        """RIS export for EndNote/Zotero import (nature-citation default).

        Missing fields are simply omitted — never invented.
        """
        blocks = []
        for p in self.papers:
            lines = ["TY  - JOUR"]
            for a in p.authors:
                lines.append(f"AU  - {_bib_escape(a)}")
            lines.append(f"TI  - {_bib_escape(p.title)}")
            venue = _bib_venue(p.source, getattr(p, "venue", ""))
            lines.append(f"JO  - {venue}")
            if p.year:
                lines.append(f"PY  - {p.year}")
            if p.volume:
                lines.append(f"VL  - {_bib_escape(p.volume)}")
            if p.issue:
                lines.append(f"IS  - {_bib_escape(p.issue)}")
            pages = getattr(p, "pages", "")
            if pages:
                first, _, last = pages.partition("-")
                lines.append(f"SP  - {_bib_escape(first)}")
                if last:
                    lines.append(f"EP  - {_bib_escape(last)}")
            if p.doi:
                lines.append(f"DO  - {_bib_escape(p.doi)}")
            if p.url:
                lines.append(f"UR  - {_bib_escape(p.url)}")
            lines.append("ER  - ")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks) + "\n"


def parse_claims_from_review(markdown: str) -> list[Claim]:
    """Extract cited claims from a review's body (skips the references section).

    A "claim" is any sentence containing at least one ``[n]`` marker. Each claim
    records the indices it cites, which the Verifier later checks against source
    abstracts.
    """
    claims: list[Claim] = []
    # Split into sections by markdown headers (## or ###).
    sections = re.split(r"(?m)^(#{2,3})\s+(.*)$", markdown)
    # sections layout after split: [pre, level, title, body, level, title, body, ...]
    current_section = ""
    for i in range(1, len(sections), 3):
        title = sections[i + 1].strip()
        body = sections[i + 2]
        if _REF_HEADER_RE.search(title):
            continue
        current_section = title
        for sentence in _SENT_SPLIT_RE.split(body):
            sentence = sentence.strip()
            if not sentence:
                continue
            indices = sorted({int(m) for m in _CITE_RE.findall(sentence)})
            if indices:
                clean = _CITE_RE.sub(lambda m: f"[{m.group(1)}]", sentence)
                claims.append(
                    Claim(text=clean, citation_indices=indices, section=current_section)
                )
    return claims


def build_provenance(claims, table, ver_results=None, chunk_store=None) -> list[dict]:
    """Claim -> cited references map, for provenance.json.

    If ``ver_results`` (list of VerificationResult) is given, each claim is
    annotated with the Verifier's verdict. If ``chunk_store`` is given, each
    citation also carries the ground-text chunk a reader should look at
    (deterministic top-1 retrieval against the claim text — nature-reader's
    stable-anchor contract, cheap enough to run for every claim).
    """
    vmap: dict[str, tuple[str, str]] = {}
    if ver_results:
        for r in ver_results:
            vmap[r.claim_text] = (r.verdict.value, r.note)
    out = []
    for c in claims:
        entry = {
            "claim": c.text,
            "section": c.section,
            "citations": [
                {"index": i, "label": table.label(i), "paper_id": table.paper_id(i)}
                for i in c.citation_indices
            ],
        }
        if chunk_store is not None:
            anchors = []
            for i in c.citation_indices[:3]:
                pid = table.paper_id(i)
                if not chunk_store.has(pid):
                    continue
                for ch in chunk_store.retrieve(pid, c.text, k=1):
                    anchors.append(
                        {
                            "index": i,
                            "chunk_id": ch.chunk_id,
                            "kind": ch.kind.value,
                            "excerpt": ch.text[:200],
                        }
                    )
                    break
            if anchors:
                entry["evidence_chunks"] = anchors
        if c.text in vmap:
            verdict, note = vmap[c.text]
            entry["verdict"] = verdict
            entry["verdict_note"] = note
        out.append(entry)
    return out
