"""Venue-aware retrieval ranking (SJR quartiles + citations + relevance).

Recall order decides which papers survive the max_papers cap. Ranking on raw
citation count starves fresh/preprint work; ranking on LLM relevance alone is
opaque. This module makes the score a weighted, EXPLAINABLE composite:

    rank = w_rel * (relevance/5)
         + w_cit * min(log10(1 + citations) / 3, 1)   # 1000+ cites saturates
         + w_ven * venue_score(SJR quartile)           # Q1=1.0 ... Q4=0.35

Venue quartiles come from the SCImago Journal Rank dataset (CC BY-NC — fetched
by ``citens sjr`` at setup time, never redistributed with the package).
Without the dataset the venue factor is a neutral 0.5, so ranking still works.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from citens.config import settings
from citens.models import ScoredPaper

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_QUARTILE_SCORES = {"q1": 1.0, "q2": 0.75, "q3": 0.55, "q4": 0.35}
NEUTRAL_VENUE = 0.5


def _norm(title: str) -> str:
    return _NON_ALNUM.sub(" ", (title or "").lower()).strip()


@dataclass(frozen=True)
class VenueInfo:
    title: str
    quartile: str  # "Q1".."Q4" or ""
    sjr: float


class SJRIndex:
    """Normalized-title lookup over SCImago data.

    Two supported formats:
      * official scimagojr.com export — semicolon-delimited, precomputed
        ``SJR Best Quartile`` column;
      * the Michael-E-Rose/SCImagoJournalRankIndicators mirror (comma-
        delimited, one row per journal/field/year) — quartiles are derived
        from each journal's latest-year SJR percentile (a global approximation
        of SCImago's field-normalized quartiles).
    """

    def __init__(self, rows: dict[str, VenueInfo]) -> None:
        self._by_title = rows

    @classmethod
    def load(cls, path: str | Path) -> SJRIndex:
        with open(path, encoding="utf-8-sig", newline="") as f:
            header = f.readline()
            delimiter = ";" if ";" in header else ","
        if "SJR Best Quartile" in header:
            return cls._load_official(path, delimiter)
        if "Sourceid" in header:
            return cls._load_mirror(path, delimiter)
        raise ValueError(f"unrecognized SJR CSV header: {header[:80]!r}")

    @classmethod
    def _load_official(cls, path: str | Path, delimiter: str) -> SJRIndex:
        rows: dict[str, VenueInfo] = {}
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                title = (row.get("Title") or "").strip()
                if not title:
                    continue
                quartile = (row.get("SJR Best Quartile") or "").strip().upper()
                try:
                    sjr = float((row.get("SJR") or "").replace(",", ".") or 0)
                except ValueError:
                    sjr = 0.0
                # later rows (newer editions appended) win
                rows[_norm(title)] = VenueInfo(title, quartile, sjr)
        return cls(rows)

    @classmethod
    def _load_mirror(cls, path: str | Path, delimiter: str) -> SJRIndex:
        # latest-year row per journal (Sourceid); SJR may be blank for old years
        latest: dict[str, tuple[int, float]] = {}  # norm title -> (year, sjr)
        titles: dict[str, str] = {}
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                title = (row.get("Title") or "").strip()
                if not title:
                    continue
                try:
                    year = int(row.get("year") or 0)
                    sjr = float((row.get("SJR") or "").replace(",", ".") or 0)
                except ValueError:
                    continue
                key = _norm(title)
                titles.setdefault(key, title)
                prev = latest.get(key)
                if prev is None or (year, sjr) > (prev[0], prev[1]):
                    latest[key] = (year, sjr)
        values = sorted(v for _, v in latest.values() if v > 0)

        def _pct(p: float) -> float:
            if not values:
                return 0.0
            return values[max(0, math.ceil(p * len(values)) - 1)]

        # Field-normalized official quartiles are unavailable in this mirror;
        # top-10/25/50% global cutoffs approximate them conservatively (a
        # mid-tier journal should not earn a global "Q1").
        p90, p75, p50 = _pct(0.90), _pct(0.75), _pct(0.50)
        rows: dict[str, VenueInfo] = {}
        for key, (_year, sjr) in latest.items():
            if sjr <= 0:
                q = ""
            elif sjr >= p90:
                q = "Q1"
            elif sjr >= p75:
                q = "Q2"
            elif sjr >= p50:
                q = "Q3"
            else:
                q = "Q4"
            rows[key] = VenueInfo(titles[key], q, sjr)
        return cls(rows)

    def __len__(self) -> int:
        return len(self._by_title)

    def lookup(self, venue: str) -> VenueInfo | None:
        """Exact normalized match, then a STRICT containment fallback.

        Containment requires the shorter side to cover >= 60% of the longer,
        so decorative suffixes ("... — Special Issue") still match while
        unrelated titles sharing a few words never do. Unknown journals
        (not in the dataset) return None -> neutral venue factor.
        """
        if not venue:
            return None
        key = _norm(venue)
        hit = self._by_title.get(key)
        if hit:
            return hit
        if len(key) < 12:  # too short to containment-match safely
            return None
        best: tuple[str, VenueInfo] | None = None
        for title, info in self._by_title.items():
            if len(title) < 12:
                continue
            covered = (
                title in key and len(title) >= 0.6 * len(key)
            ) or (key in title and len(key) >= 0.6 * len(title))
            if covered and (best is None or len(title) > len(best[0])):
                best = (title, info)
        return best[1] if best else None


def convert_rda_to_csv(rda_path: str | Path, csv_path: str | Path) -> int:
    """Convert the ikashnitsky/sjrdata ``sjr_journals.rda`` (all years, official
    field-normalized quartiles) into the official-format CSV, keeping each
    journal's latest-year row. Returns the number of journals written.
    Requires the optional ``pyreadr`` dependency."""
    import pyreadr  # lazy: only needed by `citens sjr`

    result = pyreadr.read_r(str(rda_path))
    df = list(result.values())[0]
    df = df.sort_values("year").drop_duplicates("sourceid", keep="last")

    def _ok(v):  # NaN guard (rda uses float NaN for missing strings)
        return v == v

    n = 0
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Rank", "Title", "Type", "Issn", "SJR", "SJR Best Quartile", "H index"])
        for _, row in df.iterrows():
            title = row["title"] if _ok(row["title"]) else ""
            if not str(title).strip():
                continue
            sjr = round(float(row["sjr"]), 3) if _ok(row["sjr"]) else ""
            q = row["sjr_best_quartile"] if _ok(row["sjr_best_quartile"]) else ""
            h = int(row["h_index"]) if _ok(row["h_index"]) else ""
            issn = row["issn"] if _ok(row["issn"]) else ""
            typ = row["type"] if _ok(row["type"]) else ""
            rank = int(row["rank"]) if _ok(row["rank"]) else ""
            w.writerow([rank, title, typ, issn, sjr, q, h])
            n += 1
    return n


_index: SJRIndex | None = None
_index_loaded = False


def get_sjr_index() -> SJRIndex | None:
    """Process-wide singleton; None when the CSV hasn't been downloaded."""
    global _index, _index_loaded
    if not _index_loaded:
        _index_loaded = True
        path = Path(settings.sjr_csv_path)
        if path.is_file():
            try:
                _index = SJRIndex.load(path)
            except Exception as e:  # noqa: BLE001
                print(f"  [rank] SJR data unreadable ({e}); venue factor neutral")
    return _index


def venue_score(quartile: str) -> float:
    return _QUARTILE_SCORES.get((quartile or "").lower(), NEUTRAL_VENUE)


def citation_factor(citations: int) -> float:
    return min(math.log10(1 + max(citations, 0)) / 3.0, 1.0)


def author_depth_factor(h_index: int, works: int, field_works: int = 0) -> float | None:
    """Author-engagement score in [0,1]; None when the signal is unknown.

    ``field_works`` (in-topic works, from citens collect) is the real
    深耕此领域 signal and is preferred; the total-works fallback is
    merged-author-artifact prone (OpenAlex merges same-named authors).
    Unknown (all zero) returns None so ranking can EXCLUDE the factor
    instead of punishing the paper for missing metadata.
    """
    if field_works > 0:
        w = min(math.log10(1 + field_works) / math.log10(31), 1.0)  # 30 in-field works saturates
        h = min(h_index / 40.0, 1.0)
        return 0.7 * w + 0.3 * h
    if works <= 0 and h_index <= 0:
        return None
    w = min(math.log10(1 + max(works, 1)) / 2.0, 1.0)  # 100 works saturates
    h = min(h_index / 40.0, 1.0)
    return 0.6 * w + 0.4 * h


def rank_papers(
    papers: Sequence[ScoredPaper], venue_boost: set[str] | None = None
) -> list[ScoredPaper]:
    """Fill venue_quartile/rank_score on copies, sorted by rank_score desc.

    Tie-break on relevance then citations so equal composites keep the more
    relevant / more cited paper first.
    """
    index = get_sjr_index()
    ranked: list[ScoredPaper] = []
    for p in papers:
        q = p.venue_quartile
        if not q and p.venue:
            info = index.lookup(p.venue) if index else None
            q = info.quartile if info else ""
        rel = p.relevance_score / 5.0
        cit = citation_factor(p.citation_count)
        ven = venue_score(q)
        if venue_boost and p.venue and _norm(p.venue) in venue_boost:
            ven = 1.0  # flagship field journal == Q1-level venue signal
        # weighted composite, renormalized over the factors a paper actually
        # has (author depth is optional metadata; its absence must not hurt)
        parts = [
            (settings.rank_weight_relevance, rel),
            (settings.rank_weight_citations, cit),
            (settings.rank_weight_venue, ven),
        ]
        dep = author_depth_factor(
            p.first_author_h_index, p.first_author_works, p.author_field_works
        )
        if dep is not None:
            parts.append((settings.rank_weight_author, dep))
        wsum = sum(w for w, _ in parts)
        score = sum(w * f for w, f in parts) / wsum if wsum else 0.0
        ranked.append(
            p.model_copy(update={"venue_quartile": q, "rank_score": round(score, 4)})
        )
    ranked.sort(key=lambda p: (p.rank_score, p.relevance_score, p.citation_count), reverse=True)
    return ranked


def quartile_histogram(papers: list[ScoredPaper]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for p in papers:
        key = p.venue_quartile or "unranked"
        hist[key] = hist.get(key, 0) + 1
    return hist
