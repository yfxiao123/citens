"""Clarification answers -> retrieval-side constraints.

The pre-run clarification shapes the FILTER stage only — which produced the
"63 candidates -> 6 pass" failure mode: the user demanded top journals and a
recent window, but retrieval kept surfacing whatever the pool happened to
contain, and the filter then killed it. Constraints belong in retrieval too:

* a timeframe answer becomes a publication-year window applied to the pool
  recall AND to fresh OpenAlex queries (``from_publication_date``);
* a "top journals only" answer switches on venue-strict mode — the pool
  recall keeps only whitelisted venues (from the domain profile), and a
  venue-restricted OpenAlex search adds top-journal papers the pool lacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NEAR_N_RE = re.compile(r"近\s*(\d+)\s*年")
_RANGE_RE = re.compile(r"(20\d{2})\s*[-–—]\s*(20\d{2})")
_FROM_RE = re.compile(r"(19|20)(\d{2})\s*年至今")


@dataclass(frozen=True)
class RetrievalConstraints:
    year_from: int | None = None
    year_to: int | None = None
    venue_strict: bool = False

    def matches_paper(self, paper) -> bool:
        """Year-window membership for a pool/candidate record."""
        if self.year_from and (not paper.year or paper.year < self.year_from):
            return False
        return not (self.year_to and (not paper.year or paper.year > self.year_to))

    def describe(self) -> str:
        bits = []
        if self.year_from or self.year_to:
            bits.append(f"{self.year_from or '…'}–{self.year_to or '…'}")
        if self.venue_strict:
            bits.append("仅白名单期刊")
        return " / ".join(bits)


def parse_constraints(filters: dict | None) -> RetrievalConstraints:
    """Best-effort structured read of the free-text clarification answers.

    Question ids are LLM-generated (timeframe/venue/…), so scan by key hints
    first, then fall back to scanning every answer's text.
    """
    filters = filters or {}
    year_from = year_to = None
    venue_strict = False

    time_val = _by_key_hints(filters, ("time", "year", "period", "时间", "年份"))
    if time_val:
        year_from, year_to = _parse_window(str(time_val))
    else:
        for v in filters.values():
            yf, yt = _parse_window(str(v))
            if yf or yt:
                year_from, year_to = yf, yt
                break

    venue_val = _by_key_hints(filters, ("venue", "journal", "source", "期刊", "来源"))
    candidates = [str(venue_val)] if venue_val else [str(v) for v in filters.values()]
    for text in candidates:
        if "仅" in text and ("顶" in text or "top" in text.lower()):
            venue_strict = True
            break

    return RetrievalConstraints(year_from=year_from, year_to=year_to,
                                venue_strict=venue_strict)


def _by_key_hints(filters: dict, hints: tuple) -> str:
    for k, v in filters.items():
        kl = str(k).lower()
        if any(h.lower() in kl for h in hints):
            return str(v)
    return ""


def _parse_window(text: str) -> tuple[int | None, int | None]:
    """'近5年（2019-2024）' etc. — the 近N年 phrase wins over stale explicit
    years (LLM-generated option text often carries its training-cutoff years;
    the relative phrase is what the user actually chose)."""
    m = _NEAR_N_RE.search(text)
    if m:
        n = int(m.group(1))
        import datetime

        cy = datetime.date.today().year
        return cy - n + 1, cy
    m = _FROM_RE.search(text)
    if m:
        return int(m.group(1) + m.group(2)), None
    m = _RANGE_RE.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None
