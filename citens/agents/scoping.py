"""Interpret pre-run clarification answers (RunOptions.filters) into
retrieval-shaping inputs.

Answers arrive as a free-form ``{question_id: answer}`` dict from the CLI /
API / Web-UI clarify forms. Two consumers:

1. ``filters_block()`` — a text block appended to the planner and filter
   prompts so the LLM honors the user's scoping (sub-focus, document type,
   venue bar, timeframe, ...).
2. ``min_year_from_filters()`` — a deterministic year floor parsed from
   common timeframe phrasings ("since 2020", "近5年", "recent 3 years"),
   enforced post-hoc without spending an LLM call.
"""

from __future__ import annotations

import re
from datetime import date

_SINCE_RE = re.compile(
    r"(?:(?:since|after|from|从|自)\s*((?:19|20)\d{2})|((?:19|20)\d{2})\s*(?:年以后|以来|之后|以后|起|年後))",
    re.IGNORECASE,
)
_RECENT_RE = re.compile(r"(?:last|recent|past|近|最近)\s*(\d+)\s*(?:years?|年)", re.IGNORECASE)
_BARE_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _first_group(m: re.Match) -> int:
    return int(next(g for g in m.groups() if g))


def filters_block(filters: dict | None) -> str:
    """Render clarify answers as a prompt constraints block ("" if none)."""
    if not filters:
        return ""
    lines = [
        f"- {k}: {v}" for k, v in filters.items() if str(v).strip()
    ]
    if not lines:
        return ""
    return (
        "\n用户范围约束 / User scoping answers (HONOR these when choosing "
        "queries and judging papers; papers that clearly violate a stated "
        "constraint — e.g. outside the requested timeframe or sub-focus — "
        "must be rated lower / not searched for):\n" + "\n".join(lines) + "\n"
    )


def min_year_from_filters(filters: dict | None) -> int | None:
    """Best-effort deterministic year floor from the answers, else None."""
    if not filters:
        return None
    text = " ".join(str(v) for v in filters.values())
    m = _RECENT_RE.search(text)
    if m:
        n = min(int(m.group(1)), 50)
        return date.today().year - n + 1
    m = _SINCE_RE.search(text)
    if m:
        return _first_group(m)
    m = _BARE_YEAR_RE.search(text)
    if m:
        return int(m.group(1))
    return None
