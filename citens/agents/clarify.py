"""Pre-run clarification agent.

Literature review is fundamentally interactive, but our pipeline has been
one-shot: it guesses everything from the topic string. This agent surfaces a
small set of high-value questions BEFORE the run starts, so the user can shape
the search (language, scope, depth) in a couple of clicks instead of rerunning
later. The questions are opinionated (field / venue level / language /
output language / timeframe) and each has suggested options so the UI can
render them as a form, not free text.

The API returns the questions; the CLI blocks on typed answers; the Web UI
renders them inline before POSTing /run. Answers (if any) feed the pipeline
via RunOptions.filters.
"""

from __future__ import annotations

import datetime
import re

from citens.llm import chat_json

# option text carrying an explicit year RANGE next to a 近N年 phrase is
# stale by construction — the LLM wrote its training-cutoff years; rewrite
# the range relative to today so "近5年" is actually the last 5 years
_STALE_RANGE_RE = re.compile(r"(?P<lead>近\s*\d+\s*年\s*[（(])(?P<y1>20\d{2})[-–—](?P<y2>20\d{2})(?P<tail>[）)])")

SYSTEM_PROMPT = """You are a research-scoping assistant. Given a research topic, generate 2-4 \
clarifying questions that would meaningfully shape a literature review OF THIS TOPIC.

Rules:
1. Each question must be answerable by a short, discrete choice (not open-ended prose).
2. Provide 3-5 concrete suggested answers (options) per question — these become UI choices.
3. Cover the dimensions that actually change the search: sub-field focus, scope (e.g. \
surveys only / methods / applications), timeframe, output language, venue quality bar. \
Do NOT ask generic questions (e.g. "what is your goal?").
4. Keep questions few and high-value — a user should answer them in under 20 seconds.
5. If the topic is already specific enough that nothing would change the search, return an \
empty list.
6. Write the QUESTIONS and OPTIONS in {lang_name} — the user answers these in the UI \
before any review exists; they must read like a native form, not a translation exercise. \
Keep ids in English.

Output JSON:
{"questions": [
  {"id": "focus", "question": "Which sub-focus?", "options": ["opt a", "opt b", "opt c"], "default": "opt a"}
]}"""

_LANG_NAMES = {"zh": "简体中文", "en": "English"}


def generate_clarifying_questions(topic: str) -> list[dict]:
    """Return 0-4 clarifying questions (each: id/question/options/default).

    Question/option text follows REVIEW_LANGUAGE (Chinese by default) — the
    pre-run form is the first thing a user reads; it should match the review
    language they will get.
    """
    from citens.config import settings

    lang = _LANG_NAMES.get(settings.review_language.strip().lower(), "English")
    try:
        result = chat_json(
            SYSTEM_PROMPT.replace("{lang_name}", lang),
            f"研究主题 / Topic: {topic}",
            max_tokens=1536, cheap=True,
        )
    except Exception:  # noqa: BLE001
        return []
    questions = result.get("questions", [])
    out: list[dict] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id", "")).strip()
        qtext = str(q.get("question", "")).strip()
        options = [str(o).strip() for o in q.get("options", []) if str(o).strip()]
        if qid and qtext and len(options) >= 2:
            default = str(q.get("default", options[0])).strip() or options[0]
            options = [_fresh_years(o) for o in options[:5]]
            default = _fresh_years(default)
            out.append(
                {
                    "id": qid,
                    "question": qtext,
                    "options": options,
                    "default": default,
                }
            )
    return out[:4]


def _fresh_years(option: str) -> str:
    """Rewrite stale year ranges in timeframe options relative to today.

    The LLM generates options from its training cutoff ("近5年（2019-2024）"
    in 2026 is a six-year-stale window); the relative phrase is the user's
    actual intent, so recompute the explicit years from it."""
    cy = datetime.date.today().year

    def _recount(m: re.Match) -> str:
        n = int(re.search(r"\d+", m.group("lead")).group())  # type: ignore[union-attr]
        return f"{m.group('lead')}{cy - n + 1}-{cy}{m.group('tail')}"

    if _STALE_RANGE_RE.search(option):
        return _STALE_RANGE_RE.sub(_recount, option)
    # "2000年至今"-style: keep the anchor year, nothing to recompute
    return option


def questions_to_query_filters(questions: list[dict]) -> dict:
    """Render answered questions into retrieval-hint keywords (best-effort).

    Each answer becomes an additive search hint the pipeline can append to
    queries (e.g. a chosen sub-focus or timeframe). Not every question maps to
    a filter — only those whose option text is a searchable phrase.
    """
    hints: list[str] = []
    for q in questions:
        # questions carry a chosen answer? we only have defaults here; the
        # actual answers come from the UI/CLI. This helper is used after the
        # user answers, with the answer stored in q["answer"].
        ans = q.get("answer")
        if ans and str(ans).strip():
            hints.append(str(ans).strip())
    return {"query_hints": hints}
