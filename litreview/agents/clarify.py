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

from litreview.llm import chat_json

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

Output JSON:
{"questions": [
  {"id": "focus", "question": "Which sub-focus?", "options": ["opt a", "opt b", "opt c"], "default": "opt a"}
]}"""


def generate_clarifying_questions(topic: str) -> list[dict]:
    """Return 0-4 clarifying questions (each: id/question/options/default)."""
    try:
        result = chat_json(SYSTEM_PROMPT, f"研究主题 / Topic: {topic}", max_tokens=1536)
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
            out.append(
                {
                    "id": qid,
                    "question": qtext,
                    "options": options[:5],
                    "default": str(q.get("default", options[0])).strip() or options[0],
                }
            )
    return out[:4]


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
