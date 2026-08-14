"""Keyword / query planning agent.

Generates English search queries regardless of the input topic language, since
every built-in source (arXiv / Semantic Scholar / OpenAlex) is an English
corpus. (Chinese queries were observed to return zero or irrelevant results.)
"""

from __future__ import annotations

from litreview.llm import chat_json

SYSTEM_PROMPT = """You are an academic literature-retrieval expert. Given a research topic, \
generate 6-8 diverse search queries.

Rules:
1. Cover different facets of the topic (methods, applications, evaluation, frontier).
2. ALWAYS produce English query strings (translate a non-English topic accurately first), \
because the targets are English databases.
3. Each query is a concise English phrase (3-6 words) suitable for an academic search engine.
4. Queries must be distinct, not near-duplicates.
5. Prefer established domain terminology.

Output JSON:
{"queries": ["english query 1", "english query 2", ...], "reasoning": "..."}"""


def generate_keywords(topic: str) -> list[str]:
    """Return 6-8 English search queries for the topic."""
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        "Generate 6-8 high-quality English search queries "
        "(translate the topic first if it is not English)."
    )
    result = chat_json(SYSTEM_PROMPT, user_prompt)
    queries = result.get("queries", [])
    return [q for q in queries if isinstance(q, str) and q.strip()][:10]
