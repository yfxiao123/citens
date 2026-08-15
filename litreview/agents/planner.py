"""Keyword / query planning agent with iterative expansion.

Generates English search queries regardless of the input topic language, since
every built-in source (arXiv / Semantic Scholar / OpenAlex) is an English
corpus. (Chinese queries were observed to return zero or irrelevant results.)

v2 adds:
- Dimension coverage: queries must span methods/applications/theory/empirical/survey
- Iterative refinement: given initial search results, generate refined queries
  targeting gaps and discovered subtopics
- Semantic orthogonality: penalize near-duplicate queries
"""

from __future__ import annotations

from litreview.agents.scoping import filters_block
from litreview.llm import chat_json

SYSTEM_PROMPT = """You are an academic literature-retrieval expert. Given a research topic, \
generate 6-10 diverse search queries that MAXIMIZE COVERAGE.

Rules:
1. COVER ALL DIMENSIONS — your queries must span:
   - Methods: "limit order book queueing model"
   - Applications: "order book market making strategy"
   - Theory: "price formation limit order markets"
   - Empirical: "empirical order book stylized facts"
   - Survey/Review: "limit order book survey"
2. ALWAYS produce English query strings (translate a non-English topic first).
3. Each query is a concise English phrase (3-6 words).
4. Queries must be SEMANTICALLY ORTHOGONAL — not near-duplicates or paraphrases.
5. Prefer established domain terminology (exact terms researchers use).
6. Include at least one query targeting RECENT work (add "2023..2026" or "recent").

Output JSON:
{"queries": ["...", "..."], "dimensions_covered": ["methods", "applications", ...], \
"reasoning": "..."}"""

REFINE_PROMPT = """You are an academic literature-retrieval expert doing ITERATIVE QUERY REFINEMENT.

You previously searched for a topic and got initial results. Now you need to REFINE \
the queries based on what you found and what's still missing.

Context:
- Original topic
- Queries already used
- Summary of what was found (titles of top results)
- Known gaps (sub-areas not yet covered)

Generate 3-5 NEW queries that:
1. Target the GAPS — sub-areas where no good results were found
2. Use DIFFERENT terminology than the original queries (synonyms, related terms)
3. Are still semantically orthogonal to each other
4. Cover aspects the initial search missed (check dimension coverage)

Output JSON:
{"queries": ["...", "..."], "targeted_gaps": ["gap1", "gap2"], \
"reasoning": "why these queries will find what's missing"}"""


def generate_keywords(topic: str, filters: dict | None = None) -> list[str]:
    """Return 6-10 English search queries covering all topic dimensions.

    Args:
        topic: Research topic (any language; queries are always English).
        filters: Pre-run clarification answers ({question_id: answer}) —
            rendered as constraints the queries must honor (sub-focus,
            timeframe, document type, ...).
    """
    user_prompt = (
        f"研究主题 / Topic: {topic}\n"
        f"{filters_block(filters)}"
        "\nGenerate 6-10 diverse English search queries covering methods, "
        "applications, theory, empirical findings, and surveys. "
        "(Translate the topic first if it is not English.)"
    )
    result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=2048)
    queries = result.get("queries", [])
    return [q for q in queries if isinstance(q, str) and q.strip()][:12]


def refine_queries(
    topic: str,
    original_queries: list[str],
    found_titles: list[str],
    known_gaps: list[str],
) -> list[str]:
    """Generate refined queries targeting gaps discovered in the first round.

    Args:
        topic: The research topic
        original_queries: Queries already used
        found_titles: Titles of top results from the initial search
        known_gaps: Sub-areas identified as not yet covered

    Returns:
        3-5 new English queries targeting the gaps
    """
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"已用查询 / Original queries:\n"
        + "\n".join(f"- {q}" for q in original_queries[:8])
        + "\n\n已找到的论文 / Found papers (titles):\n"
        + "\n".join(f"- {t}" for t in found_titles[:10])
        + "\n\n已知空白 / Known gaps:\n"
        + "\n".join(f"- {g}" for g in known_gaps[:5])
        + "\n\nGenerate refined queries targeting the gaps."
    )
    try:
        result = chat_json(REFINE_PROMPT, user_prompt, max_tokens=1536)
        queries = result.get("queries", [])
        # Deduplicate against original queries
        orig_lower = {q.lower().strip() for q in original_queries}
        return [
            q for q in queries
            if isinstance(q, str) and q.strip() and q.lower().strip() not in orig_lower
        ][:5]
    except Exception:  # noqa: BLE001
        return []
