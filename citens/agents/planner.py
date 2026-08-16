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

from citens.agents.scoping import filters_block
from citens.llm import chat_json

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


SEED_PROMPT = """You are an academic literature-retrieval expert. Name the LANDMARK papers of a \
research field: the canonical, highly-cited works any survey of this topic MUST cover.

Rules:
1. Name 3-5 real, well-known papers (title as published, in English).
2. Prefer papers that are demonstrably canonical (hundreds+ of citations, textbook references).
3. If you are not confident a paper exists under that exact title, omit it —
   a hallucinated title wastes a retrieval round-trip.
4. Also give 2-4 established domain TERMS used in this literature that a
   newcomer would likely miss (jargon, method names, model families).

Output JSON:
{"papers": ["exact title 1", "exact title 2"], \
"domain_terms": ["term 1", "term 2"], "reasoning": "..."}"""


def generate_seed_papers(topic: str, filters: dict | None = None) -> tuple[list[str], list[str]]:
    """Return (landmark titles, established domain terms) for the topic.

    Titles feed seed-paper expansion (see citens.search.seeds); terms are
    folded into the keyword list when they are not already covered.
    """
    result = chat_json(
        SEED_PROMPT,
        f"研究主题 / Topic: {topic}\n{filters_block(filters)}",
        max_tokens=1024,
    )
    titles = [t for t in result.get("papers", []) if isinstance(t, str) and t.strip()][:5]
    terms = [t for t in result.get("domain_terms", []) if isinstance(t, str) and t.strip()][:4]
    return titles, terms


def discover_terms(papers: list, top_k: int = 12) -> list[str]:
    """Deterministic keyword mining from already-retrieved titles/abstracts.

    Frequent content bigrams/unigrams that the current queries do not contain —
    the cheap, LLM-free half of iterative search refinement. Feed to
    :func:`refine_queries` so gap-targeted queries use terminology the field
    actually uses, not the model's guess.
    """
    import re
    from collections import Counter

    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
        "have", "has", "its", "our", "their", "these", "those", "between", "into",
        "such", "can", "using", "used", "use", "based", "study", "studies", "paper",
        "papers", "article", "results", "show", "shown", "propose", "proposed",
        "approach", "model", "models", "method", "methods", "data", "analysis",
        "also", "than", "then", "when", "where", "which", "while", "both", "more",
        "most", "less", "least", "over", "under", "within", "across", "among",
    }

    def tokens(text: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2]

    unigram: Counter = Counter()
    bigram: Counter = Counter()
    for p in papers[:20]:
        text = f"{p.title} {p.abstract[:500]}"
        toks = [t for t in tokens(text) if t not in stop and not t.isdigit()]
        unigram.update(toks)
        bigram.update(zip(toks, toks[1:], strict=False))

    # bigrams first (more specific), then unigrams, both by frequency
    all_terms = (
        [f"{a} {b}" for (a, b), c in bigram.most_common() if c >= 3]
        + [w for w, c in unigram.most_common() if c >= 4]
    )
    seen: set[str] = set()
    out: list[str] = []
    for t in all_terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= top_k:
            break
    return out


def refine_queries(
    topic: str,
    original_queries: list[str],
    found_titles: list[str],
    known_gaps: list[str],
    discovered_terms: list[str] | None = None,
) -> list[str]:
    """Generate refined queries targeting gaps discovered in the first round.

    Args:
        topic: The research topic
        original_queries: Queries already used
        found_titles: Titles of top results from the initial search
        known_gaps: Sub-areas identified as not yet covered
        discovered_terms: Terminology mined from retrieved papers
            (see :func:`discover_terms`) — use these where they fit

    Returns:
        3-5 new English queries targeting the gaps
    """
    terms_block = ""
    if discovered_terms:
        terms_block = (
            "\n\n领域实际使用的术语 / Terminology the field actually uses "
            "(PREFER these in new queries where relevant):\n"
            + "\n".join(f"- {t}" for t in discovered_terms[:12])
        )
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"已用查询 / Original queries:\n"
        + "\n".join(f"- {q}" for q in original_queries[:8])
        + "\n\n已找到的论文 / Found papers (titles):\n"
        + "\n".join(f"- {t}" for t in found_titles[:10])
        + terms_block
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
