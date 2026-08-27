"""Keyword / query planning agent with iterative expansion.

Generates English search queries regardless of the input topic language, since
every built-in source (arXiv / Semantic Scholar / OpenAlex) is an English
corpus. (Chinese queries were observed to return zero or irrelevant results.)

v3 restructures planning around CONCEPT BLOCKS (the librarian / PRESS
method for systematic-review searches): the LLM returns core concepts with
synonym variants, and deterministic code assembles the query list from them —

- one coverage query per concept's primary term,
- a few precision queries concatenating the most central concept pairs,
- synonyms held in reserve: they are swapped in later for queries that
  returned ZERO hits (the test-search calibration loop), instead of
  inflating round one.

v2 adds:
- Dimension coverage: concepts must span methods/applications/theory/empirical/survey
- Iterative refinement: given initial search results, generate refined queries
  targeting gaps and discovered subtopics
- Semantic orthogonality: penalize near-duplicate queries
"""

from __future__ import annotations

from dataclasses import dataclass, field

from citens.agents.scoping import filters_block
from citens.llm import chat_json

SYSTEM_PROMPT = """You are an academic literature-retrieval expert. Decompose a research \
topic into CONCEPT BLOCKS, the way a research librarian builds a systematic-review \
search strategy.

Rules:
1. Extract 4-6 core concepts of the topic. Together they must span the field's \
dimensions: methods, applications, theory, empirical findings, surveys / recent \
advances (e.g. for limit order books: "limit order book" / "market making" / \
"price impact" / "order book stylized facts" / "limit order book survey").
2. For EACH concept give 2-4 SYNONYMS or established variants — jargon, \
abbreviations, adjacent phrasings researchers actually write. These recover \
papers the primary term misses (a query "generative recommendation" never \
finds a paper that only says "GenRec").
3. ALWAYS English terms (translate a non-English topic first).
4. Terms are concise (1-4 words) and use established domain terminology.
5. Concepts must be DISTINCT — not paraphrases of each other.

Output JSON:
{"concepts": [{"term": "...", "synonyms": ["...", "..."]}, ...], "reasoning": "..."}"""

# max concepts that get pairwise precision combos (the "most central" ones —
# the LLM lists concepts roughly in centrality order)
_COMBO_CONCEPTS = 3
_MAX_QUERIES = 12

# terms that name a DOCUMENT KIND or a broad field, not this topic's subject:
# searched standalone they return the most-cited generic mega-surveys ever
# written (measured: "survey" pulled SF-36, 30k citations, into a generative-
# recommendation pool). They must be anchored to the topic's central concept.
_GENERIC_TERMS = {
    "survey", "surveys", "review", "reviews", "overview", "tutorial",
    "recent advances", "empirical study", "empirical analysis", "applications",
    "case study", "deep learning", "machine learning", "neural networks",
    "generative models", "large language models", "foundation models",
    "benchmark", "evaluation",
}


def _anchor_generic(term: str, central: str | None) -> str:
    """Attach a generic kind/field term to the topic's central concept.

    "survey" alone matches every discipline; "generative recommendation
    survey" matches surveys OF THIS TOPIC. Duplicate words collapse
    ("generative recommendation" + "generative models" -> "... models").
    """
    if central is None or term.lower().strip() not in _GENERIC_TERMS:
        return term
    words: list[str] = []
    for w in f"{central} {term}".split():
        if w.lower() not in {x.lower() for x in words}:
            words.append(w)
    return " ".join(words)


@dataclass
class QueryPlan:
    """A round-one search plan: assembled queries + the blocks they came from."""

    queries: list[str] = field(default_factory=list)
    concepts: list[dict] = field(default_factory=list)

    def synonyms_for(self, query: str) -> list[str]:
        """Untried synonyms of the concept a (zero-hit) query came from."""
        ql = query.lower().strip()
        for c in self.concepts:
            term = str(c.get("term", "")).lower().strip()
            if not term:
                continue
            if ql == term or term in ql or ql in term:
                return [
                    s for s in (c.get("synonyms") or [])
                    if isinstance(s, str) and s.strip()
                ]
        return []


def assemble_queries(concepts: list[dict], max_queries: int = _MAX_QUERIES) -> list[str]:
    """Deterministically assemble the searched query list from concept blocks.

    Coverage first (one query per primary term), then precision combos of the
    most central concept pairs, deduplicated case-insensitively. A combo whose
    text is contained in an existing query adds nothing (the APIs are
    bag-of-words AND) and is skipped.
    """
    terms = [
        str(c.get("term", "")).strip()
        for c in concepts
        if str(c.get("term", "")).strip()
    ]
    central = next((t for t in terms if t.lower().strip() not in _GENERIC_TERMS), None)
    terms = [_anchor_generic(t, central) for t in terms]
    out: list[str] = []
    seen: set[str] = set()

    def _add(q: str) -> None:
        ql = q.lower().strip()
        if not ql or ql in seen:
            return
        seen.add(ql)
        out.append(q.strip())

    for t in terms:
        _add(t)
    for i in range(min(len(terms), _COMBO_CONCEPTS)):
        for j in range(i + 1, min(len(terms), _COMBO_CONCEPTS)):
            if len(out) >= max_queries:
                break
            combo = f"{terms[i]} {terms[j]}"
            # only skip when an EXISTING query already contains the combo —
            # a broader query subsumes the narrower one under AND semantics.
            # (the reverse containment is the normal case: the combo narrows
            # its shorter ingredient by adding the other concept's words)
            if any(combo.lower() in q.lower() for q in out):
                continue
            _add(combo)
    return out[:max_queries]


def _fallback_plan(topic: str) -> QueryPlan:
    """LLM failure / malformed output must never abort the run."""
    return QueryPlan(queries=[topic][:200], concepts=[{"term": topic[:200], "synonyms": []}])


def plan_queries(topic: str, filters: dict | None = None) -> QueryPlan:
    """Plan round one: concept blocks (LLM) -> assembled queries (code).

    Args:
        topic: Research topic (any language; terms are always English).
        filters: Pre-run clarification answers ({question_id: answer}) —
            rendered as constraints the concepts must honor.
    """
    user_prompt = (
        f"研究主题 / Topic: {topic}\n"
        f"{filters_block(filters)}"
        "\nDecompose the topic into concept blocks with synonyms."
        " (Translate the topic first if it is not English.)"
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=2048, thinking=False, cheap=True)
        raw = result.get("concepts", [])
        concepts = [
            {
                "term": str(c.get("term", "")).strip()[:80],
                "synonyms": [
                    str(s).strip()[:80]
                    for s in (c.get("synonyms") or [])
                    if isinstance(s, str) and s.strip()
                ][:4],
            }
            for c in raw
            if isinstance(c, dict) and str(c.get("term", "")).strip()
        ][:6]
    except Exception:  # noqa: BLE001 — planning must degrade, not fail
        return _fallback_plan(topic)
    if not concepts:
        return _fallback_plan(topic)
    return QueryPlan(queries=assemble_queries(concepts), concepts=concepts)


def generate_keywords(topic: str, filters: dict | None = None) -> list[str]:
    """Backward-compatible flat list (collect.py and older callers)."""
    return plan_queries(topic, filters).queries


def synonym_fallback_queries(
    plan: QueryPlan,
    zero_hit_queries: list[str],
    already_searched: list[str],
    cap: int = 4,
) -> list[str]:
    """PRESS-style calibration: swap synonyms in for queries that hit nothing.

    A zero-hit query means the field does not use that phrasing — its
    concept's untried synonyms are exactly the variants to try next.
    """
    searched = {q.lower().strip() for q in already_searched}
    out: list[str] = []
    for q in zero_hit_queries:
        for syn in plan.synonyms_for(q):
            sl = syn.lower().strip()
            if sl and sl not in searched and syn not in out:
                out.append(syn)
            if len(out) >= cap:
                return out
    return out


def low_yield_synonym_swaps(
    plan: QueryPlan,
    yield_rows: list[dict],
    already_searched: list[str],
    cap: int = 3,
) -> list[str]:
    """Untried synonyms of directions that over-fetched off-topic material.

    The reflect-loop extension of the calibration wave: a direction with
    many pool hits but zero filter survivors retrieves the WRONG literature
    under that phrasing — its concept's synonyms are the cheapest
    replacement queries (no new LLM planning needed).
    """
    searched = {q.lower().strip() for q in already_searched}
    out: list[str] = []
    for r in yield_rows or []:
        for syn in plan.synonyms_for(str(r.get("concept", ""))):
            sl = syn.lower().strip()
            if sl and sl not in searched and syn not in out:
                out.append(syn)
            if len(out) >= cap:
                return out
    return out

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
        max_tokens=1024, cheap=True,
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
    zero_hit_queries: list[str] | None = None,
) -> list[str]:
    """Generate refined queries targeting gaps discovered in the first round.

    Args:
        topic: The research topic
        original_queries: Queries already used
        found_titles: Titles of top results from the initial search
        known_gaps: Sub-areas identified as not yet covered
        discovered_terms: Terminology mined from retrieved papers
            (see :func:`discover_terms`) — use these where they fit
        zero_hit_queries: Queries that returned zero hits across all
            sources — the field does not use that phrasing; refinement
            should replace them with different terminology

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
    zero_block = ""
    if zero_hit_queries:
        zero_block = (
            "\n\n零命中查询 / Queries that returned NOTHING (the field does not "
            "use this phrasing — replace with different terminology):\n"
            + "\n".join(f"- {q}" for q in zero_hit_queries[:6])
        )
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"已用查询 / Original queries:\n"
        + "\n".join(f"- {q}" for q in original_queries[:8])
        + "\n\n已找到的论文 / Found papers (titles):\n"
        + "\n".join(f"- {t}" for t in found_titles[:10])
        + terms_block
        + zero_block
        + "\n\n已知空白 / Known gaps:\n"
        + "\n".join(f"- {g}" for g in known_gaps[:5])
        + "\n\nGenerate refined queries targeting the gaps."
    )
    try:
        result = chat_json(REFINE_PROMPT, user_prompt, max_tokens=1536, cheap=True)
        queries = result.get("queries", [])
        # Deduplicate against original queries
        orig_lower = {q.lower().strip() for q in original_queries}
        return [
            q for q in queries
            if isinstance(q, str) and q.strip() and q.lower().strip() not in orig_lower
        ][:5]
    except Exception:  # noqa: BLE001
        return []

FACETS_PROMPT = """You are a systematic-review methodologist. Split the research topic into 5-8 SEARCH FACETS — the sub-directions a comprehensive survey of this field must cover.

Facet design rules:
- Cover: foundational/classic works, systematic surveys, each major method family, data & evaluation, applications, and the most recent advances (LLM-era, new architectures).
- Facets should not overlap heavily; together they must cover the field.
- For each facet give 2-3 diverse ENGLISH search queries.

Output JSON only:
{"facets": [{"name": "short facet name", "queries": ["query 1", "query 2"]}]}"""


def generate_facets(topic: str, filters: dict | None = None) -> list[dict]:
    """Plan the topic's search facets (the coverage-by-design layer).

    Keywords fan out per query; facets make coverage *measurable*: the
    pipeline counts papers per facet, feeds thin facets to the reflector's
    supplementary retrieval, and hands the writer an honest coverage
    statement. Mechanical call — thinking off.
    """
    user_prompt = (
        f"研究主题 / Topic: {topic}\n"
        f"{filters_block(filters)}\n\n"
        "Plan the search facets."
    )
    try:
        result = chat_json(
            FACETS_PROMPT, user_prompt, max_tokens=2048, thinking=False, cheap=True
        )
    except Exception:  # noqa: BLE001 — facets are an accelerator, not a pillar
        return []
    facets = result.get("facets", [])
    return [
        {"name": str(f.get("name", ""))[:60],
         "queries": [str(q) for q in f.get("queries", [])][:3]}
        for f in facets if f.get("name") and f.get("queries")
    ][:8]
