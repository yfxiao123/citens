"""Hypothetical-title queries (HyDE, question side of the vocabulary wall).

LitSearch-measured failure mode: the only lexical retrieval shape that
reliably crosses the vocabulary wall is a near-exact-title match (crossref
rank 0 for a gold titled "MISC: A Mixed Strategy-Aware Model integrating
COMET for Emotional Support Conversation") — canonical subfield phrases and
keyword combos stall out below rank 40. HyDE/query2doc (Gao et al. 2023;
Wang et al. 2023) close exactly this query<->document vocabulary gap by
having the LLM write the pseudo-document first; our sources have no
embedding endpoint, so the pseudo-document is a TITLE, searched as plain
text where title matching lives (crossref).

Complements the PRF pivot (corpus side): pivot mines REAL entity names from
neighbor abstracts (works when the gold carries an unguessable name such as
"SemEval-2022 Task 4"); hypothetical titles cover golds whose titles are
plain multi-concept descriptions of the question.
"""

from __future__ import annotations

from citens.llm import chat_json

SYSTEM_PROMPT = """You write plausible paper TITLES for a literature-search \
question.

Retrieval evidence: keyword searches with the question's own words miss the \
target papers; matching against realistic titles of the ANSWERING papers \
finds them. You produce those titles.

Rules:
1. 3-5 titles, each a realistic academic paper title that would fully answer \
the question (venue-style, capitalized, 4-14 words).
2. MIX two kinds: (a) descriptive multi-concept titles combining the \
question's core concepts exactly as a paper would phrase them, and (b) if a \
specific task, benchmark, or method name is plausibly involved, a title \
containing that name verbatim.
3. Do NOT hedge or explain; titles only. No question marks.
4. Reply ONLY JSON: {"titles": ["...", ...]}."""


def hypothetical_queries(question: str, k: int = 4) -> list[str]:
    """Title-shaped queries for the answering papers (HyDE variant).

    Returns [] on any LLM failure — a hypothesis that cannot be generated
    must degrade to "no extra queries", never fail the caller.
    """
    try:
        result = chat_json(
            SYSTEM_PROMPT,
            f"Literature question: {question}",
            max_tokens=400,
            thinking=False,
            cheap=True,
        )
        out: list[str] = []
        for t in (result.get("titles") or []):
            if not isinstance(t, str):
                continue
            t = t.strip().strip('"')[:120]
            if 3 <= len(t.split()) <= 16:
                out.append(t)
        return out[:k]
    except Exception:  # noqa: BLE001 — degrade to no queries, never fail
        return []
