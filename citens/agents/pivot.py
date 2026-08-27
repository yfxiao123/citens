"""Pseudo-relevance-feedback vocabulary pivot.

Bench-measured failure mode this addresses: for a find-that-paper question
whose words do not overlap the target's title/abstract (the vocabulary wall),
every lexical channel fails together — planned queries (6.7-20% on LitSearch),
one-hop snowball from the nearest neighbors (reach ~0 over 10 observed
misses), and even S2's learned recommendations from those neighbors (also 0):
the neighbors retrieval CAN see already sit in an adjacent-but-wrong
subfield, so neither citation edges nor similarity models reach the target's
micro-cluster from there.

But the neighbors' own TEXT knows the way: related-work abstracts name the
task, method, and system vocabulary of the cluster the target lives in
("emotional support conversation", "COMET"). The pivot reads the abstracts of
the papers already retrieved and coins search queries in THAT vocabulary —
the translation step query reformulation cannot do from the question alone.
"""

from __future__ import annotations

from citens.llm import chat_json
from citens.models import Paper

SYSTEM_PROMPT = """You mine search queries out of retrieved abstracts.

A literature question's own wording FAILS to retrieve its target papers — the \
field does not use those words. You are given the question plus the abstracts \
of the nearest papers already retrieved. Your job: identify the specific \
SUBFIELD vocabulary those abstracts use (task names, method names, system or \
benchmark names) that papers ANSWERING the question would share, and write \
search queries in that vocabulary.

Rules:
1. Copy named entities (task, benchmark, dataset, method names) VERBATIM as \
they appear in the abstracts — "SemEval-2022 Task 4", not "shared task 4". \
A name an abstract mentions is often the exact phrase the target's title \
contains; paraphrasing it away loses the retrieval (measured: "emotional \
support conversation" works, "emotional support dialogue systems" does not).
2. Otherwise 2-6 plain keywords per query, no boolean operators.
3. Do NOT copy the question's phrasing — the whole point is different words.
4. Prefer combinations of two specific terms over single broad terms.
5. Reply ONLY JSON: {"queries": ["...", ...]} with 3-5 queries."""


def pivot_from_abstracts(question: str, papers: list[Paper], k: int = 4) -> list[str]:
    """Queries coined from the neighbors' subfield vocabulary (PRF pivot).

    ``papers`` should be the most question-relevant papers already retrieved
    (fused/ranked head). Returns [] on any LLM failure — a pivot that cannot
    run must degrade to "no extra queries", never fail the caller.
    """
    neighbors = [
        p for p in papers if (p.abstract or "").strip() or p.title.strip()
    ][:6]
    if not neighbors:
        return []
    listing = "\n\n".join(
        f"Paper {i + 1}: {p.title}\n{(p.abstract or '')[:400]}"
        for i, p in enumerate(neighbors)
    )
    user_prompt = f"Question: {question}\n\nRetrieved nearest papers:\n{listing}"
    try:
        result = chat_json(
            SYSTEM_PROMPT, user_prompt, max_tokens=512, thinking=False, cheap=True
        )
        return [
            str(q).strip()[:90]
            for q in (result.get("queries") or [])
            if isinstance(q, str) and 2 <= len(q.strip().split()) <= 8
        ][:k]
    except Exception:  # noqa: BLE001 — pivot degrades to nothing, never fails
        return []
