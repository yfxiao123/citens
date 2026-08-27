"""LLM listwise relevance ranking — the semantic layer without embeddings.

No embedding endpoint exists on our API tier, and every retrieval source is
lexical: RRF fusion over query cells can only sum ranking signals, so the
paper the QUESTION means keeps losing to papers the WORDS match (measured:
pivot alone 13.3% but pivot fused into planned 6.7% — noise cells pushed the
gold below rank 20). LitSearch's own results show the fix shape: their
GPT-4o rerank adds +4.4 recall on top of a retriever whose candidate pool
already contains the gold, and our llm_rerank leg doubled recall@5 (6.7% ->
13.3%) exactly when the gold was in the union. The LLM reading titles
against the question is the only semantic judgment available — this module
is that judgment, shared by the bench (candidate-pool ranking) and the
harness (find-mode output ranking).
"""

from __future__ import annotations

from citens.models import Paper

_TOP_DEFAULT = 100  # ~4k listing tokens: single listwise call, no batching


def listwise_rank(
    question: str,
    papers: list[Paper],
    top: int = _TOP_DEFAULT,
    strong: bool = False,
) -> list[Paper]:
    """Reorder ``papers[:top]`` by LLM relevance to ``question``; rest follow.

    ``strong=True`` routes to the strong tier — worth it for the one call
    that orders a user-facing deliverable (LitSearch's rerank gains came
    from their strongest model; the cheap tier demoted measured golds when
    judged 100+ candidates). Falls back to the input order on any model
    failure — a ranking that cannot run must degrade to the deterministic
    order, never drop papers.
    """
    import json as _json

    from citens import llm

    candidates = papers[:top]
    if len(candidates) < 2:
        return list(papers)
    listing = "\n".join(
        f"{i}. {p.title} ({p.year or '?'}) — {(p.abstract or '')[:120]}"
        for i, p in enumerate(candidates)
    )
    sys_p = (
        "You rerank retrieval results for a literature-search question. "
        'Reply ONLY JSON: {"order": [int, ...]} — the candidate indices, '
        "best match for the question first. Rank ALL candidates."
    )
    user_p = f"Question: {question}\n\nCandidates:\n{listing}"
    try:
        raw = llm.chat(
            sys_p,
            user_p,
            cheap=not strong,
            strong=strong,
            temperature=0.0,
            response_json=True,
        )
        order = (_json.loads(raw) or {}).get("order") or []
        idxs = [i for i in order if isinstance(i, int) and 0 <= i < len(candidates)]
        idxs += [i for i in range(len(candidates)) if i not in set(idxs)]
        return [candidates[i] for i in idxs] + papers[top:]
    except Exception:  # noqa: BLE001 — model hiccup: keep deterministic order
        return list(papers)
