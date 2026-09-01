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
            # thinking MUST stay off for listwise: the hybrid reasoning model
            # burns the whole 4096-token budget on reasoning for a 50-candidate
            # listing and returns an EMPTY body (measured: 30s + empty), and
            # the empty-parse fallback silently discards the fine stage.
            # thinking=False measured 2.3s with a full order — and ranklab
            # showed BETTER recall (26.7% vs 20% on identical pools): the
            # bench's fine stage had been a silent no-op the whole time.
            thinking=False,
        )
        order = (_json.loads(raw) or {}).get("order") or []
        idxs = [i for i in order if isinstance(i, int) and 0 <= i < len(candidates)]
        idxs += [i for i in range(len(candidates)) if i not in set(idxs)]
        return [candidates[i] for i in idxs] + papers[top:]
    except Exception:  # noqa: BLE001 — model hiccup: keep deterministic order
        return list(papers)


_POINTWISE_SYS = """You score retrieval candidates for a literature-search \
question. The question usually describes ONE specific paper (or a very \
narrow set): some study that performs/answers exactly what the question \
asks. Reply ONLY JSON: {"scores": [int, ...]} — one score 0-10 per \
candidate, same order as listed. 10 = this candidate plausibly IS that \
paper (same task/system/finding); 7-9 = strongly likely; 4-6 = topically \
related but not the specific paper; 0-3 = unrelated. Judge each candidate \
independently; abstracts are truncated."""


def _pointwise_scores(
    question: str, papers: list[Paper], batch: int = 25
) -> list[int] | None:
    """Cheap per-candidate relevance scores over the WHOLE pool (coarse
    stage). Returns None on any failure — the caller then skips coarse
    ranking rather than ranking on garbage. Batches run concurrently:
    the coarse stage is latency-bound, not token-bound. batch=25 not 50:
    ranklab measured the gold drowning in a 50-batch (score 2 while it
    scored 8 alone) and b25 converting the lost gold at half the @20 cost
    (recall 26.7/33.3 vs 26.7/26.7) — finer batches, tighter attention."""
    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    from citens import llm

    def _score(chunk: list[Paper]) -> list[int] | None:
        listing = "\n".join(
            f"{i}. {p.title} ({p.year or '?'}) — "
            f"{(p.abstract or '').replace(chr(10), ' ')[:140]}"
            for i, p in enumerate(chunk)
        )
        try:
            raw = llm.chat(
                _POINTWISE_SYS,
                f"Question: {question}\n\nCandidates:\n{listing}",
                cheap=True,
                temperature=0.0,
                response_json=True,
                # thinking off: mechanical scoring measured 1.0s/call vs 20.0s
                # with identical output (ranklab 2026-08-30) — reasoning is
                # pure latency here and risks eating the body budget
                thinking=False,
            )
            vals = (_json.loads(raw) or {}).get("scores") or []
            if not isinstance(vals, list) or len(vals) != len(chunk):
                return None
            return [v if isinstance(v, int) else 0 for v in vals]
        except Exception:  # noqa: BLE001 — coarse failure = skip, not fail
            return None

    chunks = [papers[s:s + batch] for s in range(0, len(papers), batch)]
    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as ex:
        results = list(ex.map(_score, chunks))
    scores: list[int] = []
    for r in results:
        if r is None:
            return None
        scores.extend(r)
    return scores


_PAIRWISE_SYS = (
    "You compare two candidate papers for a literature-search question. "
    "Reply ONLY JSON: {\"winner\": \"A\"} or {\"winner\": \"B\"} — the paper "
    "the question is more likely asking about. Judge by how specifically "
    "the title/abstract matches the question's described contribution."
)


def _pairwise_pick(question: str, a: Paper, b: Paper) -> str | None:
    """A/B judgment between two candidates. None = failure/no preference."""
    import json as _json

    from citens import llm

    fmt = lambda p: (  # noqa: E731
        f"{p.title} ({p.year or '?'}) — "
        f"{(p.abstract or '').replace(chr(10), ' ')[:160]}"
    )
    try:
        raw = llm.chat(
            _PAIRWISE_SYS,
            f"Question: {question}\n\nA: {fmt(a)}\n\nB: {fmt(b)}",
            cheap=True,
            temperature=0.0,
            response_json=True,
            thinking=False,
        )
        w = (_json.loads(raw) or {}).get("winner")
        return w if w in ("A", "B") else None
    except Exception:  # noqa: BLE001 — a failed pair keeps the current order
        return None


def _pairwise_refine(
    question: str, papers: list[Paper], scores: list[int], band: int = 8
) -> list[Paper]:
    """One adjacent-comparison pass over the coarse head's TIED pairs.

    Measured NO-GAIN on the seed-42 pools (2026-09-01 A/B: 26.7/26.7 with
    and without; the pass DID fire — 5 tied pairs on q0 — but the gold sits
    at score 8 under a 9/10-score cluster, and tie-only swaps cannot
    promote across score levels). Kept behind ``cascade_rank(pairwise=)``
    because it is order-conservative and cheap; the plateau it targets
    (q0: ~5 plausible answers tied, question itself ambiguous) is a label
    ambiguity, not a resolution problem. Degrades to input order.
    """
    head_p = list(papers[:band])
    head_s = list(scores[:band])
    for i in range(len(head_p) - 1):
        if head_s[i] != head_s[i + 1]:
            continue  # coarse scores already separate this pair
        w = _pairwise_pick(question, head_p[i], head_p[i + 1])
        if w == "B":
            head_p[i], head_p[i + 1] = head_p[i + 1], head_p[i]
            head_s[i], head_s[i + 1] = head_s[i + 1], head_s[i]
    return head_p + list(papers[band:])


def cascade_rank(
    question: str,
    papers: list[Paper],
    coarse_keep: int = 50,
    strong: bool = False,
    pairwise: bool = False,
) -> list[Paper]:
    """Two-stage rank: pointwise coarse over the whole pool, listwise fine.

    Why a cascade (bench-measured): the cheap listwise over 100+ candidates
    is high-variance — it promotes some golds to #1 and demotes others below
    rank 20 in the same run; over ~30 it was stable and doubled recall@5.
    The coarse pointwise pass sees EVERY candidate (no truncation loss: a
    gold at pool position 52 still gets scored), keeps the top
    ``coarse_keep``; the listwise fine stage then orders a set small enough
    to be reliable. Any stage failure degrades to the remaining stage, and
    ultimately to the input order — never drops papers.

    ``pairwise``: A/B re-judgement of adjacent TIED pairs in the coarse
    head before the fine stage — measured no-gain (see _pairwise_refine),
    off by default.
    """
    scores: list[int] | None = None
    if len(papers) <= coarse_keep:
        coarse_order = list(papers)  # pool already fits the fine stage
    else:
        scores = _pointwise_scores(question, papers)
        if scores is None:
            coarse_order = list(papers)
        else:
            idx = sorted(range(len(papers)), key=lambda i: (-scores[i], i))
            coarse_order = [papers[i] for i in idx]
            scores = [scores[i] for i in idx]
    if pairwise and scores is not None and len(coarse_order) > 1:
        coarse_order = _pairwise_refine(question, coarse_order, scores)
    fine = listwise_rank(
        question, coarse_order[:coarse_keep], top=coarse_keep, strong=strong
    )
    return fine + coarse_order[coarse_keep:]
