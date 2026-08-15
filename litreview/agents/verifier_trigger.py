"""Verification-triggered supplementary retrieval.

When a claim is judged UNSUPPORTED, the conventional response is to rewrite or
drop it. But an unsupported claim often signals a deeper problem: the evidence
for it was never retrieved. This module derives targeted search queries from
the unsupported claims and feeds them back into retrieval — the review closes
the loop between "what we wrote" and "what the literature actually supports".
"""

from __future__ import annotations

from litreview.llm import chat
from litreview.models import Claim, Verdict, VerificationResult

_TRIGGER_PROMPT = """You are a literature-retrieval analyst. A survey of the topic below made a \
claim, but the claim's cited papers do not support it (the claim is UNSUPPORTED — it may be \
about a finding, method, or term that the cited sources never mention).

Your job: derive 2-4 targeted ENGLISH search queries that would retrieve the literature \
actually supporting this claim — the specific method, phenomenon, or finding it refers to.

Rules:
1. Queries must be concise English phrases (3-6 words), suitable for academic search engines.
2. Only derive queries for what the claim itself asserts — do NOT guess at the underlying \
literature's identity.
3. If the claim is too vague or generic to derive useful queries, return an empty list.

Output JSON:
{"queries": ["english query 1", "english query 2", ...], "reasoning": "..."}"""


def _derive_queries(claim: Claim, topic: str) -> list[str]:
    """LLM derives targeted queries for one unsupported claim (empty on failure)."""
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"论断 / Claim: {claim.text}\n\n"
        f"被引论文编号 / Cited: {claim.citation_indices}\n\n"
        "Derive targeted English search queries to retrieve literature supporting this claim."
    )
    try:
        raw = chat(_TRIGGER_PROMPT, user_prompt, max_tokens=1024, response_json=True, strong=True)
        import json as _json

        data = _json.loads(raw)
        return [
            q for q in data.get("queries", []) if isinstance(q, str) and q.strip()
        ][:4]
    except Exception:  # noqa: BLE001
        return []


def collect_unsupported_queries(
    claims: list[Claim],
    ver_results: list[VerificationResult],
    topic: str,
    *,
    max_claims: int = 5,
) -> list[str]:
    """Map unsupported claims to supplementary search queries.

    Verdicts carry claim_text; we zip them back to the source claims by text
    match (verifier results are ordered like the claims list). Derives queries
    for up to `max_claims` unsupported claims, dedupes, caps at 6.
    """
    unsupported_texts = {
        r.claim_text for r in ver_results if r.verdict == Verdict.UNSUPPORTED
    }
    if not unsupported_texts:
        return []
    by_text = {c.text: c for c in claims}
    targets = [by_text[t] for t in unsupported_texts if t in by_text][:max_claims]
    queries: list[str] = []
    for claim in targets:
        queries.extend(_derive_queries(claim, topic))
        if len(queries) >= 6:
            break
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= 6:
            break
    return out
