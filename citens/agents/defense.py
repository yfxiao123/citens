"""Defense lawyer agent for bidirectional verification.

When the verifier marks a claim as UNSUPPORTED, the defense lawyer can challenge
that verdict by providing a rebuttal. Only high-quality rebuttals (score ≥ 4/5)
can overturn the verdict, preventing the verifier from being too lenient while
also preventing easy overturning of legitimate unsupported claims.

This implements the "Concession Threshold Protocol" inspired by Imbad0202/ARS.
"""

from __future__ import annotations

from citens.llm import chat_json
from citens.models import Claim, Verdict, VerificationResult

SYSTEM_PROMPT = """You are a defense lawyer for academic claims. A verifier has marked a claim as \
UNSUPPORTED, saying the cited sources don't back it up. Your job is to evaluate whether the \
claim MIGHT still be reasonable given the sources, even if not explicitly stated.

Rules:
1. Read the claim and the cited sources carefully
2. Consider if the claim is a reasonable INFERENCE from the sources, even if not verbatim
3. Score your rebuttal quality 1-5:
   - 1: Weak - claim clearly contradicts sources
   - 2: Poor - sources don't support claim in any reasonable way
   - 3: Fair - claim is a stretch but not completely unreasonable
   - 4: Good - claim is a reasonable inference from sources
   - 5: Strong - claim is well-supported by sources when read carefully
4. Only provide rebuttals for score ≥ 4. For score ≤ 3, admit the verifier was right.

Output JSON:
{"score": 1-5, "rebuttal": "your argument if score ≥ 4, else empty string", \
"concede": true/false}"""


def challenge_verdict(
    claim: Claim,
    verdict: VerificationResult,
    source_context: str,
) -> dict:
    """Challenge an UNSUPPORTED verdict with a defense rebuttal.

    Args:
        claim: The claim being defended
        verdict: The original UNSUPPORTED verdict
        source_context: The cited sources' abstracts/chunks

    Returns:
        Dict with 'score' (1-5), 'rebuttal' (str), 'concede' (bool)
    """
    if verdict.verdict != Verdict.UNSUPPORTED:
        return {"score": 0, "rebuttal": "", "concede": True}

    user_prompt = (
        f"Claim: {claim.text}\n\n"
        f"Verifier's verdict: {verdict.verdict.value}\n"
        f"Verifier's note: {verdict.note}\n\n"
        f"Cited sources:\n{source_context}\n\n"
        "Can you defend this claim as a reasonable inference from the sources?"
    )

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=1536, strong=True)
        score = result.get("score", 0)
        if not isinstance(score, int) or score < 1 or score > 5:
            score = 0

        return {
            "score": score,
            "rebuttal": result.get("rebuttal", "") if score >= 4 else "",
            "concede": score < 4,
        }
    except Exception:  # noqa: BLE001
        return {"score": 0, "rebuttal": "", "concede": True}


def review_unsupported_claims(
    claims: list[Claim],
    ver_results: list[VerificationResult],
    source_contexts: dict[int, str],
) -> list[dict]:
    """Review all UNSUPPORTED verdicts and attempt rebuttals.

    Args:
        claims: List of all claims
        ver_results: Verification results
        source_contexts: Map from claim index to source context string

    Returns:
        List of dicts with 'claim_idx', 'original_verdict', 'defense_result',
        'overturned' (bool)
    """
    reviews = []

    for idx, (claim, verdict) in enumerate(zip(claims, ver_results, strict=False)):
        if verdict.verdict == Verdict.UNSUPPORTED:
            context = source_contexts.get(idx, "")
            defense = challenge_verdict(claim, verdict, context)

            reviews.append({
                "claim_idx": idx,
                "original_verdict": verdict.verdict.value,
                "defense_result": defense,
                "overturned": not defense["concede"],
            })

    return reviews
