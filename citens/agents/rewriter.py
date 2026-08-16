"""Claim-rewriting agent: turn unsupported claims into grounded ones.

The verifier flags claims the cited sources do not back; the defense lawyer
rescues verifier misjudgments. What remains genuinely unsupported is not a
reporting problem to annotate — it is a writing defect to FIX. This agent
rewrites each surviving unsupported claim so it IS supported by its sources:

- weaken/qualify the overreach (drop specifics the source never states),
- or drop citations that contribute nothing to the claim,
- never introduce new facts (a prettier fabrication is still a fabrication).

Rewrites are then re-verified by the normal verifier, closing the loop:
precision stops being a report and becomes a pipeline behavior.
"""

from __future__ import annotations

from citens.grounding import ChunkStore, CitationTable
from citens.llm import chat_json
from citens.models import Claim, VerificationResult

SYSTEM_PROMPT = """You are a meticulous academic editor fixing citation problems in a survey.

You are given claims that a verifier judged UNSUPPORTED against the cited \
papers' ground text, along with that ground text. Rewrite each claim so that \
it IS supported by the sources it cites.

Rewrite rules (strict):
1. WEAKEN or QUALIFY — remove the specifics the source does not state \
(magnitudes, mechanisms, generalizations). "X improves Y by 30%" -> "X is \
reported to improve Y" when only the direction is grounded.
2. DROP dead citations — remove [n] markers of papers that do not support \
any part of the claim. Keep only citations that plausibly back the rewrite.
3. NEVER add new facts, numbers, or mechanisms not present in the ground \
text. A shorter, hedged claim is correct; an embellished one is not.
4. Keep the claim's role in the survey (it still says something useful).
5. Keep the same citation-marker format: [n] or [n][m].

Output JSON:
{"rewrites": [
  {"claim_index": 0, "new_text": "...", "note": "removed unsupported magnitude"}
]}"""


def rewrite_unsupported_claims(
    claims: list[Claim],
    ver_results: list[VerificationResult],
    table: CitationTable,
    chunk_store: ChunkStore,
    *,
    batch_size: int = 8,
) -> dict[int, dict]:
    """Rewrite unsupported claims to match their sources.

    Returns {claim_index: {"new_text": ..., "note": ...}} for accepted
    rewrites. Claims whose rewrite keeps zero citations are dropped (nothing
    in the rewrite is grounded, so the claim cannot be saved).
    """
    targets = [
        i for i, r in enumerate(ver_results)
        if r.verdict.value == "unsupported" and i < len(claims)
    ]
    if not targets:
        return {}

    rewrites: dict[int, dict] = {}
    for start in range(0, len(targets), batch_size):
        batch_idx = targets[start : start + batch_size]
        context_lines = []
        claim_lines = []
        for j, i in enumerate(batch_idx):
            claim = claims[i]
            claim_lines.append(f"Claim {j} (original): {claim.text}")
            for cite in claim.citation_indices:
                pid = table.paper_id(cite)
                chunks = chunk_store.chunks_for(pid)[:3]
                if chunks:
                    body = "\n".join(c.text for c in chunks)
                    context_lines.append(f"[{cite}] {table.label(cite)}\n{body}\n")
        user_prompt = (
            "Ground text of cited papers:\n"
            + "\n".join(context_lines[:24])
            + "\n\nClaims judged unsupported:\n"
            + "\n".join(claim_lines)
            + f"\n\nRewrite all {len(batch_idx)} claims per the rules."
        )
        try:
            result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=4096, strong=True)
        except Exception as e:  # noqa: BLE001
            print(f"    rewrite batch failed: {e}")
            continue
        for entry in result.get("rewrites", []):
            if not isinstance(entry, dict) or "claim_index" not in entry:
                continue
            try:
                j = int(entry["claim_index"])
            except (TypeError, ValueError):
                continue
            if not 0 <= j < len(batch_idx):
                continue
            new_text = str(entry.get("new_text", "")).strip()
            # a rewrite with no citation markers saves nothing: it cannot be
            # re-verified and would leave an ungrounded sentence in the survey
            if not new_text or "[" not in new_text:
                continue
            rewrites[batch_idx[j]] = {
                "new_text": new_text,
                "note": str(entry.get("note", ""))[:200],
            }
    return rewrites
