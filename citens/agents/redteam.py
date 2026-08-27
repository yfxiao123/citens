"""Red-team adversarial review of the final document.

The verifier checks each CITATION against its source; the defense step re-
tries each CLAIM. Neither can see the document as a whole. The red team
attacks the REVIEW: causal language resting on correlational evidence,
numbers stronger than their sources, one-sided evidence bases, statements
in different sections that cannot both hold, missing limitations. One
attack pass + one bounded revision pass (deep mode only) — findings that
cannot be fixed land in the Limitations section instead of being silently
dropped, per the project's honesty rule.
"""

from __future__ import annotations

from citens.config import settings
from citens.llm import chat_json

SYSTEM_PROMPT = """You are a hostile expert reviewer (red team) attacking a literature \
review before publication. Report every weakness a serious opponent would exploit. \
Focus on DOCUMENT-level failures that per-claim verification cannot see:

- OVERCLAIM: causal or mechanistic language resting on correlational / \
single-study / preprint-only evidence
- UNSUPPORTED_NUMBER: numbers or percentages stronger than, or absent from, \
the cited evidence
- CHERRY_PICKING: one-sided evidence base; contradicting studies silently omitted
- INTERNAL_CONTRADICTION: statements in different sections that cannot both hold
- MISSING_LIMITATION: no honest treatment of the evidence base's weaknesses \
(sample bias, field skew, paywall gaps)

Rules:
- Do NOT report typos, style, citation formatting, or anything the verification \
summary already flags.
- Quote the exact offending sentence fragment in "excerpt".
- Max 8 findings, most damaging first. If the review is genuinely solid, \
return an empty list — do not invent attacks.

Output JSON:
{"findings": [{"type": "OVERCLAIM", "severity": "high|medium|low", \
"excerpt": "...", "attack": "why this fails peer review", \
"fix": "concrete revision instruction"}]}"""

FIX_PROMPT = """You are revising a literature review after a red-team attack. Apply the \
fixes below with a minimal-touch policy:

1. OVERCLAIM / UNSUPPORTED_NUMBER: weaken the wording until the evidence \
supports it ("X causes Y" -> "X is associated with Y in [study]").
2. INTERNAL_CONTRADICTION: resolve toward the better-evidenced statement.
3. CHERRY_PICKING / MISSING_LIMITATION / unfixable findings: fold them into \
an honest "Limitations" section (create it if absent).

HARD RULES — violations make the revision worthless:
- Never add, remove, or renumber citations.
- Never introduce new numbers, findings, or claims.
- Never delete cited evidence; only qualify it.
- Keep every section heading except where a Limitations section is created.

Output JSON: {"review": "<the COMPLETE revised review markdown>"}"""


def red_team_review(review_md: str, context_note: str = "") -> list[dict]:
    """Attack the final review; returns findings sorted by severity."""
    ctx = f"\nVerification context: {context_note}\n" if context_note else ""
    try:
        result = chat_json(
            SYSTEM_PROMPT,
            f"Review to attack:\n\n{review_md[:60000]}{ctx}",
            max_tokens=3072,
            strong=True,
            thinking=settings.judge_thinking,
        )
    except Exception:  # noqa: BLE001 — an attack pass must never kill the run
        return []
    order = {"high": 0, "medium": 1, "low": 2}
    findings = [
        {
            "type": str(f.get("type", ""))[:30],
            "severity": str(f.get("severity", "medium")).lower()[:8],
            "excerpt": str(f.get("excerpt", ""))[:200],
            "attack": str(f.get("attack", ""))[:300],
            "fix": str(f.get("fix", ""))[:300],
        }
        for f in result.get("findings", [])
        # MISSING_LIMITATION has no excerpt to quote (it attacks an absence)
        if isinstance(f, dict) and (f.get("excerpt") or f.get("attack"))
    ][:8]
    return sorted(findings, key=lambda f: order.get(f["severity"], 3))


def apply_red_team_fixes(review_md: str, findings: list[dict]) -> str | None:
    """One bounded revision pass; None when nothing to do or the model balked."""
    if not findings:
        return None
    digest = "\n".join(
        f"- [{f['severity']}] {f['type']}: {f['attack']} → fix: {f['fix']}"
        + (f" (excerpt: \"{f['excerpt']}\")" if f["excerpt"] else "")
        for f in findings
    )
    try:
        # the revision must echo the FULL review back — a fixed cap truncates
        # long documents (acceptance run B: 23k-char review overflowed 8192
        # and the fragment guard rightly refused), so the budget scales with
        # the review's length
        budget = max(8192, min(32000, int(len(review_md) * 1.5)))
        revised = chat_json(
            FIX_PROMPT,
            f"Findings to apply:\n{digest}\n\nReview to revise:\n\n{review_md[:60000]}",
            max_tokens=budget,
            strong=True,
            thinking=settings.judge_thinking,
            temperature=0.2,
        )
    except Exception:  # noqa: BLE001
        return None
    # the fix prompt demands the complete markdown back; a fragment means the
    # model balked — returning it would silently truncate the review
    text = revised.get("review") if isinstance(revised, dict) else None
    if not text or not isinstance(text, str) or len(text) < 0.5 * len(review_md):
        return None
    return text
