"""Dialogue health monitoring agent.

Detects systematic biases in the review pipeline:
- Premature convergence: all themes agree, no contradictions identified
- Verifier too lenient: precision > 95% (likely missing unsupported claims)
- Topic consensus bias: all papers in a theme support the same viewpoint
- Absence blindness: no canonical works detected as missing (unusual)

When issues are detected, auto-injects adversarial queries or tightens verification.
Inspired by Imbad0202/ARS's "Dialogue Health Indicator".
"""

from __future__ import annotations

from litreview.llm import chat_json
from litreview.models import SynthesisResult, Verdict, VerificationResult

SYSTEM_PROMPT = """You are a methodology critic reviewing a literature review pipeline for systematic \
biases. Given the synthesis results and verification statistics, identify potential issues:

1. "premature_convergence": All themes show consensus, few/no contradictions. May indicate \
the search was too narrow or the verifier missed problems.

2. "verifier_too_lenient": Citation precision > 95%. May indicate the verifier is not \
catching unsupported claims.

3. "topic_consensus_bias": Within themes, all papers support the same viewpoint with no \
dissenting voices. May indicate selection bias.

4. "absence_blindness": Absence audit found 0 missing canonical works. Unusual for most \
topics - may indicate the audit agent is broken or the search was unusually complete.

Output JSON:
{"issues": ["issue1", "issue2", ...], "adversarial_queries": ["query to find counter-evidence"], \
"recommendation": "what to do next"}"""


def check_health(
    synthesis: SynthesisResult,
    ver_results: list[VerificationResult],
    absence_audit: dict,
    theme_paper_counts: dict[str, int],
) -> dict:
    """Check pipeline health and detect systematic biases.

    Args:
        synthesis: The synthesis result (consensus/contradictions/gaps)
        ver_results: All verification results
        absence_audit: Output from audit_coverage()
        theme_paper_counts: Map from theme name to number of papers

    Returns:
        Dict with 'issues' (list), 'adversarial_queries' (list),
        'recommendation' (str), 'metrics' (dict)
    """
    # Compute metrics
    verifiable = [r for r in ver_results if r.verdict != Verdict.UNVERIFIABLE]
    supported = sum(1 for r in verifiable if r.verdict == Verdict.SUPPORTED)
    partial = sum(1 for r in verifiable if r.verdict == Verdict.PARTIAL)
    unsupported = sum(1 for r in verifiable if r.verdict == Verdict.UNSUPPORTED)

    precision = (supported + partial) / len(verifiable) if verifiable else 0.0

    n_consensus = len(synthesis.consensus)
    n_contradictions = len(synthesis.contradictions)
    n_gaps = len(synthesis.gaps)

    absent_count = len(absence_audit.get("absent_canonical_papers", []))

    metrics = {
        "precision": precision,
        "n_consensus": n_consensus,
        "n_contradictions": n_contradictions,
        "n_gaps": n_gaps,
        "n_unsupported": unsupported,
        "n_absent": absent_count,
    }

    # Detect issues
    issues = []
    if n_consensus > 5 and n_contradictions == 0:
        issues.append("premature_convergence")
    if precision > 0.95 and len(verifiable) > 20:
        issues.append("verifier_too_lenient")
    if absent_count == 0 and len(verifiable) > 10:
        issues.append("absence_blindness")

    # Generate adversarial queries if issues detected
    adversarial_queries = []
    if issues:
        user_prompt = (
            f"Metrics:\n"
            f"  Precision: {precision:.2f}\n"
            f"  Consensus: {n_consensus}, Contradictions: {n_contradictions}, Gaps: {n_gaps}\n"
            f"  Unsupported claims: {unsupported}\n"
            f"  Absent canonical works: {absent_count}\n\n"
            f"Issues detected: {', '.join(issues)}\n\n"
            "Generate 2-3 adversarial queries to find counter-evidence or missing perspectives."
        )

        try:
            result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=1536, strong=True)
            adversarial_queries = result.get("adversarial_queries", [])[:3]
        except Exception:  # noqa: BLE001
            adversarial_queries = []

    recommendation = ""
    if "premature_convergence" in issues:
        recommendation = "Search for papers that contradict the consensus findings."
    elif "verifier_too_lenient" in issues:
        recommendation = "Tighten verification criteria or manually review high-confidence claims."
    elif "absence_blindness" in issues:
        recommendation = "Re-run absence audit with broader canonical work list."
    else:
        recommendation = "Pipeline appears healthy. No systematic biases detected."

    return {
        "issues": issues,
        "adversarial_queries": adversarial_queries,
        "recommendation": recommendation,
        "metrics": metrics,
    }
