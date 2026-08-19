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

import json
from pathlib import Path

from citens.config import settings
from citens.llm import chat_json, strong_model
from citens.models import SynthesisResult, Verdict, VerificationResult

# The human-audited calibration set that binds the judge's self-reported
# precision to an audited grounded rate. Calibration is model-specific: swap
# LLM_MODEL_STRONG and the golden numbers no longer apply until re-audited.
_GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "verifier_calibration_201038.json"


def calibration_status() -> dict:
    """Which judge model/thinking the golden calibration was measured on."""
    try:
        d = json.loads(_GOLDEN.read_text(encoding="utf-8"))
        return {
            "calibrated_model": d.get("judge_model", ""),
            "calibrated_thinking": d.get("judge_thinking", ""),
            "calibrated_at": d.get("audited_at", ""),
        }
    except (OSError, json.JSONDecodeError):
        return {}

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
    canary: dict | None = None,
) -> dict:
    """Check pipeline health and detect systematic biases.

    Args:
        synthesis: The synthesis result (consensus/contradictions/gaps)
        ver_results: All verification results
        absence_audit: Output from audit_coverage()
        theme_paper_counts: Map from theme name to number of papers
        canary: Output from canary_check() — the judge's false-accept rate on
            synthetic unsupported claims (None when not measured)

    Returns:
        Dict with 'issues' (list), 'adversarial_queries' (list),
        'recommendation' (str), 'metrics' (dict)
    """
    # Compute metrics
    judge_model = strong_model()
    verifiable = [r for r in ver_results if r.verdict != Verdict.UNVERIFIABLE]
    supported = sum(1 for r in verifiable if r.verdict == Verdict.SUPPORTED)
    partial = sum(1 for r in verifiable if r.verdict == Verdict.PARTIAL)
    background = sum(1 for r in verifiable if r.verdict == Verdict.BACKGROUND)
    contradictory = sum(1 for r in verifiable if r.verdict == Verdict.CONTRADICTORY)
    unsupported = sum(1 for r in verifiable if r.verdict == Verdict.UNSUPPORTED)
    unverifiable = sum(1 for r in ver_results if r.verdict == Verdict.UNVERIFIABLE)

    precision = (supported + partial) / len(verifiable) if verifiable else 0.0
    unverifiable_rate = unverifiable / len(ver_results) if ver_results else 0.0
    canary_far = (canary or {}).get("false_accept_rate")

    n_consensus = len(synthesis.consensus)
    n_contradictions = len(synthesis.contradictions)
    n_gaps = len(synthesis.gaps)

    absent_count = len(absence_audit.get("absent_canonical_papers", []))

    metrics = {
        "precision": precision,
        "unverifiable_rate": round(unverifiable_rate, 3),
        "n_consensus": n_consensus,
        "n_contradictions": n_contradictions,
        "n_gaps": n_gaps,
        "n_unsupported": unsupported,
        "n_background": background,
        "n_contradictory": contradictory,
        "n_absent": absent_count,
    }
    if canary_far is not None:
        metrics["canary_false_accept_rate"] = canary_far

    # Detect issues
    issues = []
    if n_consensus > 5 and n_contradictions == 0:
        issues.append("premature_convergence")
    if precision > 0.95 and len(verifiable) > 20:
        issues.append("verifier_too_lenient")
    if absent_count == 0 and len(verifiable) > 10:
        issues.append("absence_blindness")
    if len(verifiable) >= 10 and background > 0.2 * len(verifiable):
        issues.append("reviews_cited_as_primary")
    # measured (not inferred) leniency: canaries are unsupported by
    # construction, so even one passing is a calibration failure
    if canary_far is not None and (canary or {}).get("injected", 0) >= 2 and canary_far > 0.34:
        issues.append("verifier_false_accept")

    # calibration is model-specific: a different judge model (or thinking
    # level) than the golden set's means the reported precision is no longer
    # anchored to the audited grounded rate
    cal = calibration_status()
    calib_model = cal.get("calibrated_model", "")
    calib_thinking = str(cal.get("calibrated_thinking", ""))
    judge_model_uncalibrated = bool(
        calib_model and judge_model and judge_model != calib_model
    ) or bool(calib_thinking and str(settings.judge_thinking) != calib_thinking)
    if judge_model_uncalibrated:
        issues.append("judge_model_uncalibrated")
    metrics["judge_model"] = judge_model
    if cal:
        metrics["calibrated_model"] = calib_model or None

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
            result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=1536, strong=True,
                              thinking=settings.judge_thinking)
            adversarial_queries = result.get("adversarial_queries", [])[:3]
        except Exception:  # noqa: BLE001
            adversarial_queries = []

    recommendation = ""
    if "premature_convergence" in issues:
        recommendation = "Search for papers that contradict the consensus findings."
    elif "verifier_false_accept" in issues:
        recommendation = (
            f"Judge passed {round((canary_far or 0) * 100)}% of synthetic unsupported "
            "claims — verifier calibration is broken, review verdicts manually."
        )
    elif "judge_model_uncalibrated" in issues:
        recommendation = (
            f"Judge model '{judge_model}' (thinking={settings.judge_thinking}) differs from "
            f"the human-calibrated '{calib_model}' (thinking={calib_thinking}, {cal.get('calibrated_at', '')}) "
            "— the reported precision is unanchored; re-run `citens audit` calibration "
            "before trusting the number."
        )
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
