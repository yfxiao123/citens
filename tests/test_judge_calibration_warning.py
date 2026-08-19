"""judge-model-vs-calibration mismatch must surface in the health report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens.agents.health import calibration_status, check_health  # noqa: E402
from citens.config import settings  # noqa: E402
from citens.models import SynthesisResult, Verdict, VerificationResult  # noqa: E402


def _synth():
    return SynthesisResult(consensus=["a"] * 2, contradictions=["b"], gaps=["c"])


def _results():
    return [VerificationResult(claim_text="x", verdict=Verdict.SUPPORTED)] * 3


def test_golden_records_judge_model():
    cal = calibration_status()
    assert cal.get("calibrated_model"), "golden set must record judge_model"
    assert cal.get("calibrated_thinking")


def test_mismatched_model_warns(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_model_strong", "some-other-model")
    monkeypatch.setattr(settings, "judge_thinking", "low")
    report = check_health(_synth(), _results(), {"absent_canonical_papers": ["x"]}, {})
    assert "judge_model_uncalibrated" in report["issues"]
    assert "unanchored" in report["recommendation"]


def test_mismatched_thinking_warns(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_model_strong", "")
    monkeypatch.setattr(settings, "judge_thinking", "true")
    report = check_health(_synth(), _results(), {"absent_canonical_papers": ["x"]}, {})
    assert "judge_model_uncalibrated" in report["issues"]


def test_matching_model_no_warning(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "llm_model_strong", "")
    monkeypatch.setattr(settings, "judge_thinking", "low")
    report = check_health(_synth(), _results(), {"absent_canonical_papers": ["x"]}, {})
    assert "judge_model_uncalibrated" not in report["issues"]
