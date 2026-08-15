"""Offline tests for the feedback-loop agents (audit / verifier-trigger /
clarify). LLM calls are mocked — no network, no API keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import litreview.agents.audit as audit_mod  # noqa: E402
import litreview.agents.clarify as clarify_mod  # noqa: E402
import litreview.agents.verifier_trigger as vt_mod  # noqa: E402
from litreview.models import Claim, Verdict, VerificationResult  # noqa: E402

# --- absence audit ---------------------------------------------------------


def test_missing_to_queries():
    audit = {
        "absent_canonical_papers": [
            {"title": "A Stochastic Model for Order Book Dynamics", "authors": "Cont", "year": 2010},
            {"title": "High-frequency trading in a limit order book", "authors": "Unknown", "year": 2008},
        ],
        "missing_venue_areas": ["queueing theory"],
    }
    qs = audit_mod.missing_to_queries(audit)
    assert qs[0] == "A Stochastic Model for Order Book Dynamics Cont"
    assert "High-frequency trading" in qs[1]
    assert "queueing theory" in qs[-1]
    assert len(qs) <= 6


def test_missing_to_queries_empty():
    assert audit_mod.missing_to_queries({}) == []
    assert audit_mod.missing_to_queries({"absent_canonical_papers": []}) == []


def test_audit_parses_llm_output(monkeypatch):
    """The agent must keep only dicts with non-empty titles from LLM JSON."""
    monkeypatch.setattr(
        audit_mod,
        "chat_json",
        lambda *a, **k: {
            "absent_canonical_papers": [
                {"title": "  ", "authors": "X"},
                {"title": "Real Paper", "authors": "Y", "year": 2020},
                "not-a-dict",
            ],
            "missing_venue_areas": ["z"],
            "audit_note": "ok",
        },
    )
    out = audit_mod.audit_coverage("topic", ["p1"])
    assert len(out["absent_canonical_papers"]) == 1
    assert out["absent_canonical_papers"][0]["title"] == "Real Paper"
    assert out["missing_venue_areas"] == ["z"]


# --- verifier-triggered supplement -----------------------------------------


def test_collect_unsupported_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(
        vt_mod, "chat", lambda *a, **k: '{"queries": ["q1", "q2", "q1"], "reasoning": "x"}'
    )
    claims = [
        Claim(text="claim A", citation_indices=[0]),
        Claim(text="claim B", citation_indices=[1]),
    ]
    results = [
        VerificationResult(claim_text="claim A", verdict=Verdict.UNSUPPORTED, citation_indices=[0]),
        VerificationResult(claim_text="claim B", verdict=Verdict.SUPPORTED, citation_indices=[1]),
    ]
    qs = vt_mod.collect_unsupported_queries(claims, results, "topic")
    assert qs == ["q1", "q2"]  # deduped
    assert len(qs) <= 6


def test_collect_unsupported_none():
    claims = [Claim(text="c", citation_indices=[0])]
    results = [VerificationResult(claim_text="c", verdict=Verdict.SUPPORTED, citation_indices=[0])]
    assert vt_mod.collect_unsupported_queries(claims, results, "topic") == []


def test_derive_queries_handles_bad_json(monkeypatch):
    monkeypatch.setattr(vt_mod, "chat", lambda *a, **k: "not json")
    qs = vt_mod._derive_queries(Claim(text="c", citation_indices=[0]), "t")
    assert qs == []


# --- clarify ---------------------------------------------------------------


def test_generate_questions_filters_malformed(monkeypatch):
    monkeypatch.setattr(
        clarify_mod,
        "chat_json",
        lambda *a, **k: {
            "questions": [
                {"id": "focus", "question": "Which focus?", "options": ["a", "b"], "default": "a"},
                {"id": "", "question": "no id", "options": ["x", "y"]},  # dropped
                {"id": "bad", "question": "only one option", "options": ["x"]},  # dropped
            ]
        },
    )
    out = clarify_mod.generate_clarifying_questions("topic")
    assert len(out) == 1
    assert out[0]["id"] == "focus"
    assert out[0]["default"] == "a"


def test_generate_questions_empty_on_error(monkeypatch):
    monkeypatch.setattr(clarify_mod, "chat_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    assert clarify_mod.generate_clarifying_questions("t") == []


if __name__ == "__main__":
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as td:
                import inspect

                if "monkeypatch" in inspect.signature(fn).parameters:
                    class _MP:
                        def setattr(self, obj, attr, value):
                            setattr(obj, attr, value)

                    fn(_MP())
                else:
                    fn()
            print(f"PASS {name}")
    print("all feedback-loop tests passed")
