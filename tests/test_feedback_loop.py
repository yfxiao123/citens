"""Offline tests for the feedback-loop agents (audit / verifier-trigger /
clarify). LLM calls are mocked — no network, no API keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import citens.agents.audit as audit_mod  # noqa: E402
import citens.agents.clarify as clarify_mod  # noqa: E402
import citens.agents.verifier_trigger as vt_mod  # noqa: E402
from citens.models import Claim, Verdict, VerificationResult  # noqa: E402

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


def test_clarify_questions_follow_review_language(monkeypatch):
    """The pre-run form must speak the review language (zh default)."""
    captured = {}

    class FakeResult(dict):
        pass

    def fake_chat_json(system, user, **kw):
        captured["system"] = system
        return {"questions": []}

    monkeypatch.setattr(clarify_mod, "chat_json", fake_chat_json)
    from citens.config import settings

    monkeypatch.setattr(settings, "review_language", "zh")
    clarify_mod.generate_clarifying_questions("订单簿建模")
    assert "简体中文" in captured["system"]

    monkeypatch.setattr(settings, "review_language", "en")
    clarify_mod.generate_clarifying_questions("order book")
    assert "English" in captured["system"] and "简体中文" not in captured["system"]


# --- provenance-driven reflect feedback -------------------------------------


def test_low_yield_directions_thresholds():
    from citens.orchestration.support import low_yield_directions

    rows = [
        {"concept": "good direction", "hits": 10, "kept": 6},
        {"concept": "junk direction", "hits": 12, "kept": 0},   # over-fetching
        {"concept": "thin direction", "hits": 2, "kept": 0},    # too small to judge
        {"concept": "weak direction", "hits": 5, "kept": 1},    # some survivors
    ]
    out = low_yield_directions(rows)
    assert [r["concept"] for r in out] == ["junk direction"]


def test_low_yield_synonym_swaps_untried_only():
    from citens.agents.planner import QueryPlan, low_yield_synonym_swaps

    plan = QueryPlan(
        queries=["market microstructure"],
        concepts=[{"term": "market microstructure", "synonyms": ["market micro-strategy", "dealer markets"]}],
    )
    out = low_yield_synonym_swaps(
        plan,
        [{"concept": "market microstructure", "hits": 9, "kept": 0}],
        already_searched=["market microstructure", "dealer markets"],
    )
    assert out == ["market micro-strategy"]
    # nothing left to swap -> empty
    assert low_yield_synonym_swaps(
        plan,
        [{"concept": "market microstructure", "hits": 9, "kept": 0}],
        ["market microstructure", "dealer markets", "market micro-strategy"],
    ) == []


def test_reflect_receives_yield_note(monkeypatch):
    import citens.agents.reflector as reflector_mod
    from citens.models import SynthesisResult

    captured = {}

    def fake_chat_json(system, user, **k):
        captured["system"] = system
        captured["user"] = user
        return {"needs_supplement": True, "rationale": "r",
                "supplementary_keywords": ["replacement query"]}

    monkeypatch.setattr(reflector_mod, "chat_json", fake_chat_json)
    out = reflector_mod.reflect(
        SynthesisResult(gaps=["gap one"]), "topic", 8,
        yield_note="junk direction 12->0",
    )
    assert out["supplementary_keywords"] == ["replacement query"]
    assert "junk direction 12->0" in captured["user"]
    assert "QUERY YIELD" in captured["system"]
    # without a yield note the block is absent (backward-compatible prompt)
    reflector_mod.reflect(SynthesisResult(gaps=[]), "topic", 8)
    assert "Query yield" not in captured["user"]
