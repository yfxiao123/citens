"""Model routing (cheap tier) + red-team adversarial review. LLM mocked."""

from __future__ import annotations

import citens.agents.redteam as redteam_mod
import citens.llm as llm_mod
from citens.agents.redteam import apply_red_team_fixes, red_team_review

# --- cheap-tier routing --------------------------------------------------------


def test_cheap_model_falls_back_to_default(monkeypatch):
    from citens.config import settings

    monkeypatch.setattr(settings, "llm_model_cheap", "", raising=False)
    assert llm_mod.cheap_model() == settings.llm_model
    monkeypatch.setattr(settings, "llm_model_cheap", "flash-tier", raising=False)
    assert llm_mod.cheap_model() == "flash-tier"


def test_chat_routes_cheap_and_strong(monkeypatch):
    from citens.config import settings

    seen = {}

    class _FakeBackend:
        def chat(self, system, user, **kw):
            return "ok"

    def fake_backend(model=None):
        seen["model"] = model
        return _FakeBackend()

    monkeypatch.setattr(llm_mod, "get_backend", fake_backend)
    monkeypatch.setattr(llm_mod, "_backends", {})
    # the disk cache would short-circuit before the backend is ever chosen
    monkeypatch.setattr(llm_mod.cache, "get", lambda *a, **k: None)
    monkeypatch.setattr(llm_mod.cache, "put", lambda *a, **k: None)
    monkeypatch.setattr(settings, "llm_model_cheap", "flash", raising=False)
    monkeypatch.setattr(settings, "llm_model_strong", "frontier", raising=False)

    llm_mod.chat("s", "u")
    assert seen["model"] == settings.llm_model
    llm_mod.chat("s", "u", strong=True)
    assert seen["model"] == "frontier"
    llm_mod.chat("s", "u", cheap=True)
    assert seen["model"] == "flash"
    # cheap wins when both flags given (explicit mechanical call)
    llm_mod.chat("s", "u", cheap=True, strong=True)
    assert seen["model"] == "flash"


def test_mechanical_stages_request_cheap(monkeypatch):
    """The routed agents must actually pass cheap=True through."""
    captured = {}

    def fake_chat_json(system, user, **kw):
        captured.update(kw)
        return {"concepts": [{"term": "t", "synonyms": []}]}

    monkeypatch.setattr("citens.agents.planner.chat_json", fake_chat_json)
    from citens.agents.planner import plan_queries

    plan_queries("some topic")
    assert captured.get("cheap") is True


# --- red team -------------------------------------------------------------------


_FINDINGS = {
    "findings": [
        {
            "type": "OVERCLAIM",
            "severity": "high",
            "excerpt": "X causes Y",
            "attack": "correlational evidence only",
            "fix": "weaken to association",
        },
        {
            "type": "MISSING_LIMITATION",
            "severity": "low",
            "excerpt": "",
            "attack": "no limitations",
            "fix": "add section",
        },
    ]
}


def test_red_team_review_parses_and_sorts(monkeypatch):
    def fake_chat_json(system, user, **kw):
        assert kw.get("strong") is True  # attacker runs on the strong tier
        return _FINDINGS

    monkeypatch.setattr(redteam_mod, "chat_json", fake_chat_json)
    out = red_team_review("# Review\nX causes Y", context_note="precision 90%")
    assert out[0]["severity"] == "high"  # high sorted first
    assert out[0]["type"] == "OVERCLAIM"
    assert len(out) == 2


def test_red_team_review_swallows_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(redteam_mod, "chat_json", boom)
    assert red_team_review("# Review") == []


def test_apply_fixes_rejects_fragments(monkeypatch):
    # a revision half the length of the original = model balked -> None,
    # never silently truncate the review
    def fake_chat_json(system, user, **kw):
        return {"review": "# stub"}

    monkeypatch.setattr(redteam_mod, "chat_json", fake_chat_json)
    assert apply_red_team_fixes("# Review\n" + "body " * 500, _FINDINGS["findings"]) is None
    assert apply_red_team_fixes("# Review", []) is None


def test_apply_fixes_accepts_full_rewrite(monkeypatch):
    body = "# Review\n" + "careful body text. " * 200

    def fake_chat_json(system, user, **kw):
        return {"review": body + "\n## Limitations\n\nhonest ones"}

    monkeypatch.setattr(redteam_mod, "chat_json", fake_chat_json)
    out = apply_red_team_fixes(body, _FINDINGS["findings"])
    assert out is not None and "Limitations" in out
