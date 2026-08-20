"""Settings API: masked read-back, .env write-through, live apply, auth."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from citens import llm  # noqa: E402
from citens.config import settings  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd is the portable app dir: .env lives here
    monkeypatch.setattr(settings, "api_token", "")
    from citens.api.app import app

    with TestClient(app) as c:
        yield c


def test_get_masks_secrets_and_reports_workdir(client):
    (Path.cwd() / ".env").write_text(
        "LLM_API_KEY=sk-1234567890abcdef\nLLM_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    j = client.get("/settings").json()
    by_key = {f["key"]: f for f in j["fields"]}
    assert by_key["LLM_API_KEY"]["current"].startswith("sk-1")
    assert "abcdef" not in by_key["LLM_API_KEY"]["current"]  # tail only
    assert by_key["LLM_API_KEY"]["set"] is True
    assert by_key["LLM_MODEL"]["current"] == "deepseek-chat"
    assert j["workdir"] == str(Path.cwd())


def test_save_writes_env_and_applies_in_memory(client, monkeypatch):
    (Path.cwd() / ".env").write_text(
        "LLM_API_KEY=sk-oldkey000000\nLLM_MODEL=old-model\nCUSTOM_LINE=keep-me\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm, "_backends", {"old-model": object()})
    r = client.post(
        "/settings",
        json={"updates": {"LLM_API_KEY": "sk-newkey123456", "LLM_MODEL": "new-model"}},
    )
    assert r.status_code == 200
    assert r.json()["applied"] == ["LLM_API_KEY", "LLM_MODEL"]
    # live settings object updated, backend cache dropped
    assert settings.llm_api_key == "sk-newkey123456"
    assert settings.llm_model == "new-model"
    assert llm._backends == {}
    # .env rewritten, unknown lines preserved, secrets readable
    env = (Path.cwd() / ".env").read_text(encoding="utf-8")
    assert "sk-newkey123456" in env
    assert "CUSTOM_LINE=keep-me" in env
    assert "old-model" not in env


def test_save_rejects_unknown_keys(client):
    r = client.post("/settings", json={"updates": {"EVIL_KEY": "x"}})
    assert r.status_code == 400


def test_settings_require_token_when_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_token", "sekrit")
    assert client.get("/settings").status_code == 401
    assert client.post("/settings", json={"updates": {}}).status_code == 401


def test_env_store_roundtrip(tmp_path):
    from citens.api.envstore import read_env_value, update_env_file

    f = tmp_path / ".env"
    f.write_text("# comment stays\nA=1\n\nB=2\n", encoding="utf-8")
    update_env_file(f, {"B": "22", "C": "3"})
    text = f.read_text(encoding="utf-8")
    assert "# comment stays" in text
    assert "A=1" in text and "B=22" in text and "C=3" in text
    update_env_file(f, {"A": ""})  # empty -> removed
    assert read_env_value(f, "A") is None
    assert read_env_value(f, "B") == "22"


def test_health_reports_llm_configured(client, monkeypatch):
    (Path.cwd() / ".env").write_text("LLM_API_KEY=sk-1234567890\n", encoding="utf-8")
    monkeypatch.setattr(settings, "llm_api_key", "sk-1234567890")
    assert client.get("/health").json()["llm_configured"] is True
    monkeypatch.setattr(settings, "llm_api_key", "")
    assert client.get("/health").json()["llm_configured"] is False
