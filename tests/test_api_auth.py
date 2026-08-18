"""API auth + CORS: /run must not be an open wallet on a exposed server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from citens.api.app import app
from citens.config import settings


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "s3cret")
    # CORS off for these tests (only auth behavior)
    monkeypatch.setattr(settings, "cors_origins", "")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def open_client(monkeypatch):
    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "cors_origins", "")
    with TestClient(app) as c:
        yield c


def test_health_stays_public(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_run_requires_token(client):
    r = client.post("/run", json={"topic": "deep learning"})
    assert r.status_code == 401


def test_run_rejects_wrong_token(client):
    r = client.post(
        "/run", json={"topic": "deep learning"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_runs_and_result_require_token(client):
    assert client.get("/runs").status_code == 401
    assert client.get("/result/whatever").status_code == 401


def test_no_token_configured_means_no_auth(open_client):
    """Localhost dev default: everything open."""
    assert open_client.get("/runs").status_code == 200


# --- artifact file serving (the console's audit-browser / export links) ----


def test_artifact_requires_token(client):
    assert client.get("/artifact/x/review_browser.html").status_code == 401


def test_artifact_whitelist_and_traversal(open_client, monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "review_browser.html").write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "runs"))

    # whitelisted file serves with its content type
    r = open_client.get("/artifact/r1/review_browser.html")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert b"ok" in r.content

    # non-whitelisted name: 404, never a directory listing or arbitrary file
    assert open_client.get("/artifact/r1/meta.json").status_code == 404
    assert open_client.get("/artifact/r1/../../.env").status_code == 404
    assert open_client.get("/artifact/..%2F..%2F.env/x").status_code == 404
    assert open_client.get("/artifact/no-such-run/review.md").status_code == 404
