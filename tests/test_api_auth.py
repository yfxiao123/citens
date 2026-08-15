"""API auth + CORS: /run must not be an open wallet on a exposed server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    from citens.api.app import app
    from citens.config import settings

    monkeypatch.setattr(settings, "api_token", "s3cret")
    # CORS off for these tests (only auth behavior)
    monkeypatch.setattr(settings, "cors_origins", "")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def open_client(monkeypatch):
    from citens.api.app import app
    from citens.config import settings

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
