"""History ordering + the stable-data-home rules (anti "records vanished").

The 2026-08-23 report: after downloading a new exe to a different folder,
the history panel was empty — data defaulted to "next to the exe", so every
fresh download opened a brand-new workspace. Resolution rules now reattach
to the last-used workdir; /runs sorts by actual run time (dir-NAME reverse
sort put 中文 topics in Unicode order, not time order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from citens import desktop  # noqa: E402
from citens.config import settings  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "api_token", "")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path / "runs"))
    from citens.api.app import app

    with TestClient(app) as c:
        yield c


def _make_run(root: Path, name: str, topic: str, precision=None):
    d = root / name
    d.mkdir(parents=True)
    (d / "review.md").write_text("# x", encoding="utf-8")
    meta = {"topic": topic}
    if precision is not None:
        meta["citation_precision"] = precision
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_runs_sorted_by_time_not_topic_unicode(client, tmp_path):
    root = tmp_path / "runs"
    # Unicode order would put 订单簿 (U+8BA2) after 生成式 (U+751F); the
    # 2026-08-23 timestamps are the ground truth
    _make_run(root, "订单簿建模-20260823_020000", "订单簿建模")
    _make_run(root, "生成式推荐-20260823_030000", "生成式推荐")
    _make_run(root, "大语言模型-20260822_235959", "大语言模型")
    j = client.get("/runs").json()
    topics = [r["topic"] for r in j["runs"]]
    assert topics == ["生成式推荐", "订单簿建模", "大语言模型"]
    assert j["runs"][0]["time"].startswith("08-23 03")


def test_runs_skips_incomplete_dirs_and_formats_time(client, tmp_path):
    root = tmp_path / "runs"
    (root / "incomplete-20260823_010000").mkdir(parents=True)  # no review.md
    _make_run(root, "ok-run-20260823_050000", "ok")
    j = client.get("/runs").json()
    assert [r["topic"] for r in j["runs"]] == ["ok"]
    assert j["runs"][0]["time"] == "08-23 05:00"


# --- workdir resolution -------------------------------------------------------


@pytest.fixture
def homes(tmp_path, monkeypatch):
    app_dir = tmp_path / "appdir"
    app_dir.mkdir()
    home = tmp_path / "localdata"
    monkeypatch.setenv("LOCALAPPDATA", str(home))
    return app_dir, home


def test_first_launch_creates_machine_home(homes):
    app_dir, home = homes
    wd = desktop._resolve_workdir(app_dir)
    assert wd == home / "CiteLens"
    assert wd.is_dir()
    assert (home / "CiteLens" / "workdir.txt").is_file()


def test_bare_exe_reattaches_to_last_workdir(homes, tmp_path):
    app_dir, home = homes
    # a previous launch (from anywhere) used a data dir; the pointer remembers
    prev = tmp_path / "mydata"
    prev.mkdir()
    desktop._remember_workdir(prev)
    # a freshly downloaded exe in a NEW folder, no .env next to it:
    fresh = tmp_path / "Downloads" / "appdir2"
    fresh.mkdir(parents=True)
    assert desktop._resolve_workdir(fresh) == prev


def test_env_next_to_exe_wins_portable_mode(homes):
    app_dir, home = homes
    (app_dir / ".env").write_text("LLM_MODEL=deepseek-chat\n", encoding="utf-8")
    assert desktop._resolve_workdir(app_dir) == app_dir


def test_redirect_wins_and_migrates_env(homes, tmp_path):
    app_dir, home = homes
    wd = tmp_path / "chosen"
    (app_dir / ".env").write_text(
        f"CITELENS_WORKDIR={wd}\nLLM_API_KEY=sk-x\n", encoding="utf-8"
    )
    got = desktop._resolve_workdir(app_dir)
    assert got == wd
    # config migrated into the workdir (minus the pointer line)
    env = (wd / ".env").read_text(encoding="utf-8")
    assert "LLM_API_KEY=sk-x" in env
    assert "CITELENS_WORKDIR" not in env
    # and the pointer updated so future bare exe copies reattach here
    pointer = Path(desktop._machine_home() / "workdir.txt").read_text(encoding="utf-8")
    assert str(wd) in pointer


def test_stale_pointer_falls_back_to_home(homes, tmp_path):
    app_dir, home = homes
    stale = tmp_path / "deleted-dir"
    (home / "CiteLens").mkdir(parents=True)
    (home / "CiteLens" / "workdir.txt").write_text(str(stale), encoding="utf-8")
    assert desktop._resolve_workdir(app_dir) == home / "CiteLens"
