#!/usr/bin/env python
"""install.py — register the CiteLens MCP tool with the agents on this machine.

Agent-agnostic by design
------------------------
Almost every modern agent client speaks **MCP over stdio**, and almost all of
them store MCP servers in the *same shape* (a JSON map at some path, or a TOML
table). So this installer is not a pile of per-agent adapters: it reads a
declarative registry (`agents.json`) and has four writers.

    dialect          how the server is registered                    used by
    --------------   ----------------------------------------------  ----------------------------
    json-map         file[container...][name] = {type,command,args}   claude zcode cursor windsurf
                                                                     gemini qwen cline roo kilo ...
    json-opencode    file.mcp[name] = {type:local, command:[...]}     opencode
    toml-block       file += [mcp_servers.name] command/args/env      codex
    dsh-bundle       bundle patch row + `dsh plugin add`              dsh
    snippet          we only print paste-ready config                 zed goose continue amp ...
    cli-only         no MCP client; the CLI shim is the integration   aider, scripts, CI

Adding an agent is adding an entry to `agents.json` — or dropping your own JSON
into `~/.citlens/agents.d/` (same schema; your entry wins). No code change.

The other half of "works with any agent" is not registration at all:

  * one shared data home (`~/.citlens/config.json` -> home) means every agent
    reads and writes the SAME literature pool, so an agent that searched can
    hand its pool to an agent that writes the review;
  * jobs are detached processes, so a long search survives the session that
    started it and any other agent can poll `status` / read the artifacts;
  * the skill (`~/.agents/skills/citlens`) is the cross-agent convention: DSH,
    Codex and the `npx skills` ecosystem read it, so the *method* travels even
    to a client that never saw this installer.

Usage
-----
    python install.py list                     # every known client, detected or not
    python install.py status                   # what is registered right now
    python install.py plan                     # dry run of apply (detected clients)
    python install.py apply                    # register every DETECTED client
    python install.py apply --all              # also create configs for absent clients
    python install.py apply --only cursor,gemini
    python install.py snippet zed              # paste-ready config for a manual client
    python install.py uninstall                # remove only what this script added

Safety: a file is never written without a `<file>.citlens-backup-<ts>` copy, TOML
edits live between `# >>> citlens ...>>>` markers, JSON edits only add/remove the
one `citlens` key, a skill dir is only deleted when its SKILL.md says
`name: citlens`, and no secret is ever written (keys stay in the engine's .env).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../citlens-mode/portable
MODE_ROOT = HERE.parent                          # .../citlens-mode
REPO_ROOT = MODE_ROOT.parent                     # .../personal-knowledge

SERVER = HERE / "citlens_mcp.py"
SKILL_SRC = HERE / "skill" / "citlens"
REGISTRY_FILE = HERE / "agents.json"

USER = Path.home()
CITLENS_DIR = USER / ".citlens"
CONFIG_PATH = CITLENS_DIR / "config.json"
STATE_PATH = CITLENS_DIR / "state.json"
AGENTS_D = CITLENS_DIR / "agents.d"
BUNDLE_DIR = CITLENS_DIR / "dsh-bundle"
JOBS_DIR = CITLENS_DIR / "jobs"
SHIM_PATH = CITLENS_DIR / "bin" / "citlens.cmd"
SKILL_HUB = USER / ".agents" / "skills"

MCP_START = "# >>> citlens mcp (managed by ~/.citlens/install.py) >>>"
MCP_END = "# <<< citlens mcp <<<"
SKILL_START = "# >>> citlens skill (managed by ~/.citlens/install.py) >>>"
SKILL_END = "# <<< citlens skill <<<"

DSH_BUNDLE_NAME = "@local/dsh-citlens-mcp"
SERVER_NAME = "citlens"   # the MCP tool name every agent registers under


# ── registry ─────────────────────────────────────────────────────────────────


def load_registry() -> dict:
    """agents.json (built in) <- ~/.citlens/agents.d/*.json (user additions win)."""
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"cannot read registry {REGISTRY_FILE}: {e}")
    agents = dict(data.get("builtin") or {})
    if AGENTS_D.is_dir():
        for f in sorted(AGENTS_D.glob("*.json")):
            try:
                extra = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for name, entry in (extra.get("agents") or extra).items():
                if isinstance(entry, dict) and not name.startswith("_"):
                    agents[name] = {**agents.get(name, {}), **entry}
    return agents


def expand(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def entry_file(entry: dict) -> Path | None:
    return expand(entry.get("file"))


def is_detected(entry: dict) -> bool:
    for cand in entry.get("detect") or []:
        p = expand(cand)
        if p and p.exists():
            return True
    f = entry_file(entry)
    return bool(f and f.is_file())


def detected_agents(registry: dict) -> list[str]:
    return [n for n, e in registry.items() if is_detected(e)]


def writable_dialects() -> set[str]:
    return {"json-map", "json-opencode", "toml-block", "dsh-bundle"}


# ── discovery of the engine ─────────────────────────────────────────────────


def _first_existing(*cands) -> Path | None:
    for c in cands:
        if not c:
            continue
        p = expand(str(c))
        if p and p.exists():
            return p
    return None


def _existing_config() -> dict:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _drivers() -> tuple[str, str]:
    """The two driver scripts, in priority order.

    `portable/drivers/` is what a clone of the published branch carries; the
    `../tools/` copy is the local working tree this project grew up in. Keeping
    both means the same installer works before and after publishing.
    """
    for base in (HERE / "drivers", MODE_ROOT / "tools"):
        recall, review = base / "citlens_recall.py", base / "citens_review.py"
        if recall.is_file() and review.is_file():
            return str(recall), str(review)
    return str(HERE / "drivers" / "citlens_recall.py"), str(HERE / "drivers" / "citens_review.py")


def discover() -> dict:
    """Resolve engine / interpreter / drivers / data home.

    Priority: explicit environment, then an existing valid `~/.citlens/config.json`
    (so a fresh clone on an already-configured machine reuses the working engine
    instead of writing broken paths), then the on-disk layout.

    Two layouts are supported: the development one (`citlens-mode/portable` next
    to `../文献检索agent`) and the published one (`portable/` at the root of the
    engine repository, so the engine is `MODE_ROOT` itself).
    """
    old = _existing_config()
    env_engine = os.environ.get("CITLENS_ENGINE")
    engine = _first_existing(
        env_engine,
        old.get("engine") if env_engine is None else None,
        MODE_ROOT if (MODE_ROOT / "citens").is_dir() else None,
        REPO_ROOT / "文献检索agent",
        REPO_ROOT / "文献检索agent-minimal",
    )
    python = _first_existing(
        os.environ.get("CITLENS_PYTHON"),
        old.get("python") if engine and old.get("engine") and str(engine) == str(old.get("engine")) else None,
        (engine / ".venv" / "Scripts" / "python.exe") if engine else None,
        (engine / ".venv" / "bin" / "python") if engine else None,
        sys.executable,
    )
    home = _first_existing(
        os.environ.get("CITLENS_HOME"),
        old.get("home"),
        REPO_ROOT / "citlens test",
        engine / "data" if engine else None,
        CITLENS_DIR / "data",
    )
    recall_drv, review_drv = _drivers()
    old_recall = _first_existing(old.get("recall_driver"))
    old_review = _first_existing(old.get("review_driver"))
    if old_recall and old_review:
        recall_drv, review_drv = str(old_recall), str(old_review)
    return {
        "home": str(home) if home else None,
        "python": str(python) if python else sys.executable,
        "engine": str(engine) if engine else None,
        "recall_driver": recall_drv,
        "review_driver": review_drv,
        "jobs_dir": str(JOBS_DIR),
        "sources": ["arxiv", "semantic_scholar", "openalex", "crossref"],
        "families": ["planned", "raw", "hyde", "pivot", "oa_title",
                     "s2_rec", "snowball", "web", "survey"],
        "target": 60,
        "language": "zh",
        "mode": "deep_review",
        "papers": 12,
        "wait_seconds": 45,
        "max_concurrent_jobs": 3,
    }


# ── planning / io ───────────────────────────────────────────────────────────


class Plan:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.changes: list[str] = []
        self.backups: list[str] = []

    def note(self, msg: str) -> None:
        self.changes.append(msg)
        print(("  [plan] " if self.dry_run else "  [done] ") + msg)

    def info(self, msg: str) -> None:
        print("  [info] " + msg)

    def backup(self, path: Path) -> None:
        if self.dry_run or not path.is_file():
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = path.with_name(path.name + f".citlens-backup-{ts}")
        if bak.exists():
            bak = path.with_name(path.name + f".citlens-backup-{ts}-{len(self.backups)}")
        shutil.copy2(path, bak)
        self.backups.append(str(bak))
        print(f"  [backup] {bak}")

    def write_text(self, path: Path, text: str) -> bool:
        try:
            if path.is_file():
                with path.open(encoding="utf-8", newline="") as fh:
                    if fh.read() == text:
                        return False
        except Exception:  # noqa: BLE001
            pass
        if self.dry_run:
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".citlens-tmp")
        tmp.write_text(text, encoding="utf-8", newline="")
        os.replace(tmp, path)
        return True


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except Exception:  # noqa: BLE001
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            return ""


def _load_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(default)
    except Exception:  # noqa: BLE001
        return dict(default)


def _load_json_checked(path: Path, default: dict) -> tuple[dict, bool]:
    """Read a client config, but never turn "unparsable" into "empty".

    An agent config can be JSON5-with-comments or mid-write. Overwriting it with
    a fresh object would silently destroy the user's settings, so the writers
    abort instead and leave the file (and the backup policy) alone.
    """
    if not path.is_file():
        return dict(default), True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return dict(default), False
    if not isinstance(data, dict):
        return dict(default), False
    return data, True


def _dump_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _strip_block(text: str, start: str, end: str) -> str:
    pat = re.compile(rf"^{re.escape(start)}\s*$.*?^{re.escape(end)}\s*$\n?", re.M | re.S)
    return pat.sub("", text)


def _upsert_block(text: str, start: str, end: str, body: str) -> str:
    text = _strip_block(text, start, end)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + ("\n" if text.strip() else "") + f"{start}\n{body.rstrip()}\n{end}\n"


def _toml_str(value: str) -> str:
    if "'" in value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return f"'{value}'"


def _toml_list(values) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _server_argv(cfg: dict) -> list[str]:
    return [str(SERVER), "serve", "--config", str(CONFIG_PATH)]


def _server_env() -> dict:
    return {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _stdio_entry(cfg: dict, with_env: bool = True) -> dict:
    entry = {"type": "stdio", "command": cfg["python"], "args": _server_argv(cfg)}
    if with_env:
        entry["env"] = _server_env()
    return entry


# ── shared config + CLI shim + skill hub ────────────────────────────────────


def install_config(p: Plan, cfg: dict) -> None:
    existing = {}
    try:
        existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        existing = {}
    merged = {**cfg, **{k: v for k, v in existing.items() if v not in (None, "", [], {})}}
    for key in ("home", "python", "recall_driver", "review_driver", "jobs_dir"):
        merged[key] = cfg[key]
    text = _dump_json(merged)
    if _read_text(CONFIG_PATH) == text:
        return
    if p.dry_run:
        p.note(f"would write shared config {CONFIG_PATH}")
        return
    p.backup(CONFIG_PATH)
    if p.write_text(CONFIG_PATH, text):
        p.note(f"shared config {CONFIG_PATH}")


def install_shim(p: Plan, cfg: dict) -> None:
    body = (
        "@echo off\r\n"
        "rem citlens CLI fallback (managed by ~/.citlens/install.py)\r\n"
        "setlocal\r\n"
        'set "PYTHONUTF8=1"\r\n'
        'set "PYTHONIOENCODING=utf-8"\r\n'
        f'"{cfg["python"]}" "{SERVER}" cli --config "{CONFIG_PATH}" %*\r\n'
    )
    if _read_text(SHIM_PATH) == body:
        return
    if p.dry_run:
        p.note(f"would write CLI shim {SHIM_PATH}")
        return
    SHIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if p.write_text(SHIM_PATH, body):
        p.note(f"CLI shim {SHIM_PATH}")


def _skill_dir_is_ours(path: Path) -> bool:
    f = path / "SKILL.md"
    if not f.is_file():
        return False
    return bool(re.search(r"^name:\s*citlens\s*$", _read_text(f), re.M))


def _tree_equal(src: Path, dst: Path) -> bool:
    if not dst.is_dir():
        return False
    src_files = {p.relative_to(src): p for p in src.rglob("*") if p.is_file()}
    dst_files = {p.relative_to(dst): p for p in dst.rglob("*") if p.is_file()}
    if set(src_files) != set(dst_files):
        return False
    for rel, sp in src_files.items():
        try:
            if sp.read_bytes() != dst_files[rel].read_bytes():
                return False
        except Exception:  # noqa: BLE001
            return False
    return True


def skill_targets(registry: dict, names: list[str] | None = None) -> list[Path]:
    """The shared hub plus every skill directory of the agents being installed.

    The hub (`~/.agents/skills`) is the cross-agent convention and is always
    written; per-agent directories are only touched for agents actually present,
    so `apply` never litters a machine with skill dirs for absent clients.
    """
    out: list[Path] = [SKILL_HUB / "citlens"]
    wanted = set(names) if names is not None else set(registry)
    for name, entry in registry.items():
        if name not in wanted:
            continue
        for d in entry.get("skillDirs") or []:
            p = expand(d)
            if p:
                target = p / "citlens" if p.name != "citlens" else p
                if target not in out:
                    out.append(target)
    return out


def install_skills(p: Plan, registry: dict, names: list[str] | None = None) -> None:
    if not (SKILL_SRC / "SKILL.md").is_file():
        p.note(f"!! skill source missing: {SKILL_SRC}/SKILL.md")
        return
    for target in skill_targets(registry, names):
        if target.exists() and not _skill_dir_is_ours(target):
            p.note(f"!! refusing to overwrite foreign skill dir {target}")
            continue
        if _tree_equal(SKILL_SRC, target):
            continue
        if p.dry_run:
            p.note(f"would copy skill -> {target}")
            continue
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(SKILL_SRC, target)
        p.note(f"skill -> {target}")


def remove_skills(p: Plan, registry: dict) -> None:
    for target in skill_targets(registry):
        if not target.exists():
            continue
        if not _skill_dir_is_ours(target):
            p.info(f"leaving foreign skill dir alone: {target}")
            continue
        if p.dry_run:
            p.note(f"would remove skill {target}")
            continue
        shutil.rmtree(target)
        p.note(f"removed skill {target}")


# ── dialect writers ─────────────────────────────────────────────────────────


def _container(data: dict, path: list[str], create: bool) -> dict | None:
    node = data
    for key in path:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            if not create:
                return None
            nxt = {}
            node[key] = nxt
        node = nxt
    return node


def agent_json_map(p: Plan, name: str, entry: dict, cfg: dict, dry_run: bool) -> None:
    path = entry_file(entry)
    if not path:
        p.note(f"!! {name}: registry entry has no file")
        return
    data, ok = _load_json_checked(path, {})
    if not ok:
        p.note(f"!! {name}: {path} is not plain JSON (comments or half-written?); leaving it "
               f"alone — run `install.py snippet {name}` and paste the entry")
        return
    container_path = list(entry.get("container") or ["mcpServers"])
    servers = _container(data, container_path, create=True)
    wanted = _stdio_entry(cfg, with_env=bool(entry.get("env", True)))
    if servers.get(SERVER_NAME) == wanted:
        p.info(f"{instance(name, entry)}: already registered ({path})")
        return
    servers[SERVER_NAME] = wanted
    text = _dump_json(data)
    if dry_run:
        p.note(f"would set {'.'.join(container_path)}.{SERVER_NAME} in {path}")
        return
    p.backup(path)
    p.write_text(path, text)
    p.note(f"{name} <- {path}")


def agent_json_opencode(p: Plan, name: str, entry: dict, cfg: dict, dry_run: bool) -> None:
    path = entry_file(entry)
    if not path:
        p.note(f"!! {name}: registry entry has no file")
        return
    data, ok = _load_json_checked(path, {"$schema": "https://opencode.ai/config.json"})
    if not ok:
        p.note(f"!! {name}: {path} is not plain JSON; leaving it alone — run "
               f"`install.py snippet {name}` and paste the entry")
        return
    mcp = _container(data, list(entry.get("container") or ["mcp"]), create=True)
    wanted = {
        "type": "local",
        "command": [cfg["python"], *_server_argv(cfg)],
        "enabled": True,
        "environment": _server_env(),
    }
    if mcp.get(SERVER_NAME) == wanted:
        p.info(f"{name}: already registered ({path})")
        return
    mcp[SERVER_NAME] = wanted
    text = _dump_json(data)
    if dry_run:
        p.note(f"would set mcp.{SERVER_NAME} in {path}")
        return
    p.backup(path)
    p.write_text(path, text)
    p.note(f"{name} <- {path}")


def agent_toml_block(p: Plan, name: str, entry: dict, cfg: dict, dry_run: bool) -> None:
    path = entry_file(entry)
    if not path:
        p.note(f"!! {name}: registry entry has no file")
        return
    text = _read_text(path)
    body = "\n".join([
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {_toml_str(cfg['python'])}",
        f"args = {_toml_list(_server_argv(cfg))}",
        "startup_timeout_sec = 60",
        "",
        f"[mcp_servers.{SERVER_NAME}.env]",
        'PYTHONIOENCODING = "utf-8"',
        'PYTHONUTF8 = "1"',
    ])
    new = _upsert_block(text, MCP_START, MCP_END, body)
    skill_md = expand(entry.get("skillEntry")) if entry.get("skillEntry") else None
    if skill_md and skill_md.is_file():
        new = _upsert_block(new, SKILL_START, SKILL_END, "\n".join([
            "[[skills.config]]", f"path = {_toml_str(str(skill_md))}", "enabled = true"]))
    if new == text:
        p.info(f"{name}: already registered ({path})")
        return
    if dry_run:
        p.note(f"would add [mcp_servers.{SERVER_NAME}] to {path}")
        return
    p.backup(path)
    p.write_text(path, new)
    p.note(f"{name} <- {path}")


def remove_simple(p: Plan, name: str, entry: dict, dry_run: bool) -> None:
    dialect = entry.get("dialect")
    if dialect == "dsh-bundle":
        remove_dsh(p, dry_run)
        return
    path = entry_file(entry)
    if not path or not path.is_file():
        p.info(f"{name}: no config file")
        return
    if dialect == "toml-block":
        text = _read_text(path)
        new = _strip_block(_strip_block(text, SKILL_START, SKILL_END), MCP_START, MCP_END)
        if new == text:
            p.info(f"{name}: nothing to remove")
            return
        if dry_run:
            p.note(f"would strip citlens blocks from {path}")
            return
        p.backup(path)
        p.write_text(path, new)
        p.note(f"{name} removed <- {path}")
        return
    data, ok = _load_json_checked(path, {})
    if not ok:
        p.info(f"{name}: {path} is not plain JSON; not touching it")
        return
    container = _container(data, list(entry.get("container") or ["mcpServers"]), create=False)
    if not container or SERVER_NAME not in container:
        p.info(f"{name}: nothing to remove")
        return
    del container[SERVER_NAME]
    if dry_run:
        p.note(f"would remove {SERVER_NAME} from {path}")
        return
    p.backup(path)
    p.write_text(path, _dump_json(data))
    p.note(f"{name} removed <- {path}")


# ── DSH (bundle-based) ──────────────────────────────────────────────────────


def dsh_home() -> Path:
    return expand(os.environ.get("DSH_HOME") or (USER / ".dsh"))  # type: ignore[return-value]


def dsh_bundle_files(cfg: dict) -> tuple[str, str]:
    pkg = _dump_json({
        "name": DSH_BUNDLE_NAME,
        "version": "1.0.0",
        "private": True,
        "dsh": {"bundle": {"patch": "./cordis.patch.yml"}},
    })

    def q(s: str) -> str:
        return "'" + str(s).replace("\\", "/") + "'"

    patch = (
        "# CiteLens MCP client row (generated by ~/.citlens/install.py).\n"
        "# Host-level, so every DSH preset/mode gets the mcp__citlens__* tools.\n"
        "- insert:\n"
        "    - id: citlens-mcp\n"
        "      name: '@deepseek-ai/dsh-mcp-client'\n"
        "      config:\n"
        "        serverName: citlens\n"
        "        transport: stdio\n"
        f"        command: {q(cfg['python'])}\n"
        "        args:\n"
        + "".join(f"          - {q(a)}\n" for a in _server_argv(cfg))
        + "        env:\n"
        "          PYTHONIOENCODING: utf-8\n"
        "          PYTHONUTF8: '1'\n"
        "        failOnStartupError: true\n"
    )
    return pkg, patch


def _dsh_profile_has_bundle(profile_dir: Path) -> bool:
    try:
        pkg = json.loads((profile_dir / "package.json").read_text(encoding="utf-8"))
        return DSH_BUNDLE_NAME in (pkg.get("dsh", {}).get("profile", {}).get("bundles") or [])
    except Exception:  # noqa: BLE001
        return False


def install_dsh(p: Plan, cfg: dict, dry_run: bool, profile: str, auto: bool) -> None:
    pkg, patch = dsh_bundle_files(cfg)
    profile_dir = dsh_home() / "profiles" / profile
    if not profile_dir.is_dir():
        p.info(f"dsh: profile '{profile}' not found under {dsh_home()} (pass --dsh-profile)")
        return
    stale = True
    try:
        stale = ((BUNDLE_DIR / "package.json").read_text(encoding="utf-8") != pkg
                 or (BUNDLE_DIR / "cordis.patch.yml").read_text(encoding="utf-8") != patch)
    except Exception:  # noqa: BLE001
        stale = True
    if dry_run:
        if stale:
            p.note(f"would write bundle {BUNDLE_DIR}")
        if _dsh_profile_has_bundle(profile_dir):
            p.info(f"dsh profile '{profile}' already lists {DSH_BUNDLE_NAME}")
        else:
            p.note(f"would run: dsh plugin --profile {profile} add \"{BUNDLE_DIR}\"")
        return
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    wrote = p.write_text(BUNDLE_DIR / "package.json", pkg)
    wrote = p.write_text(BUNDLE_DIR / "cordis.patch.yml", patch) or wrote
    if wrote:
        p.note(f"dsh bundle {BUNDLE_DIR}")
    if _dsh_profile_has_bundle(profile_dir):
        if wrote:
            p.info(f"dsh profile '{profile}' already lists {DSH_BUNDLE_NAME}")
        return
    dsh = shutil.which("dsh") or shutil.which("dsh.cmd")
    if not (auto and dsh):
        print(f"  [manual] register it yourself:  dsh plugin --profile {profile} add \"{BUNDLE_DIR}\"")
        return
    try:
        r = subprocess.run([dsh, "plugin", "--profile", profile, "add", str(BUNDLE_DIR)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300)
        if _dsh_profile_has_bundle(profile_dir):
            p.note(f"dsh <- {profile_dir / 'package.json'} (via dsh plugin add)")
        else:
            p.note(f"!! dsh plugin add ran (exit {r.returncode}) but the profile does not list "
                   f"{DSH_BUNDLE_NAME}; run manually:\n"
                   f"    dsh plugin --profile {profile} add \"{BUNDLE_DIR}\"")
    except Exception as e:  # noqa: BLE001
        p.note(f"!! dsh plugin add failed ({e}); run manually: "
               f"dsh plugin --profile {profile} add \"{BUNDLE_DIR}\"")


def remove_dsh(p: Plan, dry_run: bool, profile: str = "desktop", auto: bool = True) -> None:
    profile_dir = dsh_home() / "profiles" / profile
    if not _dsh_profile_has_bundle(profile_dir):
        p.info("dsh: nothing to remove")
        return
    if dry_run:
        p.note(f"would run: dsh plugin --profile {profile} remove {DSH_BUNDLE_NAME}")
        return
    dsh = shutil.which("dsh") or shutil.which("dsh.cmd")
    if auto and dsh:
        try:
            subprocess.run([dsh, "plugin", "--profile", profile, "remove", DSH_BUNDLE_NAME],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300)
        except Exception as e:  # noqa: BLE001
            p.note(f"!! dsh plugin remove failed: {e}")
    if _dsh_profile_has_bundle(profile_dir):
        p.note(f"!! still listed in {profile_dir / 'package.json'}; remove \"{DSH_BUNDLE_NAME}\" "
               "from dsh.profile.bundles by hand")
    else:
        p.note(f"dsh removed from profile '{profile}'")


# ── snippets (for clients we cannot safely write) ──────────────────────────


def instance(name: str, entry: dict) -> str:
    return f"{name} ({entry.get('label') or name})"


def print_snippet(name: str, entry: dict, cfg: dict) -> int:
    dialect = entry.get("dialect")
    print(f"\n# {instance(name, entry)}  [{entry.get('confidence', '?')}]")
    if entry.get("note"):
        print(f"# {entry['note']}")
    if entry.get("file"):
        print(f"# file: {expand(entry['file'])}")
    print()
    if dialect == "toml-block":
        print("\n".join([
            f"[mcp_servers.{SERVER_NAME}]",
            f"command = {_toml_str(cfg['python'])}",
            f"args = {_toml_list(_server_argv(cfg))}",
            "",
            f"[mcp_servers.{SERVER_NAME}.env]",
            'PYTHONIOENCODING = "utf-8"',
            'PYTHONUTF8 = "1"',
        ]))
    elif dialect == "json-opencode":
        print(_dump_json({"mcp": {SERVER_NAME: {
            "type": "local",
            "command": [cfg["python"], *_server_argv(cfg)],
            "enabled": True,
            "environment": _server_env(),
        }}}))
    elif dialect == "dsh-bundle":
        _, patch = dsh_bundle_files(cfg)
        print(patch)
    elif dialect == "cli-only":
        print("No MCP client. Integration path:")
        print(f'  "{SHIM_PATH}" doctor')
        print(f'  "{SHIM_PATH}" search --topic "<topic>" --target 80')
        print("Or point the agent at the skill file so it knows the tool exists:")
        print(f"  {SKILL_HUB / 'citlens' / 'SKILL.md'}")
    else:  # json-map and any unknown shape
        print(_dump_json({"mcpServers": {SERVER_NAME: _stdio_entry(cfg)}}))
        if entry.get("container") and list(entry["container"]) != ["mcpServers"]:
            print(f"# merge the inner object under: {'.'.join(entry['container'])}.{SERVER_NAME}")
    print("\n# Shared config (already written by `apply`):", CONFIG_PATH)
    print(f"# Skill card: {SKILL_HUB / 'citlens' / 'SKILL.md'}")
    return 0


# ── commands ────────────────────────────────────────────────────────────────


def cmd_list(registry: dict) -> int:
    det = set(detected_agents(registry))
    print(f"{'agent':<12} {'dialect':<14} {'confidence':<10} {'detected':<9} file / note")
    print("-" * 100)
    for name, entry in registry.items():
        f = entry_file(entry)
        loc = str(f) if f else (entry.get("note") or "")
        print(f"{name:<12} {entry.get('dialect', '?'):<14} {entry.get('confidence', '?'):<10} "
              f"{('yes' if name in det else '-'):<9} {loc}")
    print(f"\n{len(det)} detected of {len(registry)} known. "
          f"`apply` registers the detected ones; `apply --all` also creates absent configs.")
    print("Add your own client: drop a file into " + str(AGENTS_D) + " shaped like agents.json.")
    return 0


def cmd_status(registry: dict) -> int:
    cfg = discover()
    print("CiteLens portable tool — registration status\n")
    print(f"server      {SERVER}")
    print(f"config      {CONFIG_PATH}  ({'present' if CONFIG_PATH.is_file() else 'MISSING'})")
    print(f"home        {cfg.get('home')}")
    print(f"python      {cfg.get('python')}")
    print(f"skill hub   {SKILL_HUB / 'citlens'}  "
          f"[{'present' if _skill_dir_is_ours(SKILL_HUB / 'citlens') else 'absent'}]")
    print()
    for name, entry in registry.items():
        dialect = entry.get("dialect")
        det = "detected" if is_detected(entry) else "not installed"
        if dialect == "dsh-bundle":
            states = []
            for prof in ("desktop", "web"):
                pd = dsh_home() / "profiles" / prof
                if pd.is_dir():
                    states.append(f"profile '{prof}': "
                                  f"{'registered' if _dsh_profile_has_bundle(pd) else 'absent'}")
            print(f"dsh          {det} | " + (" | ".join(states) or "no profiles"))
            continue
        path = entry_file(entry)
        if dialect == "toml-block":
            reg = MCP_START in _read_text(path) if path else False
        elif dialect in ("json-map", "json-opencode"):
            data = _load_json(path, {}) if path else {}
            container = _container(data, list(entry.get("container") or ["mcpServers"]),
                                   create=False) if path else None
            reg = bool(container and SERVER_NAME in container)
        elif dialect == "cli-only":
            reg = False  # nothing to register: this client has no MCP surface
        else:
            reg = False  # snippet-only: nothing to check
        if dialect == "cli-only":
            mark = "cli-ready" if SHIM_PATH.is_file() else "absent"
        elif dialect == "snippet":
            mark = "manual"
        else:
            mark = "registered" if reg else "absent"
        print(f"{name:<12} {det:<14} {mark:<12} {path or entry.get('note') or ''}")
    print()
    print("Per-agent skill dirs:")
    for t in skill_targets(registry):
        state = "present" if _skill_dir_is_ours(t) else ("FOREIGN" if t.exists() else "absent")
        print(f"  [{state:<7}] {t}")
    print("\nVerify inside any client with: doctor  (mcp__citlens__doctor / citlens:doctor)")
    return 0


def _apply_one(p: Plan, registry: dict, cfg: dict, name: str, remove: bool,
               dry_run: bool, profile: str, dsh_auto: bool) -> None:
    entry = registry.get(name)
    print(f"\n== {name} ==")
    if not entry:
        p.note(f"!! unknown agent {name!r}; see `install.py list`")
        return
    dialect = entry.get("dialect")
    if remove:
        if dialect == "dsh-bundle":
            remove_dsh(p, dry_run, profile, dsh_auto)
        elif dialect in writable_dialects():
            remove_simple(p, name, entry, dry_run)
        else:
            p.info(f"{name}: nothing was written by this installer")
        return
    if dialect == "json-map":
        agent_json_map(p, name, entry, cfg, dry_run)
    elif dialect == "json-opencode":
        agent_json_opencode(p, name, entry, cfg, dry_run)
    elif dialect == "toml-block":
        agent_toml_block(p, name, entry, cfg, dry_run)
    elif dialect == "dsh-bundle":
        install_dsh(p, cfg, dry_run, profile, dsh_auto)
    elif dialect == "cli-only":
        p.info(f"{name}: no MCP — use the CLI shim ({SHIM_PATH}); skill card explains it")
    else:
        p.info(f"{name}: cannot be written safely [{entry.get('confidence')}]; "
               f"run `install.py snippet {name}` and paste it")


def cmd_apply(registry: dict, names: list[str], *, dry_run: bool, remove: bool,
              profile: str, dsh_auto: bool, prelude: bool = True,
              overrides: dict | None = None) -> int:
    cfg = discover()
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    if overrides:
        if overrides.get("engine") and not overrides.get("python"):
            eng = Path(overrides["engine"]).resolve()
            for venv in (eng / ".venv" / "Scripts" / "python.exe", eng / ".venv" / "bin" / "python"):
                if venv.is_file():
                    overrides.setdefault("python", str(venv))
                    break
        for key in ("engine", "python", "home"):
            if overrides.get(key):
                cfg[key] = str(Path(overrides[key]).expanduser().resolve())
    p = Plan(dry_run)
    if dry_run:
        print("DRY RUN — nothing will be written.\n")
    if prelude and not remove:
        install_config(p, cfg)
        install_shim(p, cfg)
        install_skills(p, registry, names)
    elif prelude and remove:
        remove_skills(p, registry)
    for name in names:
        _apply_one(p, registry, cfg, name, remove, dry_run, profile, dsh_auto)
    print()
    if p.changes:
        print(("Would change: " if dry_run else "Changed: ") + f"{len(p.changes)} item(s)")
    else:
        print("Nothing to do — already up to date.")
    if not remove:
        print("\nVerify with:\n"
              f'  "{cfg["python"]}" "{SERVER}" selftest\n'
              f"  {SHIM_PATH} doctor")
    if not dry_run:
        state = {"version": 1, "last_run": {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "uninstall" if remove else ("plan" if dry_run else "apply"),
            "agents": names,
        }, "backups": p.backups, "config": str(CONFIG_PATH), "server": str(SERVER)}
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(_dump_json(state), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    only: list[str] = []
    profile = "desktop"
    dsh_auto = True
    all_clients = False
    overrides: dict = {}
    dry_run = cmd == "plan"
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--only" and i + 1 < len(rest):
            only = [x.strip() for x in rest[i + 1].split(",") if x.strip()]
            i += 2
        elif tok in ("--engine", "--home", "--python") and i + 1 < len(rest):
            overrides[tok[2:]] = rest[i + 1]
            i += 2
        elif tok == "--dsh-profile" and i + 1 < len(rest):
            profile = rest[i + 1]
            i += 2
        elif tok == "--no-dsh-auto":
            dsh_auto = False
            i += 1
        elif tok in ("--all", "-a"):
            all_clients = True
            i += 1
        elif tok == "--dry-run":
            dry_run = True
            i += 1
        else:
            i += 1
    registry = load_registry()
    if cmd == "list":
        return cmd_list(registry)
    if cmd == "status":
        return cmd_status(registry)
    if cmd == "snippet":
        target = rest[0] if rest and not rest[0].startswith("--") else ""
        if not target:
            print("usage: install.py snippet <agent>   (see `install.py list`)")
            return 2
        if target not in registry:
            print(f"unknown agent {target!r}")
            return 2
        return print_snippet(target, registry[target], discover())
    if cmd not in ("plan", "apply", "uninstall"):
        print(f"unknown command {cmd!r}; use list | plan | apply | status | snippet | uninstall")
        return 2
    if only:
        names = only
        unknown = [n for n in names if n not in registry]
        if unknown:
            print(f"unknown agent(s): {', '.join(unknown)} — see `install.py list`")
            return 2
    elif all_clients:
        names = [n for n, e in registry.items() if e.get("dialect") in writable_dialects()]
    else:
        names = detected_agents(registry)
        if not names:
            print("No agent detected on this machine. Use --all to create configs anyway, "
                  "or `list` to see the supported clients.")
            return 1
    return cmd_apply(registry, names, dry_run=dry_run or cmd == "plan",
                     remove=cmd == "uninstall", profile=profile, dsh_auto=dsh_auto,
                     overrides=overrides)


if __name__ == "__main__":
    sys.exit(main())
