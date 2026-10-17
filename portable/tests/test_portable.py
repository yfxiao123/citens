#!/usr/bin/env python
"""test_portable.py — offline verification of the portable CiteLens tool.

Two things are checked, both without network and without touching the user's
real agent configs:

1. the MCP server speaks the protocol correctly (handshake, tool list, tool
   calls, error paths, stdout purity, clean exit) — driven by a tiny stdio
   client in this file;
2. `install.py` discovers the engine and plans/applies idempotently against a
   throwaway HOME, so the four config dialects are exercised for real.

Run:  python test_portable.py            (plain, no pytest needed)
      python -m pytest test_portable.py  (if pytest is available)
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORTABLE = HERE.parent
SERVER = PORTABLE / "citlens_mcp.py"
INSTALL = PORTABLE / "install.py"

PASS, FAIL = [], []

for _stream in (sys.stdout, sys.stderr):
    try:  # a zh-CN console is GBK; config diffs contain non-GBK characters
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (f"   [{detail}]" if detail and not ok else ""))

# ── a minimal MCP stdio client ────────────────────────────────────────────────


class MCPClient:
    """Speaks newline-delimited JSON-RPC exactly like a real client would."""

    def __init__(self, config: Path):
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER), "serve", "--config", str(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.raw_stdout: list[str] = []
        self._next = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.raw_stdout.append(line)
            self.lines.put(line)

    def send(self, msg: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict | None = None, timeout: float = 120.0) -> dict:
        self._next += 1
        msg_id = self._next
        self.send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"no response to {method}")
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise TimeoutError(f"no response to {method}") from None
            msg = json.loads(line)
            if msg.get("id") == msg_id:
                return msg

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return -1


def _temp_config() -> tuple[Path, Path]:
    """Real engine paths (so doctor is meaningful) + a throwaway jobs dir."""
    sys.path.insert(0, str(PORTABLE))
    import install  # noqa: PLC0415

    cfg = install.discover()
    tmp = Path(tempfile.mkdtemp(prefix="citlens-test-"))
    cfg["jobs_dir"] = str(tmp / "jobs")
    cfg_path = tmp / "config.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    return cfg_path, tmp


# ── protocol tests ───────────────────────────────────────────────────────────


def test_protocol() -> None:
    print("\n== MCP protocol ==")
    cfg_path, tmp = _temp_config()
    c = MCPClient(cfg_path)
    try:
        init = c.request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1"},
        })
        res = init.get("result") or {}
        check("initialize returns protocolVersion", res.get("protocolVersion") == "2025-06-18",
              str(res.get("protocolVersion")))
        check("initialize advertises tools", bool((res.get("capabilities") or {}).get("tools") is not None))
        check("serverInfo name is citlens", (res.get("serverInfo") or {}).get("name") == "citlens")
        check("initialize carries instructions",
              "literature" in (res.get("instructions") or "").lower())

        # old clients negotiate down; unknown versions get our newest supported one
        init_old = c.request("initialize", {"protocolVersion": "2024-11-05"})
        check("protocol negotiation honours 2024-11-05",
              (init_old.get("result") or {}).get("protocolVersion") == "2024-11-05")
        init_bogus = c.request("initialize", {"protocolVersion": "1999-01-01"})
        check("unknown protocol version falls back",
              (init_bogus.get("result") or {}).get("protocolVersion") == "2025-06-18")

        c.notify("notifications/initialized")

        ping = c.request("ping")
        check("ping returns an empty result", ping.get("result") == {})

        tools = (c.request("tools/list").get("result") or {}).get("tools") or []
        names = {t.get("name") for t in tools}
        expected = {"search", "pool", "plan", "questions", "review", "status", "doctor"}
        check("tools/list exposes all seven tools", names == expected, str(sorted(names)))
        check("every tool has a description and an object inputSchema", all(
            t.get("description") and (t.get("inputSchema") or {}).get("type") == "object"
            for t in tools))
        search_schema = next(t for t in tools if t["name"] == "search")["inputSchema"]
        check("search schema requires topic", search_schema.get("required") == ["topic"])
        check("search schema enumerates the nine families", len(
            (search_schema["properties"]["families"]["items"].get("enum") or [])) == 9)

        # doctor: a real, offline tool call
        doc = c.request("tools/call", {"name": "doctor", "arguments": {}})
        payload = _payload(doc)
        check("doctor call is not an error", not (doc.get("result") or {}).get("isError"))
        check("doctor reports ok", payload.get("ok") is True, str(payload.get("notes")))
        check("doctor never leaks key values", "sk-" not in json.dumps(payload))
        check("doctor lists env key names", "env_keys_present" in payload)

        # refusals are structured results, not protocol errors
        refused = c.request("tools/call", {"name": "review",
                                          "arguments": {"topic": "test"}})
        rp = _payload(refused)
        check("review refuses without explicit user request",
              rp.get("state") == "refused" and (refused.get("result") or {}).get("isError") is True)

        bad = c.request("tools/call", {"name": "search",
                                       "arguments": {"topic": "x", "families": ["nope"]}})
        bp = _payload(bad)
        check("unknown family is rejected without spawning a job",
              bp.get("ok") is False and "unknown families" in str(bp.get("error")))

        missing = c.request("tools/call", {"name": "pool", "arguments": {}})
        check("pool without topic is a clean error",
              _payload(missing).get("ok") is False)

        unknown_tool = c.request("tools/call", {"name": "nope", "arguments": {}})
        check("unknown tool returns isError with the tool list",
              (unknown_tool.get("result") or {}).get("isError") is True
              and "tools" in _payload(unknown_tool))

        err = c.request("does/not/exist", {})
        check("unknown method returns -32601",
              (err.get("error") or {}).get("code") == -32601)

        res_empty = c.request("resources/list")
        check("resources/list answers empty (clients probe it)",
              (res_empty.get("result") or {}).get("resources") == [])

        # stdout purity: every frame must be a complete JSON-RPC object
        check("stdout contains only JSON-RPC frames", all(
            _is_jsonrpc_line(line) for line in c.raw_stdout))
        code = c.close()
        check("clean exit after stdin closes", code == 0, str(code))
    finally:
        try:
            c.proc.kill()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(tmp, ignore_errors=True)


def _payload(response: dict) -> dict:
    content = (response.get("result") or {}).get("content") or []
    if not content:
        return {}
    try:
        return json.loads(content[0].get("text") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def _is_jsonrpc_line(line: str) -> bool:
    try:
        msg = json.loads(line)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(msg, dict) and msg.get("jsonrpc") == "2.0" and (
        "result" in msg or "error" in msg or "method" in msg)


# ── install tests ────────────────────────────────────────────────────────────


def _sandbox_home(tmp: Path) -> dict:
    env = dict(os.environ)
    env["USERPROFILE"] = str(tmp)   # Path.home() on Windows
    env["HOME"] = str(tmp)          # Path.home() on POSIX
    env["DSH_HOME"] = str(tmp / ".dsh")
    env["CODEX_HOME"] = str(tmp / ".codex")
    env["ZCODE_HOME"] = str(tmp / ".zcode")
    env.pop("CITLENS_CONFIG", None)
    return env


def _run_install(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(INSTALL), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=300)


def test_install() -> None:
    print("\n== installer (sandbox HOME) ==")
    tmp = Path(tempfile.mkdtemp(prefix="citlens-home-"))
    # seed the four config dialects with realistic pre-existing content
    (tmp / ".codex").mkdir(parents=True)
    (tmp / ".codex" / "config.toml").write_text(
        'model = "deepseek-v4-flash"\n\n[mcp_servers.other]\ncommand = "x"\nargs = []\n',
        encoding="utf-8")
    (tmp / ".claude.json").write_text(json.dumps(
        {"numStartups": 12, "mcpServers": {"web-reader": {"type": "http",
                                                          "url": "https://example.invalid/mcp"}}},
        ensure_ascii=False), encoding="utf-8")
    (tmp / ".zcode" / "cli").mkdir(parents=True)
    (tmp / ".zcode" / "cli" / "config.json").write_text(json.dumps(
        {"mcp": {"servers": {"firecrawl": {"type": "stdio", "command": "npx"}}}},
        ensure_ascii=False), encoding="utf-8")
    dsh_profile = tmp / ".dsh" / "profiles" / "desktop"
    dsh_profile.mkdir(parents=True)
    (dsh_profile / "package.json").write_text(json.dumps(
        {"name": "dsh-profile-desktop", "private": True,
         "dsh": {"profile": {"bundles": ["@deepseek-ai/dsh-base"]}}}, indent=2), encoding="utf-8")
    env = _sandbox_home(tmp)
    try:
        plan = _run_install(env, "plan")
        check("plan runs", plan.returncode == 0, plan.stderr[-400:])
        check("plan writes nothing", not (tmp / ".citlens" / "config.json").exists())
        check("plan covers the detected clients",
              all(a in plan.stdout for a in ("codex", "claude", "zcode", "dsh")))

        apply1 = _run_install(env, "apply", "--no-dsh-auto")
        check("apply runs", apply1.returncode == 0, apply1.stderr[-400:])
        cfg = json.loads((tmp / ".citlens" / "config.json").read_text(encoding="utf-8"))
        check("shared config written with absolute engine paths",
              Path(cfg["python"]).is_file() and Path(cfg["recall_driver"]).is_file())
        check("CLI shim written", (tmp / ".citlens" / "bin" / "citlens.cmd").is_file())

        toml = (tmp / ".codex" / "config.toml").read_text(encoding="utf-8")
        check("codex: pre-existing model line kept", 'model = "deepseek-v4-flash"' in toml)
        check("codex: pre-existing mcp server kept", "[mcp_servers.other]" in toml)
        check("codex: citlens mcp block added", "[mcp_servers.citlens]" in toml)
        check("codex: skill block added", "[[skills.config]]" in toml)
        check("codex: backup written", any(p.name.startswith("config.toml.citlens-backup-")
                                           for p in (tmp / ".codex").iterdir()))

        claude = json.loads((tmp / ".claude.json").read_text(encoding="utf-8"))
        check("claude: existing server kept", "web-reader" in claude["mcpServers"])
        check("claude: existing top-level keys kept", claude.get("numStartups") == 12)
        check("claude: citlens stdio entry added",
              claude["mcpServers"]["citlens"]["type"] == "stdio"
              and Path(claude["mcpServers"]["citlens"]["command"]).is_file())

        zcode = json.loads((tmp / ".zcode" / "cli" / "config.json").read_text(encoding="utf-8"))
        check("zcode: existing server kept", "firecrawl" in zcode["mcp"]["servers"])
        check("zcode: citlens entry added",
              zcode["mcp"]["servers"]["citlens"]["type"] == "stdio")

        bundle = tmp / ".citlens" / "dsh-bundle"
        check("dsh: bundle package.json written", (bundle / "package.json").is_file())
        patch = (bundle / "cordis.patch.yml").read_text(encoding="utf-8")
        check("dsh: patch inserts the mcp-client row",
              "@deepseek-ai/dsh-mcp-client" in patch and "serverName: citlens" in patch
              and "transport: stdio" in patch)

        for target in (tmp / ".agents" / "skills" / "citlens",
                       tmp / ".claude" / "skills" / "citlens",
                       tmp / ".zcode" / "skills" / "citlens"):
            check(f"skill distributed -> {target.relative_to(tmp)}",
                  (target / "SKILL.md").is_file()
                  and "name: citlens" in (target / "SKILL.md").read_text(encoding="utf-8"))

        # idempotency: a second apply must change nothing and add no new backup
        before = {p: len(list(p.parent.glob(p.name + ".citlens-backup-*")))
                  for p in (tmp / ".codex" / "config.toml",
                            tmp / ".claude.json",
                            tmp / ".zcode" / "cli" / "config.json")}
        apply2 = _run_install(env, "apply", "--no-dsh-auto")
        after = {p: len(list(p.parent.glob(p.name + ".citlens-backup-*"))) for p in before}
        check("second apply is idempotent (no new backups)", before == after, f"{before} -> {after}")
        check("second apply says nothing to do", "Nothing to do" in apply2.stdout,
              apply2.stdout[-900:])

        status = _run_install(env, "status")
        check("status reports the three config-file clients as registered",
              status.stdout.count("registered") >= 3, status.stdout[-800:])
        check("status reports the dsh profile and the skill hub",
              "profile 'desktop'" in status.stdout and "skill hub" in status.stdout)
        check("status reports per-agent skill dirs", "Per-agent skill dirs:" in status.stdout)

        un = _run_install(env, "uninstall", "--no-dsh-auto")
        check("uninstall runs", un.returncode == 0, un.stderr[-400:])
        toml2 = (tmp / ".codex" / "config.toml").read_text(encoding="utf-8")
        check("uninstall: codex block gone, rest intact",
              "[mcp_servers.citlens]" not in toml2 and "[mcp_servers.other]" in toml2
              and 'model = "deepseek-v4-flash"' in toml2)
        claude2 = json.loads((tmp / ".claude.json").read_text(encoding="utf-8"))
        check("uninstall: claude entry gone, rest intact",
              "citlens" not in claude2["mcpServers"] and "web-reader" in claude2["mcpServers"])
        zcode2 = json.loads((tmp / ".zcode" / "cli" / "config.json").read_text(encoding="utf-8"))
        check("uninstall: zcode entry gone, rest intact",
              "citlens" not in zcode2["mcp"]["servers"] and "firecrawl" in zcode2["mcp"]["servers"])
        check("uninstall: skill dirs removed",
              not (tmp / ".agents" / "skills" / "citlens").exists())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cli_fallback() -> None:
    print("\n== CLI fallback (no MCP) ==")
    cfg_path, tmp = _temp_config()
    try:
        r = subprocess.run([sys.executable, str(SERVER), "cli", "--config", str(cfg_path), "doctor"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=180)
        ok = r.returncode == 0
        payload = {}
        if ok:
            try:
                payload = json.loads(r.stdout)
            except Exception:  # noqa: BLE001
                ok = False
        check("cli doctor prints one JSON document", ok and payload.get("ok") is True)
        bad = subprocess.run([sys.executable, str(SERVER), "cli", "--config", str(cfg_path)],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=60)
        check("cli without a tool prints usage JSON", bad.returncode == 2
              and "tools" in bad.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selftest() -> None:
    print("\n== selftest ==")
    r = subprocess.run([sys.executable, str(SERVER), "selftest"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    check("selftest exits 0 on a healthy install", r.returncode == 0, r.stdout[-600:])
    check("selftest lists the seven tools", r.stdout.count(",") >= 6 and "search" in r.stdout)


def test_registry() -> None:
    """The point of the registry: a new client is data, not code."""
    print("\n== agent registry (data-driven extension) ==")
    tmp = Path(tempfile.mkdtemp(prefix="citlens-reg-"))
    (tmp / ".codex").mkdir(parents=True)
    (tmp / ".codex" / "config.toml").write_text('model = "x"\n', encoding="utf-8")
    env = _sandbox_home(tmp)
    try:
        listing = _run_install(env, "list")
        check("list runs", listing.returncode == 0, listing.stderr[-300:])
        check("list shows the mainstream clients",
              all(n in listing.stdout for n in ("cursor", "windsurf", "gemini", "opencode",
                                                "cline", "zed", "aider")))
        check("list marks dialects and confidence", "json-map" in listing.stdout
              and "verified" in listing.stdout)

        snip = _run_install(env, "snippet", "zed")
        check("snippet prints a paste-ready config for a manual client",
              snip.returncode == 0 and "mcpServers" in snip.stdout
              and "context_servers" in snip.stdout)

        snip2 = _run_install(env, "snippet", "aider")
        check("snippet explains the CLI-only integration",
              snip2.returncode == 0 and "CLI shim" not in snip2.stdout.lower()
              and "citlens.cmd" in snip2.stdout)

        # drop a brand-new client into ~/.citlens/agents.d and install it
        (tmp / ".citlens" / "agents.d").mkdir(parents=True)
        (tmp / ".citlens" / "agents.d" / "custom.json").write_text(json.dumps({
            "agents": {"myagent": {
                "label": "My Agent",
                "dialect": "json-map",
                "file": "~/.myagent/mcp.json",
                "container": ["mcpServers"],
                "env": True,
            }}
        }), encoding="utf-8")
        listing2 = _run_install(env, "list")
        check("a user-registered client appears without code changes",
              "myagent" in listing2.stdout)

        applied = _run_install(env, "apply", "--only", "myagent", "--no-dsh-auto")
        target = tmp / ".myagent" / "mcp.json"
        ok = applied.returncode == 0 and target.is_file()
        entry = {}
        if ok:
            entry = json.loads(target.read_text(encoding="utf-8"))["mcpServers"]["citlens"]
        check("apply writes the stdio entry for a custom client",
              ok and entry.get("type") == "stdio" and Path(entry.get("command", "")).is_file())
        # --only still installs the shared prelude (config + shim + hub): that is
        # shared infrastructure, not per-agent state. It must not touch other agents.
        check("--only still writes the shared prelude", (tmp / ".citlens" / "config.json").is_file())
        codex_after = (tmp / ".codex" / "config.toml").read_text(encoding="utf-8")
        check("--only leaves other agents' configs untouched",
              "[mcp_servers.citlens]" not in codex_after)

        removed = _run_install(env, "uninstall", "--only", "myagent", "--no-dsh-auto")
        data = json.loads(target.read_text(encoding="utf-8"))
        check("uninstall removes only the citlens key",
              removed.returncode == 0 and "citlens" not in data.get("mcpServers", {}))

        # a config file we cannot parse must never be "repaired" into emptiness
        bad = tmp / ".badagent" / "mcp.json"
        bad.parent.mkdir(parents=True)
        bad.write_text('{\n  // json5 comment\n  "mcpServers": {}\n}\n', encoding="utf-8")
        (tmp / ".citlens" / "agents.d" / "bad.json").write_text(json.dumps({
            "agents": {"badagent": {"label": "Bad", "dialect": "json-map",
                                    "file": "~/.badagent/mcp.json",
                                    "container": ["mcpServers"]}}
        }), encoding="utf-8")
        bad_run = _run_install(env, "apply", "--only", "badagent", "--no-dsh-auto")
        check("unparsable client config is left untouched, not overwritten",
              bad_run.returncode == 0 and bad.read_text(encoding="utf-8").startswith("{\n  //"))

        # --all creates configs for absent clients instead of skipping them
        all_run = _run_install(env, "apply", "--all", "--no-dsh-auto")
        check("apply --all creates configs for absent clients",
              all_run.returncode == 0 and (tmp / ".cursor" / "mcp.json").is_file()
              and (tmp / ".config" / "opencode" / "opencode.json").is_file())
        cursor = json.loads((tmp / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
        check("cursor entry uses the shared server argv",
              cursor["mcpServers"]["citlens"]["args"][:1] == [str(SERVER)])
        opencode = json.loads((tmp / ".config" / "opencode" / "opencode.json").read_text(encoding="utf-8"))
        check("opencode uses its own dialect (type=local, command list)",
              opencode["mcp"]["citlens"]["type"] == "local"
              and isinstance(opencode["mcp"]["citlens"]["command"], list))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print(f"server: {SERVER}")
    test_selftest()
    test_protocol()
    test_install()
    test_registry()
    test_cli_fallback()
    test_engine_override_and_driver_scoping()
    test_init_wizard()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 0 if not FAIL else 1


def test_engine_override_and_driver_scoping() -> None:
    print("\n== installer engine/driver scoping (sandbox HOME) ==")
    tmp = Path(tempfile.mkdtemp(prefix="citlens-scope-"))
    # an old, coherent install living elsewhere with its own drivers
    old_engine = tmp / "old-engine"
    (old_engine / "tools").mkdir(parents=True)
    (old_engine / "tools" / "citlens_recall.py").write_text("# old", encoding="utf-8")
    (old_engine / "tools" / "citens_review.py").write_text("# old", encoding="utf-8")
    cc = tmp / ".citlens"
    cc.mkdir(parents=True)
    (cc / "config.json").write_text(json.dumps({
        "home": str(tmp / "old-home"),
        "python": sys.executable,
        "engine": str(old_engine),
        "recall_driver": str(old_engine / "tools" / "citlens_recall.py"),
        "review_driver": str(old_engine / "tools" / "citens_review.py"),
        "jobs_dir": str(cc / "jobs"),
    }, ensure_ascii=False), encoding="utf-8")
    # one detectable client, else apply exits "no agent detected"
    (tmp / ".zcode" / "cli").mkdir(parents=True)
    (tmp / ".zcode" / "cli" / "config.json").write_text(
        '{"mcp": {"servers": {}}}', encoding="utf-8")
    env = _sandbox_home(tmp)
    try:
        # 1) no --engine: a coherent old install is reused verbatim
        r1 = _run_install(env, "apply", "--no-dsh-auto")
        check("plain apply on coherent install runs",
              r1.returncode == 0, r1.stdout[-300:])
        cfg1 = json.loads((cc / "config.json").read_text(encoding="utf-8"))
        check("coherent old install keeps its engine",
              Path(cfg1["engine"]).resolve() == old_engine.resolve())
        check("coherent old install keeps its drivers",
              Path(cfg1["recall_driver"]).resolve()
              == (old_engine / "tools" / "citlens_recall.py").resolve())

        # 2) explicit --engine <this repo>: engine AND drivers must switch
        # to this tree (measured bug 2026-10-18: --engine was silently
        # ignored against an existing config and old drivers survived)
        this_repo = PORTABLE.parent
        r2 = _run_install(env, "apply", "--no-dsh-auto",
                          "--engine", str(this_repo),
                          "--python", sys.executable, "--home", str(tmp))
        check("override apply runs", r2.returncode == 0, r2.stdout[-300:])
        cfg2 = json.loads((cc / "config.json").read_text(encoding="utf-8"))
        check("--engine wins over existing config",
              Path(cfg2["engine"]).resolve() == this_repo.resolve(),
              f"got {cfg2['engine']}")
        check("engine switch carries this tree's drivers",
              Path(cfg2["recall_driver"]).resolve()
              == (PORTABLE / "drivers" / "citlens_recall.py").resolve(),
              f"got {cfg2['recall_driver']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_init_wizard() -> None:
    print("\n== init wizard (tmp home) ==")
    tmp = Path(tempfile.mkdtemp(prefix="citlens-init-"))
    env = _sandbox_home(tmp)
    try:
        # non-interactive on an empty home: required keys missing -> rc 1
        r1 = _run_install(env, "init", "--home", str(tmp), "--non-interactive")
        check("init reports missing required keys",
              r1.returncode == 1 and "LLM_API_KEY" in r1.stdout)
        check("init without sets writes nothing",
              not (tmp / ".env").exists())

        # scripted --set writes keys, never clobbers existing values
        r2 = _run_install(env, "init", "--home", str(tmp), "--non-interactive",
                          "--set", "LLM_API_KEY=sk-test", "--set", "LLM_MODEL=m1",
                          "--set", "LLM_API_BASE=https://api.example.com")
        check("init --set satisfies required keys", r2.returncode == 0, r2.stdout[-300:])
        envp = (tmp / ".env").read_text(encoding="utf-8")
        check("env file holds the set keys", "LLM_API_KEY=sk-test" in envp
              and "LLM_MODEL=m1" in envp)
        r3 = _run_install(env, "init", "--home", str(tmp), "--non-interactive",
                          "--set", "SERPER_API_KEY=sp-1")
        envp2 = (tmp / ".env").read_text(encoding="utf-8")
        check("second init keeps earlier keys", "LLM_API_KEY=sk-test" in envp2
              and "SERPER_API_KEY=sp-1" in envp2)
        check("second init still ok with required present", r3.returncode == 0)

        # upsert replaces in place, no duplicates
        r4 = _run_install(env, "init", "--home", str(tmp), "--non-interactive",
                          "--set", "LLM_MODEL=m2")
        envp3 = (tmp / ".env").read_text(encoding="utf-8")
        check("upsert replaces value in place",
              "LLM_MODEL=m2" in envp3 and "LLM_MODEL=m1" not in envp3
              and envp3.count("LLM_MODEL=") == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
