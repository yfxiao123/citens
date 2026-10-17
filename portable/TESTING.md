# TESTING — trial the portable CiteLens tool from a clean clone

Goal: prove that a **fresh clone of this branch** can find literature and (only
if asked) write a review, on a machine that already has other agents — and that a
different agent (ZCode here) can consume it without any per-agent code.

Everything below is inside the clone. Nothing outside `~/.citlens` is touched
until the final registration step, and that step is reversible.

## 1. Get a working engine (once)

```bash
git clone -b test/portable-mcp https://github.com/yfxiao123/citens.git citens-portable-test
cd citens-portable-test

# Windows paths shown; POSIX: .venv/bin/python
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env
```

Put the model endpoint into `.env` (this is the only secret; it never leaves the
engine's `.env`, and the installer never copies it into an agent config):

```
LLM_API_BASE=https://api.deepseek.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-chat
```

Optional but recommended for breadth: `SEMANTIC_SCHOLAR_API_KEY`,
`OPENALEX_API_KEY`, `SERPER_API_KEY`. Without them the web-bridge / S2 /
polite-pool legs degrade (they are reported as failed, never as zero hits).

## 2. Install the tool layer into the clone

```bash
.venv\Scripts\python.exe portable\install.py list
.venv\Scripts\python.exe portable\install.py plan  --engine . --python .venv\Scripts\python.exe --home .
.venv\Scripts\python.exe portable\install.py apply --engine . --python .venv\Scripts\python.exe --home .
```

`--home .` keeps the pool, runs and `.env` inside the clone, so this trial writes
nothing into the development data directory. `apply` is idempotent: run it twice,
the second run prints `Nothing to do`.

Verify without any agent:

```bash
.venv\Scripts\python.exe portable\citlens_mcp.py selftest
.venv\Scripts\python.exe portable\citlens_mcp.py cli --config "%USERPROFILE%\.citlens\config.json" doctor
```

`doctor` must say `ok: true`, with `home` pointing at the clone.

## 3. Point ZCode at the clone (the actual portability test)

```bash
.venv\Scripts\python.exe portable\install.py apply --only zcode
```

This rewrites **only** `mcp.servers.citlens` in `~/.zcode/cli/config.json`
(backup: `config.json.citlens-backup-<timestamp>`). Restart ZCode so it reads the
file, then ask it:

> 用 citlens 的 doctor 看一下状态。

It should print the clone's paths. Then:

> 用 citlens 搜一下 "semantic id for generative recommendation"，target 60，等它跑完，
> 把池子大小、保留篇数、逐源 ok/empty/failed 念出来。

Then, from a **different** agent (or the CLI), read the same pool to prove the
shared-state claim:

```bash
.venv\Scripts\python.exe portable\citlens_mcp.py cli --config "%USERPROFILE%\.citlens\config.json" pool --topic "semantic id for generative recommendation" --min-citations 50
```

A review is opt-in only. If you want to test that half:

> 写一篇中文综述，主题是 …，核心 12 篇。

The agent must first call `questions` and ask you the output language, the themes
and the core-paper count — it must not invent them. The run then reports its own
honesty block (`citation_precision`, `unverifiable_rate`,
`grounding_with_fulltext`); if `with_fulltext` is 0 the output must be labelled
abstract-only.

## 4. Roll back

```bash
.venv\Scripts\python.exe portable\install.py uninstall --only zcode
```

That removes only the `citlens` key. If you had a previous registration from the
development tree, re-run its `install.py apply --only zcode` to restore it.

## What "pass" looks like

- [ ] `install.py list` shows the clone-local engine and `detected` clients
- [ ] `selftest` and `cli doctor` return `ok: true` with the clone's `home`
- [ ] `apply` twice → second run `Nothing to do` (idempotent)
- [ ] ZCode lists a `citlens` MCP server and `doctor` answers
- [ ] `search` finishes and a second agent reads the same pool by topic
- [ ] `review` refuses without `user_asked_for_review`, and asks the mandatory
      questions when you do ask
- [ ] `uninstall --only zcode` restores the config (compare with the backup)

## Known limits of this branch

- The engine still needs its own Python environment (`pip install -e .`); the
  portable layer is the *interface*, not a replacement for the engine.
- Clients marked `template` in `agents.json` (Zed, Goose, Continue, Amp, Crush,
  Kimi CLI, Warp, Copilot) are not written automatically — `install.py snippet
  <name>` prints paste-ready config.
- `install.py apply` installs for **detected** clients only; `--all` also creates
  configs for absent ones.
