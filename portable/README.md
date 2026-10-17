# CiteLens portable layer — one engine, any agent

This branch (`test/portable-mcp`) adds a client-agnostic way to use CiteLens:
**find literature with high recall and write a citation-grounded survey** — from
any agent, not just one harness.

The retrieval method stays where it is measured and tested: inside the `citens`
engine at the root of this repository. This directory adds **no retrieval logic**.
It only exposes the engine to whatever client you use:

```
engine `citens`  (this repo)
        │  spawned in its own interpreter
        ▼
portable/citlens_mcp.py        zero-dependency MCP stdio server
   tools: search · pool · plan · questions · review · status · doctor
        │
        ├── stdio MCP ──► claude · codex · dsh · zcode · gemini · qwen · opencode · cursor · windsurf · cline …
        ├── CLI ────────► any shell (aider, scripts, CI, humans)
        └── skill ──────► ~/.agents/skills/citlens   (DSH, Codex, `npx skills` ecosystem)
```

## Why this is portable

- **One server, N clients.** Every mainstream agent speaks MCP over stdio, and
  almost all of them store servers in the same shape. `agents.json` is a
  declarative registry (dialect + path + detection); `install.py` has only four
  writers. **Adding a client is adding a JSON entry — not code.**
- **One shared state.** `~/.citlens/config.json` and one data home mean every
  agent reads and writes the *same* literature pool, so an agent that searched
  can hand its pool to an agent that writes. Jobs are detached processes: a long
  search survives the session that started it.
- **One method card.** `skill/citlens/SKILL.md` carries the discipline (use the
  tool, report real numbers, never present a failed source as zero hits, only
  write a review on an explicit request) to clients that never read this README.

## Quickstart

```bash
git clone -b test/portable-mcp https://github.com/yfxiao123/citens.git
cd citens
python -m venv .venv
# Windows: .venv\Scripts\python.exe   POSIX: .venv/bin/python
.venv/Scripts/python.exe -m pip install -e .

cp .env.example .env        # then fill LLM_API_KEY / LLM_API_BASE / LLM_MODEL

.venv/Scripts/python.exe portable/install.py list      # known clients + what is installed
.venv/Scripts/python.exe portable/install.py status    # what is registered right now
.venv/Scripts/python.exe portable/install.py plan      # dry run
.venv/Scripts/python.exe portable/install.py apply     # register detected clients (idempotent)
```

`apply` writes `~/.citlens/config.json`, a CLI shim at `~/.citlens/bin/citlens.cmd`,
the skill card, and one registration per detected client. Every file it touches is
backed up first; `install.py uninstall` removes only what it added.

If the engine is not discoverable from the checkout (e.g. you keep data elsewhere),
say so explicitly — the paths are persisted in `~/.citlens/config.json`:

```bash
.venv/Scripts/python.exe portable/install.py apply \
  --engine . --python .venv/Scripts/python.exe --home .
```

## What the tools do

| tool | purpose |
|---|---|
| `doctor` | engine / drivers / data home / `.env` key names / pool count / jobs |
| `plan` | concept blocks + the English queries the recall leg would run; no retrieval |
| `search` | nine recall families → RRF under per-family seat quotas → two-stage blind LLM cascade; writes `pool.jsonl` + `search_report.md` + `exclusions.jsonl` |
| `pool` | read/filter the pool on disk (citations, years, reviews, DOIs); no new retrieval |
| `questions` | the survey engine's own clarifying questions for a topic |
| `review` | **opt-in** citation-grounded survey writer (`user_asked_for_review: true` required) |
| `status` | poll a running job: state, elapsed, log tail, artifacts, stats |

Same tools via CLI when the client has no MCP:

```bash
~/.citlens/bin/citlens.cmd doctor
~/.citlens/bin/citlens.cmd search --topic "semantic id recommendation" --target 80
~/.citlens/bin/citlens.cmd pool --topic "semantic id recommendation" --min-citations 100
```

## Supported clients

`install.py list` prints the live picture. Dialects:

| dialect | shape | clients |
|---|---|---|
| `json-map` | `file[container]["citlens"] = {type,command,args,env}` | claude, zcode, gemini, qwen, cursor, windsurf, cline, roo, kilo |
| `json-opencode` | `file.mcp["citlens"] = {type:"local", command:[...]}` | opencode |
| `toml-block` | `[mcp_servers.citlens]` (+ marker comments) | codex |
| `dsh-bundle` | bundle patch row + `dsh plugin add` | dsh |
| `snippet` / `cli-only` | paste-ready snippet / CLI only | zed, goose, continue, amp, crush, kimi, warp, copilot / aider |

Confidence is honest: `verified` = a real client accepted it during development,
`likely` = documented shape not exercised, `template` = version-dependent shape,
so the installer only prints a snippet instead of writing blindly.

Add your own client by dropping a JSON file into `~/.citlens/agents.d/`:

```json
{"agents": {"myagent": {"label": "My Agent", "dialect": "json-map",
  "file": "~/.myagent/mcp.json", "container": ["mcpServers"], "env": true,
  "detect": ["~/.myagent"], "skillDirs": ["~/.myagent/skills"]}}}
```

## Tests

```bash
.venv/Scripts/python.exe portable/tests/test_portable.py     # 72 checks, offline
```

Covers the MCP handshake (protocol negotiation, tool list, error paths, stdout
purity, clean exit), the installer (plan/apply/idempotency/uninstall, merge
fidelity, backups, data-driven client extension, refuse-to-clobber on unparsable
JSON) and the CLI fallback.

## Safety

- No secrets are written into client configs; keys stay in the engine's `.env`,
  and `doctor` reports key *names* only.
- Unparsable client configs (JSON5 with comments, half-written) are never
  "repaired" into emptiness — the installer skips them and tells you to paste a
  snippet.
- A skill directory is only replaced when its `SKILL.md` says `name: citlens`.

See `TESTING.md` for the end-to-end trial checklist.
