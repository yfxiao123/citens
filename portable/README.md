# CiteLens portable 层 — 一套引擎，任意 agent

> 安装与配置的**完整流程**在仓库根的 [INSTALL.md](../INSTALL.md)；本页是这一层的架构与细节。

这层让"搜文献 + 管理文献 + 写综述"与 agent 无关：装进哪个客户端，哪个客户端就获得同一引擎。检索方法本体留在仓库根的 `citens` 引擎里（在那里被 bench 和测试覆盖），**本目录不含任何检索逻辑**，只做暴露：

```
引擎 citens（仓库根）
        │  以自己的解释器派生
        ▼
portable/citlens_mcp.py        零依赖 MCP stdio server
   tools: search · pool · plan · questions · review · status · doctor
        │
        ├── stdio MCP ──► claude · codex · dsh · zcode · gemini · qwen · opencode · cursor · windsurf · cline …
        ├── CLI ────────► 任意 shell（aider、脚本、CI、人类）
        └── skill ──────► 各 agent 的 skill 目录
```

## 为什么可移植

- **一个 server，N 个客户端。** 主流 agent 都说 MCP-over-stdio，且注册结构大同小异。`agents.json` 是声明式注册表（方言 + 路径 + 检测 + 置信度）；`install.py` 只有四种写入器。**加一个客户端 = 加一条 JSON，不是加代码。**
- **一份共享状态。** `~/.citlens/config.json` + 单一数据家：所有 agent 读写同一个文献池——搜过的 agent 可以把池子交给写综述的 agent。任务是分离进程，启动它的会话关掉后任务继续跑。
- **一张方法卡。** `skill/citlens/SKILL.md` 把纪律带进从不读本页的客户端：用工具检索而非自造、只报工具给的真实数字、绝不把失败源报成零命中、综述只在明确要求时写。

## 命令一览

```bash
.venv/Scripts/python portable/install.py init --home .     # .env 配置向导（交互/非交互/--set）
.venv/Scripts/python portable/install.py list              # 已知客户端 + 本机安装状态
.venv/Scripts/python portable/install.py status            # 当前注册情况
.venv/Scripts/python portable/install.py plan              # 干跑预览
.venv/Scripts/python portable/install.py apply             # 写入（幂等，全部先备份）
.venv/Scripts/python portable/install.py uninstall         # 只移除本脚本加的东西
```

## 工具语义

| 工具 | 作用 |
|---|---|
| `doctor` | 引擎/驱动/数据家/`.env` 键名/池规模/最近任务 |
| `plan` | 概念块 + 召回腿将执行的查询；不检索 |
| `search` | 九召回家族 → RRF 家族配额并池 → 两级盲评 LLM 级联；产出 `pool.jsonl` + `search_report.md` + `exclusions.jsonl` |
| `pool` | 读取/过滤磁盘上的池（引用数/年份/综述/DOI）；不发起新检索 |
| `questions` | 综述引擎自带的澄清问题（语言/主题/篇数） |
| `review` | **opt-in** 的引用有据综述写作（必须 `user_asked_for_review: true`） |
| `status` | 轮询运行中任务：状态/耗时/日志尾/产物/统计 |

无 MCP 的客户端走同一套工具的 CLI 兜底：`~/.citlens/bin/citlens.cmd doctor` 等。

## 客户端方言

| 方言 | 写入形状 | 客户端 |
|---|---|---|
| `json-map` | `file[container]["citlens"] = {type,command,args,env}` | claude, zcode, gemini, qwen, cursor, windsurf, cline, roo, kilo |
| `json-opencode` | `file.mcp["citlens"] = {type:"local", command:[...]}` | opencode |
| `toml-block` | `[mcp_servers.citlens]`（带标记注释） | codex |
| `dsh-bundle` | bundle patch 行 + `dsh plugin add` | dsh |
| `snippet` / `cli-only` | 手贴片段 / 仅 CLI | zed, goose, continue, amp, crush, kimi-cli, warp, copilot / aider |

置信度诚实标注：`verified` = 开发期真实客户端接受过；`likely` = 形状有文档依据未实测；`template` = 形状随版本变，只打印片段不盲写。

自定义客户端：往 `~/.citlens/agents.d/` 丢一个 JSON 即可（同 schema，你的条目优先）。

## 测试

```bash
.venv/Scripts/python portable/tests/test_portable.py     # 85 项离线检查
```

覆盖：MCP 握手（协议协商/工具列表/错误路径/stdout 纯净/干净退出）、安装器（plan/apply/幂等/uninstall、合并保真、备份、数据驱动扩展、不可解析配置拒写）、`--engine` 覆盖与驱动作用域、init 向导（不覆盖/upsert/退出码）、CLI 兜底。

## 安全

- 客户端配置里不写任何密钥；key 只留在引擎 `.env`，`doctor` 只报键名。
- 不可解析的客户端配置（带注释的 JSON5、写了一半的）绝不"修复"成空——跳过并提示手贴片段。
- skill 目录只在 `SKILL.md` 声明 `name: citlens` 时才会被替换。

端到端试跑清单见 [TESTING.md](TESTING.md)。
