# 安装与配置全流程

目标：把 CiteLens 装进你**正在用的 agent**（Claude Code / Codex / ZCode / Cursor / Gemini CLI 等 21 种），让它获得"搜文献 → 管理文献 → 写综述"三个工具。整个过程约 5 分钟，每一步都可回滚。

- 配置项全量参考：[docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- 只想从源码/桌面版使用（不接 agent）：见 [README.md](README.md#其他使用方式)

---

## 0. 前置条件

| 需要 | 说明 |
|---|---|
| Python ≥ 3.10 | `python --version` 确认 |
| git | 用来 clone 本仓库 |
| 一个 LLM API key | OpenAI 兼容任意厂商（DeepSeek / 智谱 / OpenAI / 本地 vLLM…），唯一硬要求 |
| 网络代理（可选） | 若 `git clone` 直连 GitHub 失败：`git -c http.proxy= -c https.proxy= clone ...`；若走代理则确保代理进程在运行 |

> 全部检索源 API（OpenAlex / Semantic Scholar / Crossref / arXiv）无需 key 即可用；配 key 是为了更高配额（见第 2 步）。

## 1. 克隆并安装引擎

```bash
git clone https://github.com/yfxao123/citens.git && cd citens
python -m venv .venv
.venv/Scripts/python -m pip install -e .        # macOS/Linux: .venv/bin/python -m pip install -e .
.venv/Scripts/python -c "import citens; print('engine OK')"
```

引擎与安装器在同一个仓库里——clone 即得全部。

## 2. 配置（init 向导）

```bash
.venv/Scripts/python portable/install.py init --home .
```

- 向导逐项提示：**必填** `LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL`；**推荐** `SERPER_API_KEY`（web 桥接家族，Google SERP 免费 2500 次）、`SEMANTIC_SCHOLAR_API_KEY`（免费提速）、`OPENALEX_API_KEY` + `OPENALEX_EMAIL`（免费 1 万积分/天）。
- 回车跳过任何项；**绝不覆盖已有值**；写入 clone 根目录的 `.env`。
- 脚本/CI 场景：`init --home . --non-interactive --set LLM_API_KEY=sk-xxx --set LLM_MODEL=m1`（必填缺失时退出码 1）。

也可以手工：`cp .env.example .env` 后编辑。全量键语义见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

## 3. 安装进 agent

```bash
# 预览会写什么（dry-run，不写任何文件）
.venv/Scripts/python portable/install.py plan --engine . --python .venv/Scripts/python.exe --home .

# 执行（--only 限定客户端；省略则装到本机检测到的全部 agent）
.venv/Scripts/python portable/install.py apply --engine . --python .venv/Scripts/python.exe --home . --only zcode
```

做的事（全部先备份 `.citlens-backup-<ts>` 再写）：

1. 写共享配置 `~/.citlens/config.json`（home/python/engine/驱动五路径 + 家族与默认参数）；
2. 在目标 agent 的配置里注册 `citlens` stdio MCP 服务端；
3. 把方法卡 skill 装进 agent 的 skill 目录；
4. 生成 CLI 兜底 `~/.citlens/bin/citlens.cmd`。

`--only` 换成你的 agent 名（`install.py list` 看全部 21 个）。**template 级客户端**（copilot 等）用 `install.py snippet <name>` 拿到手贴片段。

## 4. 验收

重启你的 agent（MCP 注册只在启动时读取），然后说：

> 用 citlens 的 doctor 检查一下环境

doctor 会核对：python/引擎版本/驱动路径/`.env` 必填键（缺推荐键会明确说哪些家族降级）/最近任务。全绿后试一发真检索：

> 用 citlens 搜一下 "diffusion model recommendation" 相关文献，目标 40 篇

预期：2–6 分钟后返回池统计（家族席位、各源 ok/empty/failed）、`pool.jsonl` 与 PRISMA 排除清单路径。写作链是 opt-in 的：让 agent 先 `questions`（它会问语言/主题/篇数）再 `review`。

## 5. 数据都落在哪

数据家（`--home`，默认 clone 根）内自包含，删目录即卸载干净：

```
<home>/
├── .env               # 你的配置（init 写入）
├── citlens-runs/      # 每次 search 的运行目录（pool.jsonl / search_report.md / exclusions.jsonl）
├── runs/              # 每次 review 的运行目录（review.md / references.bib|.ris / verification.json / provenance.json）
├── data/litdb/        # 跨主题累积的文献池（检索管理 的落库）
└── .cache/            # 检索缓存 + openalex_usage.json（积分护栏记账）
```

任务队列在 `~/.citlens/jobs/`（检索/写作是分离进程，agent 关掉任务也继续跑）。

## 6. 卸载与回滚

```bash
.venv/Scripts/python portable/install.py uninstall          # 移除安装器写入的一切（注册/skill/共享配置）
```

任何被覆盖过的文件旁边都有 `.citlens-backup-<时间戳>` 副本，可手工恢复。数据目录与 `~/.citlens/jobs` 不受卸载影响。

## 7. 故障排查

| 症状 | 原因与处理 |
|---|---|
| `git clone` 连接被拒/重置 | 全局代理指向未运行的进程。直连：`git -c http.proxy= -c https.proxy= clone ...`；或启动代理软件 |
| doctor 报 `SERPER_API_KEY` 缺失 | web 桥接家族降级（其余家族不受影响）；想启用就补 key 后重跑 init |
| 检索里 openalex 源 failed/empty | 检查 `OPENALEX_API_KEY`；免费积分按 UTC 日重置（每查询 10 积分，护栏记账见 `.cache/openalex_usage.json`，上限 `OPENALEX_DAILY_BUDGET`） |
| semantic_scholar 间歇 empty | S2 免费档限流严格（约 1 req/s），进程内已做节流+重试；持续失败通常几分钟后自愈 |
| review 引用精度明显偏低 | 大概率是摘要级 grounding（拿不到 OA 全文）。看 honesty 的 `grounding_with_fulltext`，为 0 时综述必须标注 ABSTRACT-ONLY |
| agent 看不到 citlens 工具 | MCP 注册要**重启 agent**；用 doctor 的 CLI 兜底验证：`~/.citlens/bin/citlens.cmd doctor` |

## 8. 已装过一版，想切到新 clone？

```bash
# 在新 clone 里，显式指定路径（--engine 现在会真正生效，驱动随之切换）
.venv/Scripts/python portable/install.py apply --engine . --python .venv/Scripts/python.exe --home . --only zcode
```

不指定 `--engine` 时，安装器会**沿用现有配置里的引擎与驱动**（已配置机器的平滑行为）；显式指定则全套切换到新 clone。
