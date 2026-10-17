# CiteLens (citens) — 搜文献 · 管理文献 · 写综述，装进你正在用的 agent

> **English quick start** — CiteLens is an open-source literature engine: high-recall
> search (nine recall families), a verifiable writing pipeline (every claim
> citation-checked; canary + defense passes; honesty numbers reported), and a
> growing literature pool you own. Install it into whatever agent you already use:
>
> ```bash
> git clone https://github.com/yfxao123/citens.git && cd citens
> python -m venv .venv && .venv/Scripts/python -m pip install -e .   # macOS/Linux: .venv/bin/python
> .venv/Scripts/python portable/install.py init --home .             # configure API keys
> .venv/Scripts/python portable/install.py apply --engine . --python .venv/Scripts/python.exe --home . --only claude
> ```
> Then restart your agent and ask it to run the `citlens` `doctor` tool.
> 21 agent clients supported (matrix below). Full guide: [INSTALL.md](INSTALL.md) ·
> Config reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md)

---

## 这个仓库是什么

一个自托管的文献工作引擎，宗旨是三件事，**与具体 agent 无关**——装进哪个 agent，哪个 agent 就获得这套能力：

| 能力 | 怎么实现 | 从哪里用 |
|---|---|---|
| **搜集文献** | 九家族召回：planner 关键词、问题原话、HyDE 假想标题、PRF 挖词、OpenAlex 标题索引、S2 推荐、引文扩展、中英双语 web 桥接、综述种子——RRF 家族配额并池后过两级 LLM 级联精排 | agent 里的 `search` 工具；或 CLI `citens run` |
| **管理文献** | 每次检索落库 `data/litdb/`（跨主题累积、键级去重）；`pool` 工具按关键词/年份过滤与统计；写综述时导出 `references.bib` / `.ris` / `provenance.json` | agent 里的 `pool` 工具；或 CLI `citens collect` 建领域池 |
| **写综述** | 带验证链的写作：每条论断核对到原文（四档判定），蜜罐检测抓无依据论断，辩护律师反向复核，缺陷论断改写复验——最后诚实报告引用精度 | agent 里先 `questions` 后 `review`（写前必问语言/主题/篇数） |

所有检索与写作产物落在你自己的数据目录（`runs/`、`data/`、`.cache/`），不经过任何第三方服务端。绝大多数设计决策都有 bench 或实测数据背书，完整尸检记录在 [docs/bench-postmortem.md](docs/bench-postmortem.md)。

## 三分钟装进你的 agent

```bash
git clone https://github.com/yfxao123/citens.git && cd citens
python -m venv .venv
.venv/Scripts/python -m pip install -e .        # macOS/Linux 路径为 .venv/bin/python

# 1) 配置：向导提示填 LLM key（唯一硬要求）和推荐的检索源 key，绝不覆盖已有值
.venv/Scripts/python portable/install.py init --home .

# 2) 安装：先 plan 预览，再 apply 写入（每一步都有备份，可 uninstall）
.venv/Scripts/python portable/install.py plan --engine . --python .venv/Scripts/python.exe --home . --only zcode
.venv/Scripts/python portable/install.py apply --engine . --python .venv/Scripts/python.exe --home . --only zcode
```

重启你的 agent，说一句"用 citlens 的 doctor 检查一下"，然后就可以自然语言使唤它：
"帮我搜一下世界模型强化学习的文献" → "从池子里筛 2024 年以后的" → "写一篇中文综述，12 篇精读"。

### 支持矩阵

`verified` = 在真实客户端上端到端验证过；`likely` = 注册格式已验证、客户端未实测；`template` = 给出手动粘贴片段。

| 状态 | 客户端 |
|---|---|
| ✅ verified | claude · codex · zcode · aider · dsh |
| 🔶 likely | cursor · windsurf · gemini · cline · roo · kilo · opencode |
| 📋 template | qwen · zed · goose · continue · amp · crush · kimi-cli · warp · copilot |

共 21 个客户端，`install.py list` 查看全部。完整流程（每客户端差异、数据目录布局、卸载回滚、故障排查）见 **[INSTALL.md](INSTALL.md)**。

## 其他使用方式

不通过 agent 也能用，三种形态同一引擎：

- **源码 CLI**：`citens run "topic"`（一键脚本 `start.bat` / `start.sh` 自动建环境）；`citens resume` / `reverify` / `browse`；`-l en` 切英文综述
- **Web 控制台**：`pip install -e ".[api]" && citens serve --open`（本机 FastAPI 控制台；对外暴露前务必设置 `API_TOKEN`）
- **Windows 桌面单文件**：GitHub Releases 下载 `CiteLens.exe`（零 Python 依赖，首次运行有向导）

## 工作原理（一分钟版）

```
问题 ──▶ 九家族召回（词法/HyDE/PRF/引文/web桥接/综述种子） ──▶ RRF 家族配额并池
      ──▶ 粗排（pointwise 全池打分，批 25） ──▶ 精排（listwise 50） ──▶ 候选池
候选池 ──▶ 全文获取（OA 优先，拿不到的进手动清单） ──▶ 逐论断写作与核验
      ──▶ 蜜罐检测 + 辩护律师复核 + 缺陷论断改写复验 ──▶ 综述 + 诚实性数字
```

- **写作不是摘要拼接**：Synthesis 提取共识/矛盾/缺口，Reflector 至少跑一轮"补检-重写"循环；每条论断带编号引用，四档判定（supported/partial/unsupported/unverifiable）落到 `provenance.json`，不悄悄保留无依据论断。
- **排序与检索的每个参数都是测出来的**：精排强制关思考（避免正文空转）、批 25（注意力稀释）、OA 积分护栏（每查询 10 credits、日预算）——依据全部在 [docs/bench-postmortem.md](docs/bench-postmortem.md)。

| Grounding | Citation precision（实测） |
|---|---|
| 摘要级 | ~85–92% |
| 全文级（默认） | ~96–98%（改写复验后） |

`citens eval` 可在自己的 LLM 上复测精度与判定器校准。

## 配置速查

唯一硬要求是 LLM key（OpenAI 兼容任意厂商）；推荐的检索源 key 都是免费的：

| 键 | 作用 | 级别 |
|---|---|---|
| `LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL` | 大脑（规划/排序/写作/核验） | **必填** |
| `SERPER_API_KEY` | web 桥接家族（Google SERP，免费 2500 次） | 推荐 |
| `SEMANTIC_SCHOLAR_API_KEY` | S2 更高速率限制（免费） | 推荐 |
| `OPENALEX_API_KEY` + `OPENALEX_EMAIL` | OpenAlex keyed 池（免费 1 万积分/天） | 推荐 |

全量键说明（强弱档位、OA 积分预算护栏、并发、机构代理等）见 [docs/CONFIGURATION.md](docs/CONFIGURATION.md)。

## Roadmap

- 文献管理增强：池内去重报告、阅读笔记、导出格式扩展
- 语义 embedding 召回通道（攻词面断层——现有四条替代路线的排除法结论见 postmortem M 节）
- 更多 agent 客户端升级到 verified

## License

MIT。 venue 排名数据（`citens sjr` 下载的 SCImago CSV）为 CC BY-NC，仅限非商业使用。
