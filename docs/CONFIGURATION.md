# 配置参考（.env 全量键）

配置文件是数据家目录下的 `.env`（agent 路径默认为 clone 根；用 `portable/install.py init --home .` 交互生成，或 `cp .env.example .env` 手编）。运行期旋钮在 CLI 参数上，不进 `.env`。

## 必填（引擎无法运行的唯一一块）

| 键 | 语义 |
|---|---|
| `LLM_API_KEY` | OpenAI 兼容任意厂商的 key |
| `LLM_API_BASE` | API 地址，如 `https://api.deepseek.com` |
| `LLM_MODEL` | 默认档模型名。规划/打分/写作/核验都走它 |

## 模型分档（可选，不配则三档同一模型）

| 键 | 语义 |
|---|---|
| `LLM_MODEL_STRONG` | 强档：writer/synth/verifier 等智力密集阶段。留空回落 `LLM_MODEL` |
| `LLM_MODEL_CHEAP` | 廉价档：planner 家族/过滤/抽取等高量结构化阶段。留空回落 `LLM_MODEL` |
| `LLM_CONCURRENCY` | 并发请求数（默认 4；按你的 RPM 上限调） |

> 实测提醒：混合推理模型在 listwise/pointwise 排序上**必须关思考**（代码已强制 `thinking=False`）——开着思考会烧穿 token 预算导致正文为空、精排静默失效（postmortem I 节）。

## 检索源（全部免费，配 key = 更高配额）

| 键 | 语义与实测注意 |
|---|---|
| `SEARCH_SOURCES` | 逗号分隔启用源；默认 `arxiv,semantic_scholar,openalex,crossref` |
| `OPENALEX_API_KEY` | keyed 池：免费 $1/天 = **10000 积分，每个 works 查询 10 积分**，UTC 午夜重置。不配 key 落匿名池（1000/天，中午即尽且静默空结果）——强烈建议配 |
| `OPENALEX_EMAIL` | polite pool 邮箱，与 key 同用时提高突发容忍 |
| `OPENALEX_DAILY_BUDGET` | 本地积分护栏上限（默认 9000，早于 API 硬限触发告警/熔断；记账在 `<home>/.cache/openalex_usage.json`） |
| `SEMANTIC_SCHOLAR_API_KEY` | S2 免费档限流严格（约 1 req/s，进程内已节流+重试）；key 提高速率。推荐语料召回与推荐家族都依赖它 |
| `SERPER_API_KEY` | web 桥接家族的 Google SERP（serper.dev，免费 2500 次）。中英双语查询，博客/论坛文本锚定回精确论文——词法源排不进前 100 的金标主要靠它 |
| `CROSSREF_EMAIL` | Crossref polite pool |

## 全文获取与机构访问（可选）

| 键 | 语义 |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | 全文抓取走代理 |
| `EZPROXY_PREFIX` | 机构 EZproxy 前缀，订阅 PDF 经此路由 |
| `ACCESSIBLE_DOMAINS` | 已订阅域名清单 |
| `PAPERS_DIR` | 本地已有 PDF 目录（自动匹配优先用本地） |

## 运行行为（可选）

| 键 | 语义 |
|---|---|
| `DEFAULT_MAX_RESULTS` / `DEFAULT_MAX_PAPERS` / `DEFAULT_SUPPORT_PAPERS` | 检索目标/精读篇数/支撑层的默认值 |
| `REVIEW_LANGUAGE` | 综述默认语言（zh/en） |
| `JUDGE_THINKING` | 核验器的思考预算档（大规模跑时 `low` 提速） |
| `REFLECT_MAX_ROUNDS` | Reflector 补检-重写循环上限（默认 1，够用且省钱） |
| `CACHE_ENABLED` | 检索缓存开关 |

## API 服务（仅 `citens serve` 对外暴露时）

| 键 | 语义 |
|---|---|
| `API_TOKEN` | **对外暴露前必设**，否则接口裸奔 |
| `CORS_ORIGINS` | 允许的跨域来源 |

## 数据家布局

```
<home>/               # --home 指定的目录，删掉即彻底卸载
├── .env              # 本文件所述配置
├── citlens-runs/     # search 运行目录
├── runs/             # review 运行目录
├── data/litdb/       # 文献池落库（跨主题累积）
└── .cache/           # 缓存 + OA 积分记账
```
