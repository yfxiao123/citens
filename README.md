# CiteLens (citens)

**An open-source literature-review agent that writes critical, citation-grounded surveys — every claim verifiable against its source.**

Give it a topic. It plans English search queries, fans out over arXiv / Semantic Scholar / OpenAlex / Crossref, reranks the pool by **abstract relevance × citations × journal quartile (SCImago SJR)**, extracts structured findings, then writes a survey with numbered claims — and **verifies each claim against the source text**, reporting a citation-precision score instead of hoping for the best.

```
citens run 订单簿建模 -n 8
...
✓ 8 papers · 3 themes · 70 cited claims
✓ citation precision 73%   (3/8 papers grounded on full text)
```

> 中文简介见[下方](#中文简介)。

---

## Why this exists

Most "AI literature review" tools are summary aggregators: they retrieve, paraphrase, and list. Two things are almost universally missing — and they're the whole point here:

**1. Verifiable citations.** The writer only emits claims tied to numbered references. A verifier agent then judges every claim against the cited paper's ground text (full text when an open PDF exists, abstract otherwise) with a four-way verdict — `supported / partial / unsupported / unverifiable` — and the run reports **citation precision = (supported + partial) / verifiable claims**. Unsupported claims are visible in `provenance.json`, not silently kept. *(Among popular open-source agents, only PaperQA2 does comparable grounding — and it answers questions rather than writing surveys.)*

**2. A critical stance.** A Synthesis agent extracts **consensus, contradictions, and gaps** across papers; the writer argues a position instead of concatenating abstracts. A Reflector agent checks coverage, runs a gap-targeted supplementary search, and recomposes the survey — at least one full loop per run.

And one practical bet you won't find elsewhere:

**3. Full text first, honesty always.** The agent automatically fetches open-access PDFs (arXiv → OA link → Unpaywall) so claims ground on methods and results, not 150-word abstracts; what it can't fetch lands in a manual fetch list (or rides your proxy if you have one) — see [The access layer](#the-access-layer).

## How it works

```
            CLI (Typer/Rich)      FastAPI + SSE + single-page UI
                        \             /
                 ┌───────────────────────────┐
                 │        orchestrator        │  events · run-dir · cache
                 ├───────────────────────────┤
  planner → search → rank → filter → enrich → extract
        (4 sources,    (SJR      (LLM        (cross-source
         async)        quartile   score)       abstract fill-in)
         + citations)
                 → organize → synthesize(consensus/contradictions/gaps)
                 → write([n] claims) → verify(LM-as-judge vs ground text)
                 → reflect → supplement → recompose
                 ├───────────────────────────┤
                 │ grounding: full-text PDFs → chunks → retrieval        │
                 │ access: proxy · EZproxy rewrite · manual PDF drop     │
                 └───────────────────────────┘
```

Every run writes an inspectable directory:

```
runs/<topic>-<timestamp>/
├── review.md              # the survey, with [n] citation markers
├── references.bib         # BibTeX
├── references.ris         # RIS (EndNote/Zotero import)
├── review_browser.html    # self-contained audit UI (see below)
├── provenance.json        # claim → reference → verdict → evidence-chunk anchors
├── verification.json      # per-claim verdicts (5-grade) + precision + citation coverage
├── grounding.json         # which papers have full text vs abstract only
├── fetch_list.md          # papers YOU can fetch manually (see below)
└── steps/*.json           # every intermediate stage
```

References render as complete APA-style entries — full author list, `*venue*`, volume(issue), pages, DOI (journal/volume/issue/pages are harvested from OpenAlex `biblio` / Crossref and backfilled into the pool). `verification.json` also reports **citation coverage** (`papers_cited`/`papers_total`); below 70% the health report flags `thin_citation_coverage`, and the writer is under a cite-broadly rule within every theme.

**Two citation tiers.** A survey cites far more than it dissects, so the bibliography is not capped at the deep-dive set: papers that pass the relevance filter but fall beyond `-n` join as a *supporting layer* (`--support-papers`, default 15) — abstract-only bibliography entries the writer may cite for background, context, and comparisons (never primary method/result claims; the verifier checks those cites against the abstracts like any other). Core papers get full extraction and full-text grounding; supporting papers cost nothing beyond a reference entry. `-n 20` therefore yields up to ~35 references.

## Quick start

**Desktop app** (zero install — recommended if you just want to run it):

```
https://github.com/yfxiao123/citens/releases/latest/download/CiteLens.exe
```

Download, double-click: a first-run wizard asks for your LLM provider/key
(DeepSeek / OpenAI / Ollama / any OpenAI-compatible base URL), then the web
console opens in your browser. Everything (`.env`, caches, fetched PDFs,
runs) lives **next to the exe** — portable: move the folder, keep your data;
delete it, nothing remains. ~67 MB, unsigned (SmartScreen may ask "run
anyway"); the first launch takes ~10-20 s (self-extraction).

**From source** (one click — auto-creates the venv, installs dependencies
including the PDF toolchain, opens the web console in your browser):

```
双击 start.bat        # Windows
./start.sh            # macOS / Linux
```

Then type a topic in the console and press run. Manual equivalent:

```bash
uv sync                      # or: pip install -e .  (PDF toolchain is a core dependency)
cp .env.example .env         # fill in LLM_API_KEY (OpenAI-compatible)
citens sjr                   # one-time: fetch SCImago journal ranks
citens run "limit order book modeling" -n 8
```

Works with any OpenAI-compatible backend — DeepSeek (`LLM_API_BASE=https://api.deepseek.com/v1`), Ollama, OpenRouter, vLLM, Groq — or native Anthropic/Gemini via the `[multi]` extra (LiteLLM).

Useful follow-ups after a run:

```bash
citens resume runs/<dir>            # an interrupted run continues from its
                                    #   extracted papers — no re-retrieval
citens reverify runs/<dir>          # drop fetch_list.md's PDFs into papers/
                                    #   first: re-verifies every claim against
                                    #   the new full text, reports the delta
citens browse runs/<dir> --open     # single-file HTML audit browser: filter
                                    #   claims by verdict, search, expand the
                                    #   evidence chunk behind each claim
citens run "..." -l en              # English output — Chinese is the DEFAULT
                                    #   (asked interactively when -l is omitted)
```

### Desktop app (Windows, no Python needed)

Prefer not to install anything? Download the single-file desktop build:

```
https://github.com/yfxiao123/citens/releases/latest/download/CiteLens.exe
```

Double-click → the web console opens in your browser (first run: the settings
page opens automatically — pick a provider preset and paste your API key).
No terminal windows; the run's live transcript, results, and the ⏻ exit
button all live in the web console. Data (`.env`, caches, fetched PDFs,
runs) stays put across downloads: it follows the workdir you chose (⚙
settings), or an exe-folder `.env` (portable-folder mode), or reattaches
to the last-used data home — a freshly downloaded exe never starts blank.
The header chip shows the active data directory. Launching again while it
runs just opens the browser (single instance). Note: ~67 MB, unsigned (SmartScreen
may ask "run anyway"), and the first launch takes ~10-20 s (self-extraction;
nothing visible until the browser opens). Build it yourself with
`pip install -e ".[api,packaging]" && pyinstaller citens.spec`.

### Desktop app (Windows, no Python needed)

See [Quick start](#quick-start) above — the single-exe build is the fastest
way in. Prefer a console/build from source? `start.bat` below opens the same
web UI; build the exe yourself with
`pip install -e ".[api,packaging]" && pyinstaller citens.spec`.

Web UI (live agent transcript: every step, every model call, the retrieved
queries/papers/verdicts as they happen; refresh-safe — reload mid-run and
the transcript rebuilds) — what `start.bat` opens:

```bash
# start.bat / start.sh already do this; manually:
uv sync --extra api             # or: pip install -e ".[api]"
citens serve --open             # http://localhost:8000 · Ctrl+C 停止
# exposing the server beyond localhost? set API_TOKEN (bearer auth on
# /run & friends — /run spends your LLM credits) and CORS_ORIGINS.
```

CLI reference: `citens run | resume | reverify | eval | sources | sjr | version` (`citens --help`).

## The access layer

Grounding claims on full text is the single biggest precision lever, so CiteLens goes after it automatically — no institutional access required:

1. **Open-access auto-fetch (default, works everywhere)** — every paper runs through arXiv (by link, or by title match for paywalled papers with a preprint), the OA links harvested from OpenAlex / Semantic Scholar (`openAccessPdf`), Unpaywall, and CORE (with a free `CORE_API_KEY`). In an NLP/ML-heavy pool this lands 80%+ of full texts on its own; the writer's evidence excerpts (biased toward effect-size-dense passages) and the verifier both ground on them.
2. **Manual drop (always works)** — papers the agent can't fetch are listed in `fetch_list.md` with DOIs and suggested filenames. Download them wherever you have access, drop the PDFs into `papers/`, and the next run (or `citens reverify`) automatically grounds claims on their full text. No filename bookkeeping — DOI, arXiv id, or recognizable title words all match.
3. **Proxy / EZproxy (optional, for those who have it)** — set `HTTP_PROXY`/`HTTPS_PROXY` (+ `ACCESSIBLE_DOMAINS` allowlist) and full-text fetches ride it; `EZPROXY_PREFIX` rewrites publisher URLs through a library proxy. Purely additive — everything works without them.

**Keys worth configuring.** Everything above works with zero keys, but each of these measurably raises the full-text hit rate (the single biggest precision lever) — copy `.env.example` to `.env` and fill in what you can:

| key | cost | what it buys |
|---|---|---|
| `CORE_API_KEY` | free registration | repository-aggregated OA PDFs by DOI — the biggest free full-text boost |
| `SEMANTIC_SCHOLAR_API_KEY` | free registration | stable S2 access (no key: 1 rps shared, frequent 429s shrink the pool) + OA PDF links |
| `OPENALEX_EMAIL` / `CROSSREF_EMAIL` | free, no registration | polite pools — faster, more stable metadata + OA links |

Fetched PDFs are **kept**: every successfully fetched PDF lands in `papers/auto-<doi>.pdf`, so later runs (and `citens reverify`) re-ground offline even after the text cache is cleared — URLs rot and publisher versions drift, the file doesn't.

The honesty rule: a paper grounded only on its abstract is labeled as such in `grounding.json`; nothing is silently upgraded to "verified".

## Venue-aware ranking

Retrieval order decides which papers survive. Ranking on raw citations starves fresh preprints; ranking on LLM relevance alone is opaque. CiteLens blends three factors with configurable weights (default 0.6 / 0.2 / 0.2):

```
rank = 0.6 · (relevance/5)  +  0.2 · min(log10(1+cites)/3, 1)  +  0.2 · venue(SJR quartile)
```

`citens sjr` fetches the SCImago Journal Rank dataset (official field-normalized quartiles via a GitHub mirror; a no-dependency CSV fallback approximates by percentile). The SJR data is **CC BY-NC**, so it's downloaded on demand — never committed or redistributed. Without it, the venue factor is neutral and everything else still works. Every run stores its full ranking breakdown (`steps/03c_ranking.json`) — explainable by construction.

## Citation precision, honestly

Precision depends on how much ground text exists. Same topic, same model:

| grounding | claims | precision |
|---|---|---|
| abstracts only | ~50 | ~35% |
| + 1 full text (open PDF) | 53 | 47% |
| + dropped PDFs & OA fetches (3/8 full text) | 70 | **73%** |
| + snowballing, defense review, deep_review mode (13 papers) | 65 | **94%** |

The number is computed by an LLM-as-judge with the cited paper's retrieved chunks in view — it's a working measure, not a certification. Claims whose cited paper has no ground text are excluded from the denominator and reported as `unverifiable` instead.

The judge grades on a **five-grade scale**: `supported` / `partial` (both count as grounded), `background` (the source — often a survey — backs field context only; the rewriter re-aims the claim at primary sources), `contradictory` (the source disagrees; the rewrite surfaces the disagreement instead of papering over it), and `unsupported` (weaken or drop). A review whose citation is a survey can therefore no longer sneak an empirical claim past the judge.

**The judge is itself audited.** A human/strict-audit calibration pass (22 claims) found the judge self-reporting 100% where the audit grounded only 68% — the gap traced to an explicit leniency tie-break in the prompt and to claims riding on abstract-less papers. The tie-break is gone, three calibration rules now bind the judge, a canary honeypot (synthetic unsupported claims through the same judge) measures the false-accept rate on every run, and a strict spot-check's downgrades are adopted into the final verdicts. Post-fix A/B on the same claims: self-reported precision 68.2% = audited grounded rate. `verification.json` also reports `unverifiable_rate` — thin ground text is a quality signal, not just a precision problem.

Reproduce it yourself — the eval harness runs a topic set (or summarizes existing runs) and writes a comparison table:

```bash
citens eval --from-runs "runs/*"          # offline: table from existing runs
citens eval "limit order book modeling" -n 13   # live: run + collect
```

A full example run — 31-reference LLM-recommender survey, 99% audited precision on 107 claims, 21/24 full-text-grounded, self-contained audit browser — lives in [`examples/llm-recommender-systems/`](examples/llm-recommender-systems/).

## Runtime at scale

Everything LLM-bound is parallel (filter/extract/verify/write/defense run on a thread pool, `LLM_CONCURRENCY`, default 6), extraction folds quality grading into a single call per paper, and the LLM, search, and full-text layers are all disk-cached — re-runs of the same topic skip repeat calls. Every run writes `timings.json` with per-stage durations — if a run feels slow, that file says exactly where the time went.

v1.1 halved deep-run wall time and cost with three knobs (all in `.env`):

- **`JUDGE_THINKING=low`** (default) — hybrid reasoning models share one budget between thinking and body; the judge reasons at low effort. Golden-set A/B: precision 0.659 vs 0.682 at full effort (human-calibrated level), ~2x faster per verify batch. `none` is fastest but measurably lenient — scans only; `true` restores full deliberation.
- **`REFLECT_MAX_ROUNDS=1`** (default) — supplementary retrieval recomposes the whole survey (re-verification included); each extra round was ~20+ minutes for a handful of papers.
- **Fuzzy verdict reuse** — a recompose round re-judges only claims that actually changed; reworded restatements with unchanged citations and ground text reuse their earlier verdicts (the run log prints `reuse: N identical · M reworded · K to judge`).

A 20-paper deep review lands around 40 minutes on a fast backend (`-n 15 REFLECT_MAX_ROUNDS=0` roughly halves that again).

## Project layout

```
citens/
├── cli.py                     # Typer CLI
├── api/                       # FastAPI + SSE + static UI
├── orchestration/pipeline.py  # stages, events, run-dir persistence
├── agents/                    # planner/filter/extract/organize/synth/writer/verifier/reflector
├── search/                    # pluggable async sources (arXiv/S2/OpenAlex/Crossref)
├── grounding/                 # chunk store, citation table, BibTeX, provenance,
│                              #   full-text fetch, abstract enrichment, fetch list
├── ranking.py                 # SJR index + composite rerank
├── net.py                     # access layer: proxy, EZproxy rewrite, domain allowlist
└── models.py / config.py / events.py / cache.py / persistence.py
```

## Roadmap

- chunk-level provenance (claim → exact passage, not just paper)
- citation-graph snowballing (references of included papers)
- formal eval on a public survey benchmark
- PDF ingestion for a whole existing library
- multilingual survey output

## License

MIT. The runtime-fetched SCImago dataset is CC BY-NC (attributed to SCImago Lab; fetched by `citens sjr`, never redistributed here).

---

## 中文简介

**CiteLens（包名 `citens`）**：输入一个研究主题，自动完成「关键词规划 → 四源并发检索（arXiv / Semantic Scholar / OpenAlex / Crossref）→ 期刊分区×引用×相关性的复合排序 → LLM 筛选 → 摘要交叉补全 → 结构化抽取 → 主题组织 → 批判性综合（共识/矛盾/空白）→ 带引用撰写 → 逐条核验 → 反思补充检索」，产出一份**每条论断都可回溯到原文**的综述。

与“摘要拼接器”们的三点不同：

1. **可信引用**：撰写只输出带 `[n]` 标记的论断；核验 agent 逐条对照被引论文的原文（自动抓取开放 PDF——arXiv/OA/Unpaywall，抓不到的给出手动获取清单），给出 supported/partial/unsupported/unverifiable 四类判定，并报告**引用精度**。整条链路存在 `provenance.json`，可审计。
2. **批判立场**：综合 agent 显式提取跨论文的共识、矛盾与研究空白；反思 agent 发现覆盖缺口后补检并重写——不是罗列，是论证。
3. **全文优先、诚实标注**：开放获取 PDF 自动抓取并按"效应量密度"挑选证据段落；拿不到全文的论文明确标注"仅摘要核验"，绝不冒充"已核验"（有校园代理/VPN 的可选配，见 [The access layer](#the-access-layer)）。

快速开始：**免安装**——从 [Releases](https://github.com/yfxiao123/citens/releases/latest/download/CiteLens.exe) 下载 `CiteLens.exe`，双击后按向导填 API key 即可；或源码方式：双击 `start.bat`（Windows） / `./start.sh`（macOS/Linux）——自动建环境、装依赖、打开网页控制台；手动方式 `uv sync` → 填 `.env` → `citens run 主题 -n 8`。

示例产物见 [`examples/llm-recommender-systems/`](examples/llm-recommender-systems/)（主题「基于大语言模型的推荐系统」，31 篇参考文献，107 条论断，审计后引用精度 99%，21/24 篇全文溯源，附自包含审计浏览器）。
