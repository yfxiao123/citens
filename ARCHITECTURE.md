# Architecture Review — citens (2026-08)

A strict pass over "does every component live in the right language/runtime?"
Verdict first: **the core stays Python; two components were replaced where
another technology was genuinely better; several rewrites were considered and
rejected with reasons.** Nothing here is dogma — each decision lists the
trigger that would reverse it.

## Decisions

| Component | Decision | Why |
|---|---|---|
| Pipeline orchestration, agents, collect | **Python (keep)** | The workload is network/LLM-bound: a run spends minutes waiting on LLM and scholarly APIs, milliseconds in our code. A Go/Rust rewrite would not move wall time, cost, or precision, and would reset 156 tests and the pydantic model chain. |
| Lexical retrieval (BM25) | **Replaced above 400 docs with SQLite FTS5** (C, stdlib) behind the same `bm25_rank_texts` interface, silent fallback to the pure-Python scorer | C-speed ranking as pools grow; zero new dependencies; behavior contract (all indices returned) preserved. Also fixed CJK: `_terms` now emits bigrams — before, `[a-z0-9]+` matched *nothing* in Chinese text, so Chinese-pool BM25 was index order. |
| Review user interface | **Replaced markdown-only output with a self-contained HTML+JS artifact** (`review_browser.html`, `citens/browse`) | Filtering/searching claims is interactive UI — JS's home turf. Generated from a Python template with embedded JSON; **no Node toolchain enters the package**. When a real SPA arrives it consumes the same `verification.json`/`provenance.json`. |
| Literature pool storage | **JSONL (keep); rejected SQLite as source of truth** | "Open the file and read it" was an explicit product requirement for the record-first workflow. Derived indexes (`.emb.json`) stay sidecar so the pool remains append-only plain text. |
| API layer | **FastAPI (keep)** | Already the right size; SSE events already flow. |
| Full TS/React frontend | **Rejected for now** | No multi-user deployment target yet; the HTML artifact covers the single-reader audit loop. Trigger to reverse: a hosted deployment or multi-user annotations. |
| Rust/Tantivy search daemon | **Rejected for now** | Premature at pool sizes ≤ thousands. Trigger: >50k records or sub-second interactive pool search. |
| Process-isolated plugin system (dsh-style) | **Rejected** | Our "agents" are in-process functions sharing typed models; none need sandboxing or independent versioning. |

## Module map

```
citens/
├── cli.py                  typer entry: run / collect / audit / browse / resume / reverify / serve / eval / sjr
├── config.py               Settings (.env): two-tier models, polite-pool email, retriever, profile
├── models.py               Paper → ScoredPaper → ExtractedPaper; Chunk/Claim/Verdict(5-grade)
├── llm.py                  chat/chat_json, batched concurrent calls, usage telemetry, JSON retries
├── net.py                  httpx client (proxy/EZproxy prefix)
├── cache.py                disk cache namespaces (search/enrich/embed)
├── runlog.py               append-only event log, per-stage token attribution
├── collect.py              record-first pool: per-query attribution, taxonomy backfill,
│                           field-constrained + review passes, author engagement, hybrid RRF recall
├── profiles.py + profiles/ pure-data domain profiles: terms / venue whitelist / subfields /
│                           EN→ZH terminology ledger / primary-source order
├── ranking.py              composite ranking (venue boost, author depth, SJR)
├── audit.py                human audit sheet ⇄ verifier calibration
├── artifacts.py            review_browser.html generator (HTML+JS template, embedded JSON)
├── search/                 arxiv / semantic_scholar / openalex / crossref + snowball + seeds
├── grounding/              fulltext, ChunkStore (shared across rounds), retrieval
│                           (bm25 | keyword | embedding, FTS5 fast path), citations (bib+RIS), provenance anchors
├── agents/                 planner, filter, extract, organize, synth, writer, verifier (5-grade),
│                           defense, rewriter, health, reflector, verifier_trigger, …
├── orchestration/pipeline.py  run stages + reflect loop; reverify.py
├── eval/precision.py       metrics sweep
└── api/app.py              FastAPI + SSE
```

## Data flow

```
citens collect ──▶ data/litdb/<topic>.jsonl (+ .emb.json index)
                        │  recall: BM25+vector RRF, reviews pass through
citens run ─────────────┘
     planner(profile) → filter → enrich → ground(fulltext) → extract
     → compose: organize → synth → write → verify(5-grade) → defense → rewrite → leniency
     → reflect(≤2 rounds, saturation stop) → finalize
              └──▶ runs/<topic>-<ts>/  review.md · references.bib · references.ris
                    verification.json · provenance.json (chunk anchors) ·
                    review_browser.html · 审核清单.md → citens audit (calibration)
```

## Invariants worth defending

1. **Precision is a pipeline behavior, not a scorecard** — rewrite closes the
   loop; leniency spot-check and the human audit keep the judge honest.
2. **Every grade that is not "grounded" has a distinct remedy** —
   unsupported→weaken, background→re-aim at primary sources (or retrieval
   supplement), contradictory→surface the disagreement.
3. **Recall is cheap, screening is priced** — pool grows append-only;
   deterministic pre-recall keeps LLM filter cost flat.
4. **Degrade, never die** — embedding API down→BM25; one source
   down→others continue; FTS5 missing→pure-Python BM25.
5. **Domain knowledge is data** (profiles), **never code**.
