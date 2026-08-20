# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [1.2.0] — 2026-08-20

### Added — in-app settings UI (the desktop config manager)
- Settings page in the web console (⚙ in the header): manage the LLM
  backend (ANY OpenAI-compatible base URL + model, free text — not just
  the wizard's presets), scholarly-source keys (Semantic Scholar,
  OpenAlex/Crossref polite emails, CORE), and optional proxy/EZproxy —
  all in one place, secrets masked on read-back.
- `GET/POST /settings` + `POST /settings/test` (one tiny completion to
  verify key/base/model, reports latency); saved values apply to the
  live process immediately (backend cache reset) and persist to `.env`
  with unknown lines preserved.
- `CITELENS_WORKDIR`: point the data directory (runs/, papers/, pools,
  cache) anywhere — one copy of the exe, data where you choose; the
  config migrates with the data. Set it from the settings UI.

## [1.1.1] — 2026-08-20

### Added — single-exe desktop app (Windows)
- `CiteLens.exe` (PyInstaller onefile, ~67 MB): double-click → first-run
  wizard (DeepSeek / OpenAI / Ollama / any OpenAI-compatible key+base+model)
  → local web console opens in the browser. Portable: `.env`, `.cache`,
  `papers/`, `runs/`, `data/` all live next to the exe.
- `citens/desktop.py` (also `citens-desktop` console script); free-port
  fallback 8000-8009; `--import-check` smoke mode.
- `citens.spec` + `packaging` extra; `.github/workflows/release.yml` builds
  and attaches the exe to GitHub Releases on `v*` tags.
- Tested end-to-end on Windows: health/console/runs endpoints from the
  frozen bundle (first launch ~10-20 s — self-extraction + AV scan).

### Fixed — full-text harvest & arXiv resilience (round 2)
- **Fetch-time OA lookup now harvests ALL locations, from three sources**:
  Semantic Scholar `openAccessPdf` by DOI, OpenAlex `locations[].pdf_url`
  (not just `best_oa_location`), and Unpaywall's full `oa_locations[]`.
  Repository and author-homepage copies live beyond the "best" location —
  measured on a 14-paper finance run: +2 full texts (Cornell homepage copy,
  French institutional repository), 36% → 50% grounding with zero keys.
- **arXiv source bounded at 90s.** export.arxiv.org server-side rate-limits
  shared egress IPs with deliberately slow 429s (measured: TCP connect
  0.1s, then 16s-to-respond 429 / read timeouts), and the arxiv client's
  urllib layer has no per-request timeout — a 13-query search crawled 21
  minutes and still returned nothing. The adapter now gives up at its
  budget, prints why, and lets the other sources carry the run.

### Fixed — recall & full-text hit rate (user report: narrow pools, 1/8 full text)
- **Candidate-pool cap raised 3×n → 8×n** (bounded by `max_results`) before
  LLM screening. At `-n 8` the filter used to see only 24 candidates, and
  `blend_pool`'s citation-based trim cut arXiv's zero-cited records hardest
  — the OA-richest source was systematically diluted out of the pool.
- **OpenAlex `title.search` → `default.search`** (title + abstract).
  3-6-word planner queries had to match the TITLE alone; most relevant
  papers carry those words in the abstract.
- **Semantic Scholar `openAccessPdf` harvested** into `paper.pdf_url` —
  S2 returns free OA links for preprints and green-OA deposits; they were
  never requested.
- **Unpaywall landing-page fallback removed** (`url_for_pdf` only): the
  `url` field is usually an HTML landing page that fails the download
  step's content-type check after a wasted fetch.
- **arXiv per-keyword floor 1 → 5** (matches other sources): with 10+
  queries the OA-richest source contributed ~1 paper per query.

### Changed — fetched PDFs are kept, not discarded
- Successfully fetched PDFs now persist as `papers/auto-<doi>.pdf` (same
  slug-matching path as manual drops). Before: download → temp file →
  convert → delete, leaving only disposable text in `.cache` — a cleared
  cache or a rotted URL forced a full re-fetch and runs were not
  re-groundable offline. Unconvertible PDFs are not kept.
- `fulltext` / `fulltext_local` cache namespaces never expire (derived
  from persisted PDFs; a stale miss self-heals via the local-first scan).
- Local-PDF conversion cached by file mtime — re-parsing every dropped
  PDF on every run was pure repeated work.

### Added — reliability & observability
- LLM transport retries with exponential backoff (429/5xx/timeout; auth
  errors surface immediately) for both OpenAI-compatible and LiteLLM
  backends.
- Cache TTL (`CACHE_TTL_DAYS`, default 30) with throttled sweep; `llm`
  and `fulltext*` namespaces are exempt.
- Token usage attributed per-run (`llm.run_scope`) — concurrent runs in
  one process no longer cross-attribute; thread-pool jobs tagged too.
- Health report warns `judge_model_uncalibrated` when the judge
  model/thinking differs from the human-calibrated golden set — reported
  precision is unanchored until re-audited.
- Profile `evidence_bias` field (`number_density` | `none`): theoretical/
  mathematical domains keep plain BM25 excerpt order instead of the
  empirical number-density boost.
- PDF-ingestion smoke test (hand-rolled minimal PDF through MarkItDown).
- Pipeline helpers extracted to `orchestration/support.py`
  (pipeline.py 1790 → 1515 lines); imports re-exported for compatibility.

## [1.1.0] — 2026-08-19

### Changed — speed package (88-min deep run → target ≈ half)
- **Judge-side thinking → "low"** (`JUDGE_THINKING=true|low|none`): verify
  batches, defense, rewriter, spot-check, canary, reflect/absence audit,
  organize/synth/health — every structured-JSON call on the strong tier —
  now reason at low effort instead of the provider default. Hybrid models
  share the completion budget between thinking and body; full deliberation
  made each verify batch 2-12x slower. Golden-set A/B (49 claims, labels
  inherited from the 22-claim audit): HIGH 0.49 human-agreement / precision
  0.682 / ~106s · LOW 0.388 / 0.659 / 48s (errors balanced 16 lenient vs 14
  strict) · NONE 0.204 / 0.864 / 9s but systematically LENIENT (31
  lenient) — rejected as the verify default, kept reachable via env.
- **Writer ladder flipped**: sections now generate WITHOUT thinking first
  (normal → double budget → double budget WITH thinking as last resort).
  The thinking-first ladder wasted a full call per section whenever the
  thinking prefix starved the body (9 sections in the 08-19 run returned
  empty on attempt 1 and were rescued by the no-thinking retry anyway).
- **Supplementary retrieval capped at 1 round** (`REFLECT_MAX_ROUNDS`):
  deep_review used to run 2; round 2 alone cost ~37 min (a full recompose
  incl. re-verification) for 3 added papers. The absence audit still runs
  every round and its findings land in `08a_absence_audit.json` for manual
  follow-up.
- **Fuzzy verdict reuse across rounds**: a recompose round now reuses
  verdicts for near-identical restatements (similarity ≥ 0.88, identical
  citations, unchanged ground text — a paper that gained fulltext is always
  re-judged). The writer restates most unchanged facts with light rewording;
  re-judging them cost the majority of each post-supplement verify pass.

### Fixed
- **Full-text grounding was silently dead**: `markitdown[pdf]` was declared as
  an optional extra but never installed in the working venv — every PDF fetch
  (auto or user-dropped) raised `ModuleNotFoundError` inside a broad `except`
  and degraded to abstract-only grounding with no signal (0/20 full texts in
  the 08-19 stock-prediction run). It is now a core dependency, and a missing
  install prints a loud one-time warning instead of failing silently.
- **Decimal points no longer shatter claims**: the sentence splitter (widened
  to zero-width for Chinese prose in 1.0.0) also split after every Latin `.`,
  fragmenting `0.92` into `0.` + `92[15]。` — the verifier then judged
  context-free fragments. Latin `.!?` now require a following space
  (`(?<=[。！？])\s*|(?<=[.!?])\s+`); decimals (`AUC≈0.5`, `gpt-3.5`) stay
  inside one claim. Re-verification of the 08-19 LLM-recsys run with the fixed
  parser: 108 claims, precision 100% → 99% (the fixed text surfaces one
  honestly-failing claim).

### Changed
- **Number-dense evidence excerpts**: the writer's full-text excerpts
  (top BM25 chunks per paper) favored intro/method prose; candidates are now
  re-ranked with an effect-size density boost (`37%`-style tokens) before
  excerpting (top 3 per paper, 900 chars, 7.5k-char theme budget). The 08-19
  LLM-recsys run — first with a live PDF chain — carried full-text-only
  numbers into the body (`AUC≈0.5`, 128-sample tuning result, Kendall's τ
  0.92), none of which appear in the abstracts.

## [1.0.0] — 2026-08-19

### Added
- **Facet-based coverage (borrowed from academic-harness plans)**: the planner
  now emits 5-8 search facets (`01c_facets.json`); per-facet paper counts are
  computed deterministically, thin facets drive the reflector's supplementary
  queries (with a gap taxonomy — foundational classic / survey / recent
  advance — and dead-channel awareness), and the writer receives a
  coverage-honesty paragraph requirement naming thin directions explicitly
  instead of silently narrowing scope.
- **Hard citation-stacking enforcement**: a post-write BM25 lint caps
  citations per sentence at 4 (prompt rules only softened: the 08-19 run
  still had a 13-citation sentence); keepers are the most relevant cited
  papers, dropped markers are logged (`06b_stack_lint.json`).
- **Cross-round verdict cache**: compose rounds re-judge only claims whose
  text, citations, or cited ground text changed — re-compose rounds stop
  paying full verification cost. Verify batches grew 6→10.
- **Supplement-path blind gate**: reflect-supplemented papers now pass the
  same abstract gate as the main path (enrich, then demote blind papers to
  the supporting layer) — the 08-19 order-book run's 7/26 blind core papers
  entered exactly this way.

### Fixed
- verify_claims now returns results parallel to the claims list (unverifiable
  verdicts used to be appended at the end, misaligning every downstream index
  — rewriter, spot-check, defense).

### Added
- **Web console** (`citens/api/static/`, zero-build): the minimal run page is
  now a three-pane agent console — run history sidebar (reopen any past run),
  live pipeline timeline with per-step elapsed times and collapsible messages,
  a metrics strip (precision / canary false-accept / leniency corrections /
  unverifiable rate, live-sniffed from step messages and finalized from
  `verification.json`), and an artifact viewer with tabs (review · references
  with .bib/.ris download · fetch list · per-claim verdicts · audit-browser
  link). marked.js is vendored locally so rendering works offline. Backed by
  a new `GET /artifact/{run_id}/{filename}` route — whitelist + resolved-path
  containment, traversal-proof, bearer-auth'd like /run.
- **Per-request thinking control** (`reasoning_effort: "none"` via the new
  `thinking=` flag on `chat`/`chat_json`): hybrid reasoning backends
  (deepseek-v4-flash) share one completion budget between thinking and the
  visible body, so a long deliberation starves the body to empty — the
  failure mode behind today's empty verify batches, truncated organize JSON,
  dead rewriter call and an all-sections-empty writer spell. Mechanical
  callers (intent, planner) now run with thinking off; the writer's retry
  ladder escalates budget → budget → budget-without-thinking, so a provider
  spell degrades prose deliberation instead of deleting sections (live:
  3 sections rescued on the no-thinking attempt, 33 claims vs 5 without it).
- **Semantic Scholar authenticated tier**: `SEMANTIC_SCHOLAR_API_KEY` now
  rides as `x-api-key` with a process-wide 1 req/sec throttle (the key's
  shared budget) and a 429 backoff-retry; abstract enrichment gained an S2
  by-DOI source — S2 crawls publisher and preprint pages itself, so it holds
  abstracts that OpenAlex/Crossref lack (Elsevier journals deposit none;
  SSRN DOIs carry none). Live effect on the same topic: enrichment fill rate
  0/7 → 4/7.
- **Blind-paper demotion**: after enrichment, core papers with no abstract
  and no OA pdf (neither extractable nor verifiable) are swapped for the
  next-ranked abstract-bearing alternates and demoted to the supporting
  layer — `-n` now means "-n verifiable papers". Swaps logged in
  `steps/03e_blind_demotion.json`.
- `verification.json` now carries a `citation_stacking` lint
  (max citations per claim, count of >4-citation claims).

### Changed
- Writer register: formal-academic rules distilled from the nature-writing /
  nature-polishing skills now bind every section prompt — one proposition
  per sentence (split comma-chains), no essayistic commentary or rhetorical
  questions, verbs calibrated to evidence strength, topic-first paragraphs,
  no vague quantifiers. The 08-18 review averaged 98 chars/sentence with 31
  sentences over 150 chars; comma-chained multi-proposition sentences were
  the dominant informality.
- Citation-stacking cap is now numeric: at most 3 `[n]` markers per sentence
  (a 4th only when each backs a distinct part), never 5+ — the 08-18 review
  had one claim wearing 13 citations.
- Theme organization degrades instead of dying: a truncated/garbled judge
  response falls back to deterministic rank-order grouping (previously it
  killed the whole run — observed live), with the fallback visible in the
  headings ("主题 N（自动分组）").
- Verifier recalibrated from a human audit (run 201038: machine self-report
  100% vs strict-audit grounded rate 68%, 12 lenient / 1 strict verdicts).
  The leniency tie-break ("when in doubt prefer supported/partial") is
  REMOVED from the judge prompt and replaced with three calibration rules:
  no-ground-text citations contribute nothing (core resting on them →
  unsupported), interpretive framing caps a claim at partial, and every
  citation in a multi-cite claim must back its part. A/B on the same 22
  claims (`runs/ab_reverify_201038`): agreement 41%→55%, lenient 12→6,
  self-reported precision 100%→68.2% — now equal to the audited grounded rate.
- Verifier fallback for a missing/malformed judge response is `unverifiable`
  (excluded from the denominator) instead of `partial` (free precision).
- Judge-call token budgets raised (2k→8k): the calibration rules make the
  judge reason longer, and a tight budget let reasoning squeeze the JSON
  body to empty (observed live on the A/B batch).
- Leniency spot-check now has teeth: strict re-audit downgrades are ADOPTED
  (supported→partial) into the final verdicts and headline precision, not
  just reported.
- Canary honeypot per run: synthetic unsupported claims through the same
  judge in a separate call; the false-accept rate lands in
  `verification.json` and trips a `verifier_false_accept` health issue.
  Live check: 3/3 caught.
- `verification.json` now reports `unverifiable_rate`, and health metrics
  include it (thin ground text is a quality signal, not just precision).
- Writer: papers without abstracts are marked `NO ABSTRACT` in the prompt
  and fenced to title-level context (the audited run's 7 worst mis-grounds
  all came from claims about an abstract-less paper); interpretive framing
  must stand WITHOUT a citation; decorative citation stacking is banned.
- Golden set + 13 regression tests (`tests/test_verifier_calibration.py`,
  `tests/golden/verifier_calibration_201038.json`) pin the prompt contract
  so the calibration cannot silently regress.
- Large-run performance: relevance filtering (the ~300-candidate pool of a
  100-paper target) now runs on the thread pool instead of sequentially;
  defense rebuttals are parallel too; extraction folds quality grading into
  the same call (2 calls per paper → 1); filter calls use a 1024-token
  budget instead of the 4096 default
- New `--concurrency/-c` flag on `citens run` (+ hint for `-n 30+` runs)
- Every run now writes `timings.json` (per-stage durations; also on failure),
  and `RunCompleted` carries total seconds — "why was this slow" is now a
  file you open, not a guess

## [0.1.0] — 2026-08-15

First tagged release. Positioning: an open-source literature-review agent
that writes critical, citation-grounded surveys — every claim verifiable
against its source.

### Added
- Pipeline: planner → async multi-source search (arXiv / Semantic Scholar /
  OpenAlex / Crossref) → LLM filter → SJR venue-aware composite ranking →
  citation snowballing → deep structured extraction (evidence levels,
  comparison matrix) → theme organization → cross-paper synthesis
  (consensus / contradictions / gaps) → section-parallel writer with `[n]`
  claim markers → verifier (LLM-as-judge against ground text) → defense
  review of unsupported claims → health monitoring → reflect/supplement loop
- Verifiable citations: `review.md` + `references.bib` + `provenance.json` +
  `verification.json` with a headline citation-precision metric
- Access layer: campus EZproxy URL rewriting, HTTP(S) proxy with domain
  allowlist, manual PDF drop folder (`papers/`) + `fetch_list.md`
- Pre-run clarification (CLI / API / Web UI) whose answers now shape the
  search queries and filtering, including a deterministic year floor
- Output-language control (`REVIEW_LANGUAGE`, `--language`): uniform prose
  and localized headings
- Intent detection with three run modes (quick_scan / deep_review /
  interactive)
- FastAPI + SSE server with a single-page UI; Typer CLI with Rich progress
- Eval harness (`citens eval`) for reproducible precision reporting
- Run-dir persistence with per-step artifacts; LLM + search response cache

### Fixed
- Author deduplication (OpenAlex multi-affiliation entries collapsed at the
  model layer for every source)
- Defense review now receives the cited papers' ground text (previously an
  index/ID mismatch left it with empty context)
- Multi-line model-appended reference blocks fully stripped (regex lacked
  DOTALL — hallucinated entries after the first line survived)
- mypy-clean codebase, enforced in CI alongside ruff and pytest (88 tests,
  including respx-mocked search-adapter contracts)

[Unreleased]: https://github.com/yfxiao123/citens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yfxiao123/citens/releases/tag/v0.1.0
