# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Fixed — third-party audit findings (2026-08-30 RL+ads run)
- **Same-paper duplicates in references**: a UCL-repository record
  ("Cai, H" surname-first house style) of a paper whose OpenAlex record
  says "Han Cai" survived dedup — the study occupied two reference slots
  ([3]/[6]) in the final review. `_shared_author` now understands the
  "Surname, I" form; re-running the audit run's own pool through fixed
  dedup merges 24 → 22.
- **Verifier same-round consistency**: near-duplicate claims citing the
  same sources were judged in different batches and could disagree (one
  DCMAB restatement "supported", its twin "unsupported"). Conflicting
  twins now align to PARTIAL with a note — conservative, never hides a
  contradiction, no longer presents judge noise as hard failures.
- **Bib/RIS/text-reference URL hygiene**: openalex.org/W... aggregator
  work ids no longer exported (doi/arxiv/publisher landing pages still
  are) — Zotero imports were polluted.
- **OpenAlex premium api_key** (`OPENALEX_API_KEY`): wired into every
  OpenAlex call site (search, venue-restricted, anchors, bench title
  channel, quota probe). Anonymous pool is 1000 req/day (dies mid-run);
  the key lifts it to 10,000/day — verified live. Console header now
  shows the running version chip.

### Added — run cancellation (v1.4.1)
- **`POST /run/cancel/{id}` + console stop button**: cooperative
  cancellation — the pipeline stops at its next event boundary (a
  blocking LLM/search call adds its own latency), the log closes with a
  visible "已被用户中断" RunFailed, never a silent hang. The raiser is a
  BaseException on purpose so pipeline internals cannot swallow it.
- **Version in the exe filename** (`CiteLens-1.4.1.exe`, unversioned
  `CiteLens.exe` kept for the stable `releases/latest` link) and the
  console header shows the running version — the v1.4.0 exe shipped
  reporting "1.3.3" (frozen-fallback string was never bumped) and was
  indistinguishable on disk; retracted and re-released as v1.4.1.

### Added — external benchmarks (quality yardsticks beyond internal metrics)
- **`citens bench` — LitSearch live-retrieval bench**: samples real
  literature-search questions (Princeton NLP, EMNLP 2024; data snapshot in
  gitignored `bench_data/litsearch/`), runs them through the production
  searchers over live APIs, and scores gold recall@5/@20 (DOI → arXiv →
  title matching) across variants that isolate each lever: per-source,
  score-free union, planner keyword-ification (`--planned`), cheap-model
  listwise rerank (`--llm-rerank`), and the full hybrid agent
  (`--agentic N`, pool-hit rate). Details flush to disk incrementally —
  a killed run keeps every completed query. Measured (15 queries, 3
  sources, seed 13): raw-union 6.7% @20 → planned 20% @20 → agentic
  find-mode 60% pool-hit; S2/OpenAlex return ZERO hits for raw
  natural-language questions, so query formulation is not an optimization
  but the difference between 0% and working retrieval. Two bench-found
  defects fixed: combo queries must be preferred over single-concept
  coverage queries for find-style questions (gold titles ARE concept
  combinations), and multi-query fusion must be RRF, not round-robin (a
  gold at cell-rank ~5 landed fused #29 under interleave).
- **`citens coverage` — AutoSurvey-protocol coverage eval**: the recall
  axis internal metrics never measured. Given a run directory and a human
  survey's reference list (`bench_data/coverage/*.json`, fetched from S2),
  computes survey_recall / core50_recall (the survey's 50 most-cited refs
  — fair to a 20-paper review) / overlap_precision, plus a cheap-tier
  LLM judge scoring coverage-coherence-relevance and naming the missing
  core references. Title matching requires ≥3 informative tokens — short
  numbered titles collapse to a skeleton that matches everything. First
  measurement (agentic run on "LLMs in science" vs a 392-ref survey):
  verifier precision 97.5% but core50_recall 6% and judge coverage 3/10 —
  precision was solved, coverage was the blind spot.

### Changed — harness: falsifiable completion (bench-driven fixes)
- **`anchors` tool**: fetches the topic's most-cited field-defining works
  (OpenAlex, sort=cited_by_count) and reports which are MISSING from the
  pool. Pool size measures effort; anchor overlap measures coverage — the
  two diverged exactly in the coverage eval. Unreachable anchors degrade
  to a warning and never gate done forever.
- **done gate**: the first `done` without an anchors check is bounced
  once with instructions — "enough papers" must be falsified against the
  field's core, not felt. A second done is honored (the model may
  legitimately judge anchors unreachable).
- **`goal="find"` mode** (`HarnessState.goal`): targeted retrieval — the
  question names specific papers, not a survey pool. Pool size stops
  counting as success; done must name the found papers. Measured: 2 →
  8-10 orchestrator turns per question and pool-hit 0% → 60% on the
  LitSearch agentic leg; the failure mode changed from premature
  `budget:pool` exits to "ran out of steps still digging" — a better
  failure, and a tunable one.

### Fixed — bench-found defects
- **Capture-time citation enrichment (the arXiv blind spot, closed)**:
  every arXiv-leg capture carries `citation_count=0` (the arXiv API has
  no citation data), so a classic that ONLY the arXiv leg found still
  sorted as uncited. Before the pool cap, one Semantic Scholar batch
  call now joins arXiv ids → (citationCount, DOI) for zero-cited
  records (verified live: Lewis et al. 2020 → 17,476 citations restored
  from its arXiv id alone). Degrades to a no-op on failure — enrichment
  optimizes ranking, it must never fail a run.
- **`blend_pool` citation spine**: the pre-filter pool cap sorted purely
  within source groups, and an arXiv-leg capture of a field-defining work
  carries `citation_count=0` — measured live: Lewis et al. 2020 (17k
  citations) was pulled into the pool by the harness's exact-title search
  and then arbitrarily cut by the blend, never reaching the filter. Half
  the cap is now reserved for the globally most-cited papers; per-source
  diversity fill takes the rest.
- **`anchors` 429 handling**: OpenAlex rate-limit bursts (anchors runs
  right after a search wave on the same host) were swallowed as "no
  anchor works found" — misleading both the model and the audit trail.
  Now retried with backoff; persistent failure reports honestly as
  unavailable.
- **`read_paper` shows DOIs**: the model needs them to chain snowball
  anchors; without them it guessed (and snowball refused the guess).
- **Bench: planned-leg cell depth 20 → 40**: gold papers ranking 21-40
  in every (query, source) cell were invisible to fusion at zero depth.
  One depth-40 fetch now feeds both legs — truncating cells to 20
  reproduces the depth-20 fusion exactly (RRF only sums ranks), so the
  depth lever is measured at zero extra API cost (`planned_d40`).
- **Bench: find-mode budget widened** (12/14/5 → 16/18/6): both seed-13
  agentic misses ended `budget:steps` mid-dig at 8-10 LLM calls —
  targeted find-that-paper questions need the turns to read candidates
  and name exact titles.
- **find mode ignores the pool cap**: the find-goal prompt says pool
  size is NOT success, yet the loop still hard-stopped at
  `budget:pool` — seed 42's find miss died at 169 papers while still
  reformulating, a direct contradiction of the mode's own instructions.
  `budget:pool` now applies only to survey-mode economics.
- **Bench: `planned_raw` variant**: the question's own phrasing joins
  the RRF fusion (cells already fetched for the union leg — zero extra
  API). Motivated by a measured seed-42 case where union hit 100% via
  crossref title matching while planned hit 0%: the question's words
  were IN the gold title and the planner's combos were not.
- **Snowball: Semantic Scholar fallback for all three directions**:
  OpenAlex free-tier daily budget can exhaust mid-run ("Insufficient
  budget … Resets at midnight UTC"), and a dead OpenAlex used to mean a
  silent zero-candidate snowball. Backward/forward now fall back to the
  S2 citation-graph endpoints, related to S2 recommendations; DOI-style
  arXiv ids are normalized to the `ARXIV:` prefix S2 requires
  (`DOI:10.48550/arxiv.*` 404s). Provenance marks the serving provider.
- **Snowball cache: empty results are never cached**: a provider outage
  wrote zeros into the kv cache, and the retry run served those zeros
  instantly — poisoned. Only non-empty expansions are cached now.

### Changed — retrieval quality round 2 + console polish
- **Adaptive pool: citation-expansion family** (`--expand`): one-hop S2
  citation expansion from the fused pool's own head, as the sixth family
  in the quota pool. A live probe re-tested one-hop reach from
  HyDE-family anchors (the earlier all-zero verdict came from
  wrong-subfield planned anchors) and reached a never-hit gold through a
  forward citation edge — the channel is alive; the anchor quality was
  the problem.
- **HyDE dual-temperature union**: each question now gets a
  deterministic (temp 0) plus a wide (temp 0.9) sample, unioned —
  single-sample coinage was a lottery (a HyDE-caught gold vanished when
  a redraw changed the mix); coverage comes from sampling breadth.
- **Bench cascade on the strong tier** for the deliverable-ranking call
  (LitSearch's rerank gains came from their strongest model).
- **find done-gate verifies named papers**: a find-mode `done` summary
  naming papers absent from the pool (>=half informative-token overlap
  required, so one shared generic word never passes) is bounced once
  with retrieval instructions — bench seed 42 produced confident done
  calls whose named papers were not in the pool.
- **`citens sources --probe`**: one polite request per source, reading
  back the provider's own rate-limit headers and error bodies (OpenAlex
  daily budget/remaining/reset, S2 key presence + status, Crossref
  quota, arXiv latency). Provider limits are server-side accounting;
  proxies change latency, never quotas.
- **Bench cell health**: every (query, source) cell records
  ok/empty/failed into details.jsonl and the console shows `!! DEAD:`
  — an exhausted provider can no longer masquerade as "no results".
- **OpenAlex title.search channel** as a fifth adaptive family: the
  dedicated title index over HyDE-generated titles (endpoint-shaped
  near-exact-title matching); cached, failures never cached.
- **Console (web UI) smoothness**: rAF-batched feed autoscroll with
  unseen-count on the jump-latest button (per-line scrollHeight reads
  forced layout during agent bursts), `content-visibility` on feed
  lines, batched DOM trimming, reduced-motion respect; readability:
  80ch review column, zebra table rows, blockquote/tab-hover/primary-
  button states.

### Added — vocabulary-wall countermeasures (bench-validated)
The bench's central diagnosis (seed 42, 15 queries): gold papers of
find-that-paper questions live in a different lexical space than the
question — question words ∩ gold-title words ≈ ∅, planned keyword cells
never contain the gold even at depth 40, S2 relevance search returned
zero golds on all 15, and one-hop citation/recommendation bridging from
the nearest neighbors reached nothing (all four observation metrics zero
on 7/15). The LitSearch paper's own numbers agree: lexical BM25 and
commercial engines cap ~43% recall@5 where dense semantic retrieval
reaches ~75% — a vocabulary wall. The fixes below are the two proven
ways to cross it plus the semantic layer our API tier can afford:
- **`pivot` harness tool (PRF vocabulary pivot)**: reads the abstracts
  of the pool's most-relevant papers and coins queries in the subfield's
  OWN vocabulary — task/benchmark/method names copied verbatim
  (mechanism verified: mined "SemEval-2022 Task 4" retrieved its gold;
  paraphrased variants do not). Bench-validated 13.3% vs planned 6.7%
  with a superset hit set. Prompt rule 9 tells the orchestrator to reach
  for it exactly when searches return results but the RIGHT papers are
  missing; find-mode suffix teaches the same move.
- **Hypothetical-title queries (`citens.agents.hypothetical`)**: the
  question-side complement (HyDE/query2doc shape): a cheap LLM writes
  plausible TITLES of the answering papers, searched as plain text where
  title matching lives (crossref). Dissection showed near-exact-title
  matching is the ONLY lexical form that reliably crosses the wall;
  hypothetical titles systematize it. Covers descriptive-titled golds;
  pivot covers entity-named golds the question cannot guess.
- **Bench `--adaptive` leg**: hypothetical titles + PRF-mined phrases +
  planned cells + raw question, each FAMILY's fused head guaranteed a
  quota of seats in the candidate pool, then ranked by the LLM listwise
  over that pool. Global RRF alone DILUTES the good cells twice over —
  measured: pivot-only 13.3% → 6.7% fused with planned noise cells, and
  a live probe where union's rank-0 crossref hit fell out of the fused
  top-100 entirely, so the semantic rank never even saw it. The
  family-quota pool plus LLM ranking is the affordable semantic layer
  (no embedding endpoint on this API tier). Reported as `adaptive_rrf`
  (global fusion — the dilution witness) vs `adaptive` (family-quota
  pool + semantic rank) so each lever stays measurable.
- **Shared listwise ranker (`citens.agents.rerank.listwise_rank`)**: the
  judgment both consumers need — bench candidate-pool ranking and
  harness find-mode output. Falls back to input order on any failure;
  never drops papers.
- **find mode ranks its deliverable**: the harness's find output was the
  pool in insertion order — bench seed 42 had the gold IN the pool 3/5
  and in its top-5 0/5. At finish the pool is now pre-sorted by
  (query-match count, citations), LLM-ranked against the question on the
  strong tier (LitSearch's rerank gains came from their strongest model;
  the cheap tier demoted measured golds when judging 100+ candidates),
  with `papers_unranked` keeping the pre-ranking order auditable. Survey
  mode unchanged — the pipeline ranks downstream. Agentic dig budget one
  notch wider (16/18 → 20/22): every agentic miss so far died
  `budget:steps` mid-reformulation, never stalled.

### Added — robustness: model routing, red team, bounded fan-out
- **Three-tier model routing**: `LLM_MODEL_CHEAP` joins `LLM_MODEL` /
  `LLM_MODEL_STRONG`. Mechanical high-volume stages (planner family /
  filter / extract / clarify / intent — 10 call sites) pass `cheap=True`
  and route to the cheap tier; judgment stages (harness orchestrator /
  organize / audit / rewriter) stay default; quality stages stay strong.
  Per-model token accounting (`record_usage`) makes the savings visible
  per run. Empty = current behavior.
- **Red-team adversarial review** (deep mode, once per run): after the
  reflect loop, a strong-model attacker hits the FINAL document for what
  per-claim verification cannot see — overclaims on correlational
  evidence, numbers stronger than their sources, cherry-picking, internal
  contradictions, missing limitations. One bounded revision pass follows
  (hard rules: no citation/number/claim changes; unfixable findings land
  in an honest Limitations section). Findings persist as `09_redteam`;
  fragment revisions are rejected (never silently truncate a review), and
  the revision token budget scales with review length (acceptance found a
  23k-char review overflowing a fixed 8192 cap — correctly refused then,
  budgeted properly now).
- **Bounded search fan-out**: per-source `asyncio.Semaphore(6)` on the
  query fan-out (OpenAlex / S2 / Crossref) — query counts grew with
  concept-block planning + calibration waves, and unbounded bursts are
  how polite APIs decide to throttle.

### Added — agentic retrieval harness (Phase 1)
- **The workflow gains an agent mode**: `--agentic` (CLI) / `agentic: true`
  (API) replaces the three hardcoded retrieval waves (facet gap / zero-hit
  synonyms / thin-pool refine) with a budgeted tool-calling loop — the
  model perceives the pool and drives retrieval itself:
  `search` (multi-source, constraints-aware), `snowball` (citation +
  semantic, query-aware ranked), `pool_report` (composition / facet
  coverage / zero-hits), `read_paper` (title+abstract), `done`.
- **Safety rails**: budget ledger (steps / LLM calls / search calls / pool
  cap, defaults 12/14/5/150), duplicate-search rejection, saturation stop
  (3 consecutive empty searches), two-prose-turn forced finish. Every
  decision lands in the web transcript as detail lines; LLM traces show
  the orchestrator's reasoning.
- First live run (generative-recs topic, DeepSeek function calling):
  pool_report → 4x read_paper → one targeted 3-query search (+36 papers,
  pool 98→134) → pool_report → clean budget exit. Decision log persisted
  as `02h_harness`.
- `llm.chat_tool_call()` joins the LLM layer (multi-turn + tools, retried,
  traced, uncached); `_chat_with_retry(gained return_message=)`.
- Default runs unchanged — the deterministic waves remain the predictable
  path; resume/eval/golden tests unaffected.

### Added — provenance now drives the reflect loop
- Low-yield directions (>=4 pool hits, zero filter survivors — the
  direction is mis-phrased, not merely unlucky) feed the reflect loop two
  ways: the reflector's prompt carries a query-yield note ("include
  REPLACEMENT queries for these directions"), and their concepts' untried
  synonyms join the supplement-query merge deterministically. The
  provenance data collected at filter time now steers the next retrieval
  round instead of only being reported.

### Added — semantic snowball + retrieval provenance (the query-aware layer)
- **Semantic snowball direction**: citation chaining (backward/forward) is
  joined by OpenAlex `related_works` — topical neighbors that are NOT on
  any citation edge. Measured on a generative-recs topic: 4-7 unique
  topical candidates per run entering the fixed top-20 admission window
  that the citation graph alone cannot produce, skewing fresher (median
  citations 314 -> 84 in the admitted set).
- **Query-aware snowball ranking**: candidates rank by topic-term overlap
  before citations — a famous off-topic paper no longer outranks a
  less-cited paper matching the query directions.
- **Retrieval provenance**: every searcher tags papers with the query that
  retrieved them (`matched_queries`, unioned across dedup merges). 98/98
  papers tagged in the live benchmark.
- **Direction-coverage measurement**: `03e_query_yield` per concept block —
  pool hits vs filter survivors from real provenance (the non-neural
  version of query-vector clustering; the concept blocks ARE the
  directions). Surfaces over-fetching junk directions in the transcript.
- **Generic-term anchoring** (found by the benchmark): a concept like
  "survey" or "generative models" searched standalone returns every
  discipline's mega-surveys (measured: SF-36, 30k citations, in a
  generative-recs pool — snowball anchors went medical). Generic terms are
  now anchored to the central concept ("generative recommendation survey"),
  duplicate words collapsing.

### Changed — retrieval planning rebuilt on librarian methodology (concept blocks + calibration)
- **Concept-block planning**: the planner LLM now returns core concepts WITH
  synonym variants (GenRec, generative retrieval, ...), and deterministic
  code assembles the query list from them (one coverage query per concept +
  precision combos of the central pairs). A flat 3-6-word query list misses
  every paper that uses a different variant of the same concept.
- **Zero-hit calibration (PRESS test-search loop)**: per-query hit counts
  flow back from every source; a query that returned nothing gets its
  concept's untried synonyms swapped in as a follow-up wave, and zero-hit
  queries are fed to the refinement prompt ("the field does not use this
  phrasing").
- **Facet queries actually run**: the facet plan used to only *measure*
  coverage after the fact; facets the first round left thin (<3 papers) now
  get their planned queries searched for real (second wave).
- **Native per-source date filters**: the clarification year window compiles
  into OpenAlex `from/to_publication_date`, S2 `year`, Crossref
  `from/until-pub-date` params (arXiv post-filters — no client-friendly date
  syntax). The `"2023..2026"` text hack inside planner queries is gone; it
  only worked by accident on some backends and polluted relevance ranking.
  Supplementary retrieval (_supplement_search) rides the same constraints.
- New artifacts: `01a_query_plan` (concepts + synonyms + assembled queries);
  transcript gains detail lines for both calibration waves.
- `search_round()` (papers, health, per-query stats) joins the search API;
  `SearchSource.set_constraints()` is opt-in for third-party sources.
- Tests: +19 (assembly, fallback, thin-facet selection, native filters per
  source via respx, stats aggregation with failed sources excluded).

## [1.3.3] — 2026-08-23

### Fixed — "records vanished" + history not time-sorted
- A freshly downloaded exe opened an empty workspace: data defaulted to
  "next to the exe", so running a new download from a different folder
  silently started a second, blank data home. Workdir resolution now:
  CITELENS_WORKDIR redirect > exe-folder .env (portable-folder mode,
  existing setups unchanged) > the LAST used workdir (pointer under
  %LOCALAPPDATA%/CiteLens — downloads reattach to their data) > machine
  home on first launch.
- The header now shows the active data directory (📁 chip with tooltip),
  answering "where did my records go" at a glance.
- /runs sorts by the runs' actual timestamps — dir-NAME reverse sort put
  中文 topics in Unicode order (订单簿 chronologically newer runs always
  buried below), and the history entries now show a formatted time
  (MM-DD HH:MM) instead of the raw dir-name tail.

## [1.3.2] — 2026-08-23

### Fixed — full-text harvest landed 0/16 on a real run (generative recs)
- arXiv papers whose ONLY arXiv marker is the DOI (10.48550/arXiv.<id>)
  never entered the direct-pdf fast path — the URL regex expects
  arxiv.org/abs links, S2's openAccessPdf is empty for exactly these
  records, so every one degraded to the title-lookup API (the leg arXiv
  throttles hardest). The DOI (in either field) now parses straight to
  arxiv.org/pdf/<id>.pdf; old-style ids (cs/0301012) included.
- The S2 harvest leg didn't send the configured SEMANTIC_SCHOLAR_API_KEY —
  it rode the anonymous 1-rps pool and 429'd exactly when a run harvests
  many papers in a row (measured miss: a GOLD-OA ACM pdf link).
- Measured on the 0/16 run's actual paper set: **9/16 now fetch
  automatically** (was 0/16); the remaining 7 are bot-walled repositories
  (TechRxiv/SSRN/preprints.org — 403 even with a browser UA) or paywalled
  without an arXiv version — the honest fetch_list.md path.
- The transcript's ground lines now show WHY each paper is abstract-only
  ("arxiv:dl.acm.org HTTP 403", "unpaywall:www.techrxiv.org HTTP 403",
  "无OA候选") — per-leg audit trail from the harvest, visible in the UI.

## [1.3.1] — 2026-08-23

### Fixed — v1.3.0 exe crashed on double-click launch
- The windowed build replaces the absent console streams with a null sink,
  but that sink was a bare write/flush stub; uvicorn's logging formatter
  calls `isatty()` while booting and the AttributeError killed startup
  (`ValueError: Unable to configure formatter 'default'`). The sink is now
  a real `io.TextIOBase` stream (isatty/fileno/encoding/writable), and a
  regression test drives uvicorn's LOGGING_CONFIG through dictConfig with
  the null streams installed — the exact crash path. (Local launch tests
  from a terminal never hit it: GUI-subsystem processes launched FROM a
  console inherit its handles, so stdout wasn't None.)

## [1.3.0] — 2026-08-23

### Fixed — the web console showed NO progress at all during runs
- Root cause: `_event_to_dict` spread `model_dump()` over the event dict,
  and the pydantic `type` field's lowercase Literal ("run_started") clobbered
  the PascalCase class name ("RunStarted") the UI dispatches on — every
  arriving event was silently dropped by the renderer in all 1.2.x releases.
  The wire type is now set after the spread and pinned by tests.
- The SSE stream was never broken (curl always received events); the
  renderer was. 1.3.0 ships that fix plus the transcript below.

### Added — live agent transcript (the code-agent-style activity feed)
- The console's middle column is now an append-only feed: step headers,
  progress lines, and one dim line per LLM call (model · purpose · latency
  · output size), with the model's own thinking (reasoning_content) behind
  a collapsible 💭 toggle on hybrid reasoning models.
- Retrieved content is visible inline: the generated queries, per-source
  hit counts, every selected paper with relevance/citations/quartile,
  fulltext-vs-abstract outcome per paper, and the verify verdict breakdown.
- Transport: `POST /run/start` + `GET /run/events/{id}?after=seq` polling
  (borrowed from deepseek-harness's seq-replay design) replaces fetch-
  streaming for the UI — plain request/response survives proxies/AV that
  buffer text/event-stream, and `after=0` replays the whole transcript.
- Refresh recovery: the run id rides in the URL hash; a reload mid-run
  rebuilds the transcript and keeps polling (the pipeline keeps running
  server-side either way). The SSE `POST /run` endpoint remains for API
  consumers.
- CLI gets the same visibility: per-call dim transcript lines (`⌁`).

### Changed — the exe no longer opens terminal windows
- A onefile console build spawns bootloader-parent + app-child processes,
  and Windows 11's default terminal gives each its own window — every
  launch showed TWO consoles. The build is now windowed (console=False):
  the web console is the entire UI.
- Single instance: launching the exe while a console is already running
  just opens the browser to it (no duplicate invisible servers).
- Exit via the console's new ⏻ 退出 button (POST /shutdown); startup
  crashes land in `error.log` next to the exe for diagnosis.

## [1.2.3] — 2026-08-20

### Changed — pre-run clarification questions follow the review language
- The clarify form (sub-focus / timeframe / venue bar …) renders in
  Chinese by default (REVIEW_LANGUAGE=zh), English with `-l en` — it is
  the first thing a user reads; it was always English regardless of the
  output language. Year-range freshness rewriting already handled 近N年.

## [1.2.2] — 2026-08-20

### Fixed — first-launch experience (the "Failed to fetch" reports)
- Browser opens only when the server actually answers /health — v1.2.1
  opened it 1.5s after launch while the frozen exe was still
  self-extracting (30-180s under antivirus scan), so every request died
  with ERR_CONNECTION_REFUSED / "Failed to fetch".
- The console prints a loading banner IMMEDIATELY (previously blank
  during the heavy import phase), progress dots while binding, and the
  ready line with the URL when up.
- __version__ now follows package metadata (was stuck on 1.1.0).

### Added — provider presets in the settings page
- Quick-select dropdown atop the LLM group: DeepSeek (default, base+
  model prefilled), OpenAI, Ollama (local, no key), OpenRouter —
  choosing one fills base URL + model; on first run DeepSeek is
  preselected so the user only pastes a key, saves, and optionally hits
  "测试连接".

## [1.2.1] — 2026-08-20

### Changed — no more terminal first-run wizard
- Double-click now goes straight to the web console; on first run (no API
  key) the browser opens with the settings page auto-opened and a hint —
  configuring happens entirely in the UI, never in the terminal. The old
  input() wizard blocked startup: while it waited, the console served
  nothing and the browser showed ERR_CONNECTION_REFUSED.
- Startup banner states explicitly that closing the console window quits
  the app; /health reports `llm_configured` for the UI's first-run check.

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
