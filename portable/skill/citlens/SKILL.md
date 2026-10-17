---
name: citlens
description: High-recall literature search and citation-grounded survey writing through the CiteLens MCP tools (server "citlens": search, pool, plan, questions, review, status, doctor). Use for any request to find papers, search the literature, map a research field, check coverage, find surveys on a topic, or write a literature review/survey with real citations. The retrieval method is fixed inside the engine (nine recall families fused by RRF, then a two-stage blind LLM cascade); do not substitute your own web search for it.
---

# CiteLens

`citlens` is a tool, not a prompt. The retrieval method lives in a Python engine
running behind an MCP server; you call it and report what it returns. That is the
whole point: **the tool is what makes the search reproducible** — two different
agents calling `search` with the same topic get the same pool, and an agent that
searched can hand that pool to an agent that writes.

## Tools

Your client prefixes tool names with the server name (`mcp__citlens__search`,
`citlens:search`, `citlens.search` — whatever it shows you). The names are:

| tool | purpose |
|---|---|
| `doctor` | health: engine, drivers, data home, `.env` keys (names only), pool count, jobs |
| `plan` | concept blocks + English queries only; nothing retrieved. Use to test scope |
| `search` | nine-family recall → RRF → two-stage blind cascade rank; writes the pool |
| `pool` | read/filter the pool on disk; no new retrieval |
| `questions` | the survey engine's own clarifying questions for a topic |
| `review` | **opt-in** citation-grounded survey writer |
| `status` | poll a running job: state, elapsed, log tail, artifacts, stats |

If your client has no MCP, the same tools are one shell call away:

```
%USERPROFILE%\.citlens\bin\citlens.cmd search --topic "..." --target 80
%USERPROFILE%\.citlens\bin\citlens.cmd pool --topic "..." --min-citations 50
```
Same JSON, same shared data home. Never reimplement the search yourself when the
CLI or the MCP tool is available.

## The one rule

**Use `search` for literature. Do not answer a literature question from generic
web search or from memory.** The engine exists because single-source search
measurably fails: on a gold-paper benchmark the nine-family adaptive leg reached
recall@20 = 33.3% where planned keyword queries reached 6.7% and a single source
0%. Your own web search is a different, weaker instrument — and a mixture of
both makes coverage unauditable.

Your own search is still useful for things the engine does not do: chasing a
specific named paper, resolving a DOI, skimming a PDF you already have.

## Workflow

1. **`doctor` first** when anything looks wrong (or on a new machine). It tells
   you whether a failure is the tool's environment or the retrieval.
2. **`plan`** (cheap, optional) when the topic is ambiguous or you suspect it is
   too broad. It shows the concept blocks and the exact English queries the
   search will run. Refine the topic before spending a full search.
3. **`search`** with `topic`. Start with `wait_seconds` = 45. A rich topic takes
   2–6 minutes; if you get a `job_id`, poll `status` rather than starting a
   second search. Keep `append_pool` true so the shared pool grows.
4. **Read the evidence before you claim anything.** The result carries
   `artifacts.pool_jsonl`, `artifacts.search_report`, `artifacts.exclusions` and a
   `report_head`. Read `search_report.md` — it has per-family seat counts and
   per-source `ok / empty / failed` health. **A failed source is not a zero-hit
   source.** If `failed > 0`, say which source failed.
5. **`pool`** to iterate without re-searching: `min_citations`, `year_from`,
   `is_review: true` for surveys, `query` for a substring, `sort_by`. This reads
   disk; it never retrieves.
6. **Report numbers, not adjectives.** Kept count, pool size, year spread,
   missing DOI/venue counts, reviews found, and the health table. "Found 47
   papers" is weaker than "九家族共入池 49 → 保留 47，其中综述 9，无 DOI 2；
   web 家族因缺 SERPER_API_KEY 全部 failed".

## Coverage discipline

High recall is not "enough results". Before telling a user a field is covered,
check the shapes that are cheap to check:

- **Anchors.** The most-cited works on the topic should be in the pool. If a
  famous survey or the canonical paper is missing, the search is not done — run
  another `search` with a differently phrased topic, or `plan` and look at which
  concept block is being missed.
- **Surveys.** `pool` with `is_review: true`. A field with zero surveys in a
  large pool usually means the survey family missed, not that no surveys exist.
- **Time.** `pool` returns `by_year`. A pool that stops at 2021 is stale, not
  complete; sources drift.
- **Saturation.** If a second differently-phrased search adds almost nothing new,
  that is evidence of saturation. Say "no new distinct records on the second
  phrasing" rather than "exhaustive".

Never present a subset as the whole field. If you filtered the pool (as the user
asked), say what the filter was and what it removed.

## Writing a review — opt-in only

`review` writes a real, citation-grounded survey. It refuses unless
`user_asked_for_review: true`. Set that flag **only** after the user explicitly
asked you to write a survey/review/article — never because it seems helpful.

Before starting, ask and wait for answers. Call `questions` to get the engine's
own clarifications on top of yours, then ask the user:

1. **Output language** — `zh` or `en`.
2. **Which themes / sub-directions** to centre on (use the engine's `focus`,
   `scope`, `timeframe`, `venue` questions as prompts; never invent the themes).
3. **How many core papers** to deep-read (default 12).
4. **Supporting layer** — one layer of supporting papers (default 15) or off.
5. **Full-text grounding** — on by default; off is faster but the output must be
   labelled abstract-only.

Pass the answers as `language`, `themes`, `papers`, `support_papers`,
`filters: {sub_focus, timeframe, venues}`. Pass the user's topic string
**byte-identical** to the one used for `search` — pools are keyed by a slug of
the topic, and a near-miss silently starts from an empty pool (the tool warns
when the pool is empty: heed it, run `search` first).

Then report the run's own honesty block, verbatim where possible:
`citation_precision`, `pre_rewrite_precision`, `unverifiable_rate`,
`papers_cited/papers_total`, `grounding_with_fulltext/grounding_total`. If
`grounding_with_fulltext` is 0, the review is **abstract-only** — label it.
Repeat verifier health warnings (`verifier_too_lenient`, `judge_model_uncalibrated`)
instead of hiding them. Deliver `review.md` plus `references.bib`.

## Failure modes and what they mean

| symptom | meaning / action |
|---|---|
| `state: running` | normal. Poll `status`; do not start a duplicate |
| `state: lost` | the runner died without an exit state (client restart/kill). Read the artifacts and log tail; the work may be complete |
| `doctor.ok: false` | environment problem — the paths printed are the ones to fix |
| `api_keys_recommended_absent` | degraded *breadth*, not broken: no SERPER → web-bridge family fails (report it); no OPENALEX key → polite pool/throttling |
| `failed` sources in report | report them explicitly; never as zero hits |
| pool empty before `review` | run `search` for the byte-identical topic first |
| `rejected: too many concurrent jobs` | poll the existing jobs; the server caps concurrency deliberately |

## Sharing

All agents on this machine point at one data home, so the pool is shared:
search in one agent, `pool`/`review` from another. Jobs survive the agent that
started them (they are detached), so a long search is not lost when a session
ends. One `~/.citlens/config.json` configures every agent at once.
