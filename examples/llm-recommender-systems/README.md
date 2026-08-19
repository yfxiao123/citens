# Example output: 基于大语言模型的推荐系统 (LLM-based recommender systems)

A real `deep_review` run (2026-08-19, v1.0 pipeline), committed as-is so the
README's claims are inspectable. Topic chosen in the tool's primary target
domain — ACM (RecSys/KDD/SIGIR) and ACL/EMNLP venues.

| file | what it is |
|---|---|
| `review.md` | the survey itself — Chinese body, abstract + keywords, `[n]` markers refer to the numbered references at the end |
| `references.bib` / `.ris` | BibTeX / RIS exports of the same reference list (Zotero/EndNote import) |
| `review_browser.html` | self-contained audit UI — open in any browser: filter claims by verdict, search, expand the evidence chunk behind each claim |
| `verification.json` | per-claim verdicts + headline **citation precision**; `defense_overturned` records the defense pass |
| `provenance.json` | claim → reference → verdict → note, for every claim |
| `grounding.json` | which papers are grounded on full text vs abstract only |
| `comparison.md` | cross-paper comparison matrix (study type, evidence level, method rigor) |
| `fetch_list.md` | the 4 papers with no open PDF — links + suggested filenames for manual fetch into `papers/` |
| `timings.json` | per-stage wall time — the pre-v1.1 baseline this example was timed against |

## Headline numbers

- **Citation precision 99%** (107 cited claims: 50 supported, 46 partial,
  1 unsupported, 11 unverifiable — excluded from the denominator). After the
  rewrite loop converged at 100%, a strict spot-check downgraded 5 lenient
  verdicts and a full `citens reverify` (with the decimal-safe claim parser)
  settled it at 99% — the number you see is the *audited* one.
- **21/24 papers grounded on full text** — arXiv/OA/Unpaywall auto-fetch
  (this run predates the markitdown fix by hours; its 21 came via the
  auto-fetch chain once live). Effect-size numbers only present in full
  texts reached the body: TALLRec's AUC≈0.5 zero-shot baseline and
  128-sample tuning result, Kendall's τ=0.92 for LLM-judge evaluation.
- 47 candidates → 28 kept (20 deep-dive + 8 supporting bibliography), 6
  themes, 31 references, canary honeypot 3/3 caught, wall time 88 min on
  the **pre-speed-package** pipeline (v1.1 targets ~40 min for the same
  shape — `timings.json` is the per-stage receipt).

## What to look at first

1. `review_browser.html` — verdict-colored claims over the full text.
2. `verification.json` → the single `unsupported` claim and its note: the
   verifier's reasoning is recorded, not just the grade.
3. `grounding.json` → the honest split between full-text and abstract-only
   grounding; nothing silently upgraded.
