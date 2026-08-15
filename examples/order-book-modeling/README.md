# Example output: "订单簿建模" (limit order book modeling)

A real `deep_review` run (13 papers, 2026-08-15), committed as-is so the
README's precision claims are inspectable. Files, in the order a run
produces them:

| file | what it is |
|---|---|
| `review.md` | the survey itself — `[n]` markers refer to the numbered references at the end |
| `references.bib` | BibTeX export of the same reference list |
| `verification.json` | per-claim verdicts (`supported` / `partial` / `unsupported` / `unverifiable`) + the headline **citation precision = (supported + partial) / verifiable** |
| `comparison.md` | cross-paper comparison matrix (study type, evidence level, method rigor) |
| `fetch_list.md` | papers without fetched full text — links for manual download into `papers/` |

Headline numbers for this run: **93.7% citation precision** (65 claims:
36 supported, 23 partial, 4 unsupported, 2 unverifiable). 4 of the 13
papers were found by backward citation snowballing; 2 of the 4
unsupported verdicts were overturned on defense review (verifier
misread the claim's referent).

Regenerate your own with:

```bash
litreview run "limit order book modeling" --mode deep_review -n 13
```

and summarize a set of runs into a comparison table with
`litreview eval --from-runs "runs/<glob>"`.
