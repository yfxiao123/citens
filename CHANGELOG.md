# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

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
