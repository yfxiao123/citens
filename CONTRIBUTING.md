# Contributing

Thanks for considering a contribution!

## Development setup

```bash
git clone <repo> && cd litreview
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,api,pdf]"
cp .env.example .env    # then fill in your LLM key
```

## Before submitting

```bash
ruff check litreview tests     # lint (line length 100)
pytest tests/ -q               # unit tests — no network, no LLM calls
```

Both must pass. New features need a test in `tests/` (pure-logic tests; mock
anything that touches the network or an LLM).

## Conventions

- **Every claim the writer produces must stay grounded.** If you touch the
  writer or verifier, rerun an end-to-end topic and include the resulting
  citation precision in your PR description.
- **Search sources are pluggable** (`litreview/search/`): subclass
  `SearchSource`, `@register` it, and add a test with a mocked HTTP response.
- **No secrets in code or tests.** Keys live in `.env` only.
- Keep the pipeline runnable at every commit — avoid landing half-stages.

## Reporting issues

Include: the topic, the run directory (`runs/<topic>-<ts>/`) or at least
`meta.json`, and whether you had `PAPERS_DIR`/SJR data configured. Bug reports
without a reproducer will be asked for one.

## License

By contributing you agree that your contributions are licensed under the MIT
license (see [LICENSE](LICENSE)). The SCImago SJR dataset fetched by
`litreview sjr` is CC BY-NC and must never be committed to the repository.
