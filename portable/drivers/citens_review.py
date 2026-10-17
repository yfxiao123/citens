#!/usr/bin/env python
"""citens_review.py — write a citation-grounded survey, opt-in.

This is the SECOND half of the `citlens` mode and it only runs when the user
explicitly asked for a survey. It calls CiteLens' own production pipeline
(`citens.orchestration.run_pipeline`), so the whole chain is the real one:

    pool pre-recall (BM25 + vector RRF) -> rank -> filter -> enrich -> extract
    -> organize -> synthesize -> write ([n] claims) -> verify (5-grade, vs
    ground text) -> reflect -> supplement -> recompose

The literature pool built by `citlens_recall.py` (or `citens collect`) is
injected automatically when the topic slug matches, which is what makes
"search first, then write" one coherent flow instead of two unrelated runs.

Run with the engine's interpreter from the engine's data directory:

    & "...\\文献检索agent\\.venv\\Scripts\\python.exe" citens_review.py `
      --topic "generative recommendation with semantic IDs" `
      --language zh --papers 12 --mode deep_review `
      --filter sub_focus="SID tokenization" --filter timeframe="last 5 years"

Use --questions first to see the clarifying questions CiteLens itself would ask
for this topic; the mode relays those (plus language/depth) to the user.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _force_utf8_stdio() -> None:
    """The pipeline emits non-GBK characters (e.g. "\u26a0") into progress
    messages. On a zh-CN console stdout defaults to GBK, and a raise inside our
    event handler would propagate INTO the pipeline and fail the run — so make
    stdio UTF-8 first, and never let a print kill the pipeline."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 — non-reconfigurable stream is fine
            pass


def _say(msg: str, **_ignored) -> None:
    """Tolerant logger: accepts (and ignores) print-style kwargs such as
    flush=, because one of these calls sits inside the pipeline's event
    handler where an exception would abort the run itself."""
    try:
        print(msg, flush=True)
    except Exception:  # noqa: BLE001 — logging must never break the pipeline
        try:
            sys.stdout.write(msg.encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def print_questions(topic: str) -> int:
    """Print the clarification questions the engine would ask, as JSON."""
    from citens.agents.clarify import generate_clarifying_questions

    try:
        qs = generate_clarifying_questions(topic)
    except Exception as e:  # noqa: BLE001
        _say(json.dumps({"error": str(e)[:200], "questions": []}, ensure_ascii=False, indent=1))
        return 1
    _say(json.dumps({"questions": qs}, ensure_ascii=False, indent=1))
    return 0


def run(args) -> int:
    from citens.config import settings
    from citens.events import EventBus
    from citens.models import RunMode
    from citens.orchestration import RunOptions, run_pipeline

    settings.review_language = args.language
    if args.concurrency:
        settings.llm_concurrency = args.concurrency

    filters: dict[str, str] = {}
    for item in args.filter:
        if "=" not in item:
            _say(f"[citens-review] --filter needs k=v, got {item!r}")
            return 2
        k, v = item.split("=", 1)
        filters[k.strip()] = v.strip()
    if args.filters_json:
        filters.update(json.loads(args.filters_json))

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    mode = RunMode(args.mode) if args.mode else None

    options = RunOptions(
        max_papers=args.papers,
        max_results=args.max_results,
        support_papers=args.support_papers,
        sources=sources,
        profile=args.profile,
        fetch_fulltext=not args.no_fulltext,
        use_pool=not args.no_pool,
        allow_supplement=not args.no_supplement,
        agentic_retrieval=args.agentic,
        mode=mode,
        filters=filters,
    )

    # show the pool this run will inject, so "search first" is verifiable
    try:
        from citens.collect import pool_path, read_pool

        pp = pool_path(args.topic)
        n_pool = len(read_pool(args.topic)) if pp.is_file() else 0
        _say(f"[citens-review] pool: {pp} ({n_pool} records)"
              + ("" if n_pool else "  <- EMPTY: run the search leg first for a real breadth gain"))
    except Exception as e:  # noqa: BLE001
        _say(f"[citens-review] pool lookup failed: {str(e)[:120]}")

    _say(f"[citens-review] topic={args.topic!r} lang={args.language} mode={args.mode or 'auto'} "
          f"papers={args.papers} fulltext={not args.no_fulltext} filters={filters}")
    _say(f"[citens-review] started {time.strftime('%Y-%m-%d %H:%M:%S')}")

    done: dict = {}

    def on_event(ev) -> None:
        t = ev.type
        if t == "step_started":
            _say(f"[step] {ev.title or ev.step}")
        elif t == "step_progress":
            if ev.message:
                _say(f"   {ev.message}")
        elif t == "step_completed":
            _say(f"[ok]   {ev.step}: {ev.message}")
        elif t == "llm_trace":
            if args.verbose and ev.phase == "end":
                _say(f"   [llm] {ev.purpose or ev.stage} {ev.ms}ms in={ev.chars_in} out={ev.chars_out}")
        elif t == "run_failed":
            _say(f"[FAIL] {ev.step}: {ev.message}")
        elif t == "run_completed":
            done.update(run_dir=ev.run_dir, review_path=ev.review_path, summary=ev.summary or {})

    bus = EventBus()
    bus.subscribe(on_event)

    t0 = time.time()
    try:
        run_pipeline(args.topic, options, bus)
    except Exception as e:  # noqa: BLE001
        _say(f"[citens-review] pipeline raised: {str(e)[:300]}")
        return 1

    run_dir = Path(done.get("run_dir") or "")
    _say(f"\n[citens-review] finished in {time.time() - t0:.0f}s")
    _say(f"[citens-review] run_dir: {run_dir}")
    _say(f"[citens-review] review:  {done.get('review_path') or (run_dir / 'review.md')}")

    ver = _read_json(run_dir / "verification.json") or {}
    grd = _read_json(run_dir / "grounding.json") or {}
    cov = _read_json(run_dir / "coverage_eval.json") or {}
    meta = _read_json(run_dir / "meta.json") or {}
    bib = run_dir / "references.bib"
    review = run_dir / "review.md"

    _say("\n[citens-review] ---- honesty report ----")
    _say(f"  papers: pool {meta.get('total_papers', '?')} -> filtered {meta.get('filtered_papers', '?')}"
          f" | references.bib {'present' if bib.is_file() else 'MISSING'}")
    if ver:
        verifiable = ver.get("verifiable_claims")
        _say(f"  claims: {ver.get('total_claims')} total / {verifiable} verifiable"
              f" | supported {ver.get('supported')} partial {ver.get('partial')}"
              f" unsupported {ver.get('unsupported')}")
        _say(f"  citation precision: {ver.get('citation_precision')}"
              f" (pre-rewrite {ver.get('pre_rewrite_precision')}, verifier {ver.get('verifier_precision')})")
        _say(f"  unverifiable_rate: {ver.get('unverifiable_rate')}"
              f" | papers cited {ver.get('papers_cited')}/{ver.get('papers_total')}")
    if grd:
        _say(f"  grounding: full text {grd.get('with_fulltext')}/{grd.get('total')}"
              + ("  <- ABSTRACT-ONLY: label it, never call it verified"
                 if not grd.get("with_fulltext") else ""))
    if cov:
        _say(f"  coverage vs human survey: survey_recall {cov.get('survey_recall')}"
              f" core50_recall {cov.get('core50_recall')} overlap_precision {cov.get('overlap_precision')}")
    if review.is_file():
        _say(f"  review.md: {review.stat().st_size} bytes")
    _say("[citens-review] artifacts: review.md references.bib references.ris provenance.json "
         "verification.json grounding.json fetch_list.md steps/*.json")
    return 0


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="CiteLens survey writer (opt-in second half of the citlens mode)")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--language", default="zh", choices=["zh", "en"], help="综述输出语言（中文/English）")
    ap.add_argument("--papers", "-n", type=int, default=12, help="核心精读篇数（决定主题深度）")
    ap.add_argument("--max-results", type=int, default=None, help="候选池目标数")
    ap.add_argument("--support-papers", type=int, default=None, help="支撑文献层数量（默认 15，0 关闭）")
    ap.add_argument("--mode", default="deep_review", choices=["quick_scan", "deep_review", "interactive"])
    ap.add_argument("--sources", default="", help="逗号分隔检索源")
    ap.add_argument("--profile", default="", help="领域 profile（如 finance）")
    ap.add_argument("--filter", action="append", default=[], metavar="K=V",
                    help="跑前澄清约束（可重复）：sub_focus=… timeframe=… venues=… themes=…")
    ap.add_argument("--filters-json", default="", help="约束的 JSON 形式（与 --filter 合并）")
    ap.add_argument("--no-fulltext", action="store_true", help="不抓全文（快，但引用精度显著下降）")
    ap.add_argument("--no-pool", action="store_true", help="不注入先前的文献池")
    ap.add_argument("--no-supplement", action="store_true", help="跳过补检循环")
    ap.add_argument("--agentic", action="store_true", help="智能体化自适应检索")
    ap.add_argument("--concurrency", "-c", type=int, default=None)
    ap.add_argument("--questions", action="store_true", help="只打印引擎会问的澄清问题后退出")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    _force_utf8_stdio()
    args = build_args(argv)
    if args.questions:
        return print_questions(args.topic)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
