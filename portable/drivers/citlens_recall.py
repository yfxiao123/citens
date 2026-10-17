#!/usr/bin/env python
"""citlens_recall.py — the CiteLens nine-family recall leg, as a runnable driver.

This is the executable side of the `citlens` DSH mode. It reimplements, one
family at a time, the adaptive retrieval leg that CiteLens measured as its
highest-recall configuration (`citens/eval/litsearch.py:634-866`), then fuses
with RRF under per-family seat quotas, ranks with the two-stage LLM cascade,
and audits coverage before writing an inspectable run directory.

Nothing here invents a retrieval method: every step calls the `citens`
package's own functions. Run it with the engine's interpreter:

    cd "D:\\Desktop\\personal-knowledge\\citlens test"
    & "D:\\Desktop\\personal-knowledge\\文献检索agent\\.venv\\Scripts\\python.exe" ^
      "D:\\Desktop\\personal-knowledge\\citlens-mode\\tools\\citlens_recall.py" ^
      --topic "generative recommendation with semantic IDs" --target 120

Must be invoked from the engine's data directory: `.env`, `data/litdb`,
`runs/` and `.cache` all resolve against the current working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# --- CiteLens constants, copied verbatim (do not "improve" them) --------------
PER_SOURCE = 20          # per-source cell depth (== the union@20 ceiling)
PLANNED_DEPTH = 40       # planned leg per-cell depth; RRF only sums ranks
FAMILY_QUOTA = 30        # each family's guaranteed seats in the fused pool
DEFAULT_SOURCES = ["arxiv", "semantic_scholar", "openalex", "crossref"]
ALL_FAMILIES = [
    "planned", "raw", "hyde", "pivot", "oa_title",
    "s2_rec", "snowball", "web", "survey",
]


def _log(msg: str) -> None:
    """Never let logging kill the run: a zh-CN console defaults to GBK, and the
    engine emits characters outside it."""
    try:
        print(f"[citlens] {msg}", flush=True)
    except Exception:  # noqa: BLE001
        try:
            sys.stdout.write(f"[citlens] {msg}".encode("ascii", "replace").decode("ascii") + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


async def _cells(
    queries: list[str],
    sources: list[str],
    health: dict,
    depth: int = PER_SOURCE,
) -> dict[str, dict[str, list]]:
    """Per-source result lists for each query — RRF needs cell ranks, not a merge."""
    from citens.eval.litsearch import search_one_query

    out: dict[str, dict[str, list]] = {}
    for q in queries:
        out[q] = await search_one_query(q, sources, per_source=depth, health_out=health)
    return out


def _first_wins(families, quota: int) -> list:
    """Each family contributes its OWN fused head of `quota`; first family wins ties."""
    from citens.eval.litsearch import _key_of

    pool, seen = [], set()
    for fam in families:
        for p in fam[:quota]:
            k = _key_of(p)
            if k not in seen:
                seen.add(k)
                pool.append(p)
    return pool


def _health_summary(health: dict) -> dict:
    return {s: {"ok": b[0], "empty": b[1], "failed": b[2]} for s, b in sorted(health.items())}


async def recall(args) -> dict:
    from citens.agents.hypothetical import hypothetical_queries
    from citens.agents.planner import plan_queries
    from citens.agents.pivot import pivot_from_abstracts
    from citens.agents.rerank import cascade_rank
    from citens.eval.litsearch import (
        fuse_multi_query,
        oa_title_search,
        plan_keyword_queries,
    )
    from citens.search.base import deduplicate
    from citens.search.snowball import _s2_recommended, _s2_to_paper, is_article_doi, snowball
    from citens.search.survey_seed import survey_seed_papers
    # imported explicitly: citens/search/__init__.py never imports websearch, so a
    # `"web"` entry in SEARCH_SOURCES is silently dropped by the registry filter.
    from citens.search.websearch import WebSearchSource, generate_web_query_list

    want = set(args.families)
    topic, sources = args.topic, args.sources
    health: dict[str, list[int]] = {}
    notes: list[str] = []
    report: dict = {"topic": topic, "sources": sources, "families_enabled": sorted(want)}

    # ── Stage A: concept-block planning (PRESS) ───────────────────────────────
    if args.no_llm:
        # genuinely offline: no planner, no LLM-generated families. Only the raw
        # question cell and the (non-LLM) citation expansion are reachable.
        plan = None
        planned_qs = []
        want = {f for f in want if f in {"raw", "snowball"}}
        notes.append("--no-llm：跳过概念块规划与全部 LLM 生成家族，仅 raw + snowball 可用")
    else:
        plan = await asyncio.to_thread(plan_queries, topic, None)
        report["concepts"] = plan.concepts
        report["planned_queries"] = plan.queries
        # family 1 feeds on the planner list sorted LONGEST-first, top 3 — combo-first
        # selection is the measured fix (head-first picked only broad coverage queries).
        planned_qs = await asyncio.to_thread(plan_keyword_queries, topic, 3)
        _log(f"concepts={len(plan.concepts)} queries={len(plan.queries)} planned={planned_qs}")

    families: list[list] = []
    family_sizes: dict[str, int] = {}

    # ── family 1: planned ─────────────────────────────────────────────────────
    planned_cells = await _cells(planned_qs, sources, health, PLANNED_DEPTH) if planned_qs else {}
    if planned_cells:
        fam = fuse_multi_query(planned_cells, sources)
        families.append(fam)
        family_sizes["planned"] = len(fam)
        _log(f"family planned: {len(fam)}")
    else:
        notes.append("planned: 规划器未产出查询，已跳过")

    # ── family 2: raw question ────────────────────────────────────────────────
    raw_cells = await _cells([topic], sources, health, PER_SOURCE)
    raw_fam = fuse_multi_query(raw_cells, sources)
    families.append(raw_fam)
    family_sizes["raw"] = len(raw_fam)
    _log(f"family raw: {len(raw_fam)}")

    planned_raw = fuse_multi_query({**planned_cells, **raw_cells}, sources)

    # ── families 3-4: HyDE titles + PRF pivot entities ────────────────────────
    hyde_qs: list[str] = []
    piv_qs: list[str] = []
    if {"hyde", "pivot"} & want:
        try:
            hyde_qs = [
                q for q in await asyncio.to_thread(hypothetical_queries, topic)
                if q and q.lower() not in {k.lower() for k in planned_cells}
            ]
        except Exception as e:  # noqa: BLE001 — optional channel
            notes.append(f"hyde: 生成失败，跳过（{str(e)[:120]}）")
        try:
            mined = await asyncio.to_thread(pivot_from_abstracts, topic, planned_raw[:10], 4)
            piv_qs = [
                (f'"{q}"' if " " in q and '"' not in q else q)
                for q in mined
                if q.lower() not in {k.lower() for k in planned_cells}
            ]
        except Exception as e:  # noqa: BLE001
            notes.append(f"pivot: 挖掘失败，跳过（{str(e)[:120]}）")
    report["hyde_queries"], report["pivot_queries"] = hyde_qs, piv_qs

    extra_qs = [q for q in dict.fromkeys(hyde_qs + piv_qs) if q.lower() != topic.lower()]
    extra_cells = await _cells(extra_qs, sources, health) if extra_qs else {}
    if "hyde" in want:
        hs = {q.lower() for q in hyde_qs}
        hyde_cells = {q: c for q, c in extra_cells.items() if q.lower() in hs}
        if hyde_cells:
            fam = fuse_multi_query(hyde_cells, sources)
            families.append(fam)
            family_sizes["hyde"] = len(fam)
            _log(f"family hyde: {len(fam)}")
    if "pivot" in want:
        ps = {q.lower() for q in piv_qs}
        piv_cells = {q: c for q, c in extra_cells.items() if q.lower() in ps}
        if piv_cells:
            fam = fuse_multi_query(piv_cells, sources)
            families.append(fam)
            family_sizes["pivot"] = len(fam)
            _log(f"family pivot: {len(fam)}")

    # ── family 5: OpenAlex title index over the two best HyDE titles ─────────
    if "oa_title" in want and hyde_qs:
        try:
            oa_cells = await oa_title_search(hyde_qs[:2])  # 2 probes max: shared daily budget
            if any(oa_cells.values()):
                fam = fuse_multi_query(
                    {f"[title] {t}": {"openalex": cs} for t, cs in oa_cells.items()},
                    ["openalex"],
                )
                families.append(fam)
                family_sizes["oa_title"] = len(fam)
                _log(f"family oa_title: {len(fam)}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"oa_title: 失败，跳过（{str(e)[:120]}）")

    pool = _first_wins(families, FAMILY_QUOTA)
    _log(f"pool after quotas: {len(pool)}")

    # ── family 6: S2 learned recommendations from the pool's own head ────────
    if "s2_rec" in want and pool:
        try:
            rec_dois = [p.doi for p in pool[:10] if is_article_doi(p.doi)][:5]
            items = await _s2_recommended(rec_dois, limit=30) if rec_dois else []
            added = 0
            from citens.eval.litsearch import _key_of
            seen = {_key_of(p) for p in pool}
            for item in items:
                rec = _s2_to_paper(item, "s2-rec", (rec_dois[0] if rec_dois else "")[:30])
                if rec is not None and _key_of(rec) not in seen:
                    seen.add(_key_of(rec))
                    pool.append(rec)
                    added += 1
            family_sizes["s2_rec"] = added
            _log(f"family s2_rec: +{added}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"s2_rec: 失败，跳过（{str(e)[:120]}）")

    # ── family 7: one-hop citation expansion (backward + forward) ────────────
    if "snowball" in want and pool:
        try:
            from citens.eval.litsearch import _key_of
            anchors = [p for p in pool[:8] if is_article_doi(p.doi)][:5]
            if anchors:
                expanded = await snowball(
                    anchors, existing_ids={_key_of(p) for p in pool},
                    backward=True, forward=True, related=False,
                    limit_per_paper=100, top=120,
                )
                seen = {_key_of(p) for p in pool}
                added = 0
                for p in expanded[:FAMILY_QUOTA]:
                    if _key_of(p) not in seen:
                        seen.add(_key_of(p))
                        pool.append(p)
                        added += 1
                family_sizes["snowball"] = added
                _log(f"family snowball: +{added}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"snowball: 失败，跳过（{str(e)[:120]}）")

    # ── family 8: web bridge (English + Chinese corpora) ────────────────────
    if "web" in want:
        try:
            web_qs = await asyncio.to_thread(generate_web_query_list, topic)
            report["web_queries"] = web_qs
            if web_qs:
                web_papers = await WebSearchSource().search(web_qs, 16)
                from citens.eval.litsearch import _key_of
                seen = {_key_of(p) for p in pool}
                added = 0
                for p in web_papers[:FAMILY_QUOTA]:
                    if _key_of(p) not in seen:
                        seen.add(_key_of(p))
                        pool.append(p)
                        added += 1
                family_sizes["web"] = added
                _log(f"family web: +{added}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"web: 失败，跳过（{str(e)[:120]}）")

    # ── family 9: survey reference harvest (human-curated canon) ────────────
    if "survey" in want and pool:
        try:
            from citens.eval.litsearch import _key_of
            survey_papers, seeds = await survey_seed_papers(
                topic, pool[:40], planner=planned_qs, quota=FAMILY_QUOTA
            )
            report["survey_seeds"] = seeds
            seen = {_key_of(p) for p in pool}
            added = 0
            for p in survey_papers:
                if _key_of(p) not in seen:
                    seen.add(_key_of(p))
                    pool.append(p)
                    added += 1
            family_sizes["survey"] = added
            _log(f"family survey: +{added}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"survey: 失败，跳过（{str(e)[:120]}）")

    # ── Stage D: two-stage semantic cascade (the "high confidence" step) ────
    pool = deduplicate(pool)
    if args.no_llm:
        notes.append("cascade_rank 已跳过（--no-llm）：顺序为家族保底席位顺序")
        ranked = pool
    else:
        try:
            ranked = await asyncio.to_thread(cascade_rank, topic, pool, 50, True)
            _log(f"cascade_rank: {len(pool)} -> {len(ranked)}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"cascade_rank 失败，退化为家族顺序（{str(e)[:120]}）")
            ranked = pool

    if args.target:
        ranked = ranked[: args.target]

    report.update(
        family_sizes=family_sizes,
        pool_size=len(pool),
        kept=len(ranked),
        health=_health_summary(health),
        notes=notes,
    )
    # Everything the families put in the pool but the target cut is an EXCLUSION
    # and must be logged with a reason: "pool size" is not a coverage claim.
    return {"report": report, "papers": ranked, "rejected": pool}


def write_run(out_root: Path, topic: str, result: dict) -> Path:
    from citens.grounding.fulltext import slugify

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{slugify(topic)}-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    papers = result["papers"]
    report = result["report"]

    with (run_dir / "pool.jsonl").open("w", encoding="utf-8") as fh:
        for p in papers:
            fh.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")

    # PRISMA-style exclusion log: everything retrieved but not kept, with a reason
    kept_keys = {p.id for p in papers}
    with (run_dir / "exclusions.jsonl").open("w", encoding="utf-8") as fh:
        for p in result.get("rejected", []):
            if p.id in kept_keys:
                continue
            fh.write(json.dumps({
                "title": p.title, "doi": p.doi, "source": p.source,
                "reason": "ranked_below_target", "citation_count": p.citation_count,
            }, ensure_ascii=False) + "\n")

    fam_rows = "\n".join(
        f"| {k} | {v} |" for k, v in report.get("family_sizes", {}).items()
    ) or "| （无） | 0 |"
    health_rows = "\n".join(
        f"| {k} | {v['ok']} | {v['empty']} | {v['failed']} |"
        for k, v in report.get("health", {}).items()
    ) or "| （无） | 0 | 0 | 0 |"
    notes = "\n".join(f"- {n}" for n in report.get("notes", [])) or "- （无）"
    missing_venue = sum(1 for p in papers if not p.venue)
    no_doi = sum(1 for p in papers if not p.doi)
    reviews = sum(1 for p in papers if p.is_review)

    (run_dir / "search_report.md").write_text(f"""# CiteLens 检索报告 — {topic}

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
检索源：{', '.join(report.get('sources', []))}
启用家族：{', '.join(report.get('families_enabled', []))}

## 结果

- 候选池：{report.get('pool_size', 0)} 篇 → 保留 **{len(papers)}** 篇
- 其中综述 {reviews} 篇 · 无 DOI {no_doi} 篇 · 无 venue {missing_venue} 篇
- 收工条件未自动判定：请按 SKILL.md 阶段 F 跑锚点核验与饱和检查后再声明完成

## 家族 → 入池条数

| 家族 | 入池 |
|---|---|
{fam_rows}

## 逐源健康（ok / empty / failed，按检索 cell 计）

| 源 | ok | empty | failed |
|---|---|---|---|
{health_rows}

> failed > 0 时必须显式报告：源失败绝不能写成"零命中"。

## 概念块（阶段 A）

{json.dumps(report.get('concepts', []), ensure_ascii=False, indent=1)}

## 计划查询

{chr(10).join('- ' + q for q in report.get('planned_queries', [])) or '- （无）'}

## HyDE 假想标题

{chr(10).join('- ' + q for q in report.get('hyde_queries', [])) or '- （无）'}

## pivot 挖掘实体（已加引号成短语）

{chr(10).join('- ' + q for q in report.get('pivot_queries', [])) or '- （无）'}

## web 桥接查询（英文 + 中文）

{chr(10).join('- ' + q for q in report.get('web_queries', [])) or '- （无）'}

## 综述种子（参考文献收割锚点）

{chr(10).join('- ' + q for q in report.get('survey_seeds', [])) or '- （无）'}

## 备注与降级

{notes}

## 下一步（未完成即不算收工）

1. **锚点核验**：OpenAlex 取该主题被引最高工作，报告池内缺失名单（`anchors.json`）。
2. **召回审计**：`citens collect "<topic>" -n {max(len(papers), 60)} --audit-recall`，
   用 top 综述参考文献算 `coverage`。
3. **排除日志**：确认 `exclusions.jsonl` 里每条都有原因码。
4. **全文清单**：对无全文的论文写 `fetch_list.md`（DOI + 建议文件名），
   拿不到全文的标注"仅摘要级"。
""", encoding="utf-8")
    return run_dir


def build_args(argv=None):
    ap = argparse.ArgumentParser(description="CiteLens nine-family recall leg")
    ap.add_argument("--topic", required=True, help="研究主题（中英皆可；检索词会被翻成英文）")
    ap.add_argument("--target", type=int, default=0, help="保留条数上限（0 = 不截断）")
    ap.add_argument("--sources", default="", help="逗号分隔检索源；默认取 SEARCH_SOURCES")
    ap.add_argument("--families", default=",".join(ALL_FAMILIES),
                    help="逗号分隔家族；可选 " + ",".join(ALL_FAMILIES)
                    + "（planned/raw 是主干：始终执行，不受此开关限制）")
    ap.add_argument("--out", default="citlens-runs", help="报告输出根目录")
    ap.add_argument("--no-llm", action="store_true",
                    help="完全离线：跳过概念块规划、全部 LLM 生成家族与级联排序，仅 raw + snowball")
    ap.add_argument("--append-pool", action="store_true",
                    help="把结果并入 CiteLens 文献池 data/litdb/<topic>.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="只做规划并打印，不发检索请求")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    _force_utf8_stdio()
    args = build_args(argv)
    if args.sources.strip():
        args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        try:
            from citens.config import settings
            args.sources = [s.strip() for s in settings.search_sources.split(",") if s.strip()]
        except Exception:  # noqa: BLE001
            args.sources = list(DEFAULT_SOURCES)
    args.families = [f.strip() for f in args.families.split(",") if f.strip()]

    if args.dry_run:
        from citens.agents.planner import plan_queries
        from citens.eval.litsearch import plan_keyword_queries
        plan = plan_queries(args.topic, None)
        print(json.dumps({
            "concepts": plan.concepts,
            "queries": plan.queries,
            "planned_top3_longest_first": plan_keyword_queries(args.topic, 3),
            "sources": args.sources,
            "families": args.families,
        }, ensure_ascii=False, indent=1))
        return 0

    t0 = time.time()
    result = asyncio.run(recall(args))
    run_dir = write_run(Path(args.out), args.topic, result)
    papers = result["papers"]
    _log(f"kept {len(papers)} papers -> {run_dir}")

    if args.append_pool:
        from citens.collect import append_pool
        added = append_pool(args.topic, papers)
        _log(f"appended {added} new records to the literature pool")

    _log(f"elapsed {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
