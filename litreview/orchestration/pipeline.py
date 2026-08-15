"""Pipeline orchestrator.

Stages:
    planner -> search -> filter -> extract
            -> compose(organize -> synthesize -> write -> verify)
            -> [reflect -> supplement -> compose]   (one bounded loop)

Emits structured events for subscribers (CLI renderer, FastAPI SSE). Writes
every intermediate artifact to a run-directory so runs are inspectable.

The "compose" half is factored into :func:`_compose` so the reflect loop can
re-run it on the augmented paper set without duplicating logic.
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field

from litreview import cache, persistence
from litreview.agents import (
    audit_coverage,
    check_health,
    detect_intent,
    extract_papers,
    filter_papers,
    generate_keywords,
    missing_to_queries,
    organize_themes,
    reflect,
    review_unsupported_claims,
    synthesize,
    verify_claims,
    write_review_body,
)
from litreview.agents.planner import refine_queries
from litreview.agents.quality import build_comparison_matrix, render_comparison_md
from litreview.agents.writer import localized_heading
from litreview.config import settings
from litreview.events import (
    Event,
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
    StepStarted,
)
from litreview.grounding import (
    ChunkStore,
    CitationTable,
    build_provenance,
    enrich_abstracts,
    parse_claims_from_review,
    write_fetch_list,
)
from litreview.models import (
    ClaimIntentManifest,
    ExtractedPaper,
    RunMeta,
    RunMode,
    ThemeStructure,
)
from litreview.ranking import quartile_histogram, rank_papers
from litreview.search import blend_pool, search_papers
from litreview.search.snowball import snowball


@dataclass
class RunOptions:
    max_results: int | None = None
    max_papers: int | None = None
    sources: list[str] | None = None
    use_cache: bool = True
    allow_supplement: bool = True
    max_supplement_papers: int = 4
    fetch_fulltext: bool = True
    enrich_abstracts: bool = True
    extra: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)  # pre-run clarifications (see clarify.py)
    mode: RunMode | None = None  # adaptive mode (auto-detected if None)

    def resolved_max_results(self) -> int:
        return self.max_results or settings.default_max_results

    def resolved_max_papers(self) -> int | None:
        if self.max_papers is None:
            mp = settings.default_max_papers
            return mp or None
        return self.max_papers or None


@dataclass
class ComposeResult:
    themes: ThemeStructure
    review_path: str
    table: CitationTable
    claims: list
    precision: float
    synthesis: object


def _emit(bus: EventBus | None, event: Event) -> None:
    if bus is not None:
        bus.emit(event)


def _compose(
    extracted: list[ExtractedPaper],
    topic: str,
    run_dir: str,
    bus: EventBus | None,
    *,
    fetch_fulltext: bool = True,
    on_supplement_queries=None,
) -> ComposeResult:
    """organize -> synthesize -> write -> verify, persisting all artifacts.

    Returns the final table / claims / precision so the outer flow can decide
    on reflection and assemble RunMeta. `on_supplement_queries`, if given, is
    invoked with (queries, message) when verification finds unsupported claims
    that could be resolved by targeted retrieval (the caller decides whether
    to actually supplement).
    """
    # organize
    _emit(bus, StepStarted(step="organize", title="主题组织"))
    themes = organize_themes(extracted, topic)
    persistence.save_step(run_dir, "05_themes", themes)
    _emit(bus, StepCompleted(step="organize", message=f"{len(themes.themes)} 个主题"))

    # synthesize (critical cross-paper analysis)
    _emit(bus, StepStarted(step="synthesize", title="批判性综合"))
    synthesis = synthesize(extracted, themes, topic)
    persistence.save_step(run_dir, "06_synthesis", synthesis)
    _emit(
        bus,
        StepCompleted(
            step="synthesize",
            message=(
                f"共识 {len(synthesis.consensus)} · 矛盾 {len(synthesis.contradictions)} · "
                f"空白 {len(synthesis.gaps)}"
            ),
        ),
    )

    # grounding: fetch full text where available (the main precision lever)
    _emit(bus, StepStarted(step="ground", title="获取全文（溯源）"))

    def _ground_progress(i, total, title):
        _emit(bus, StepProgress(step="ground", message=title, current=i, total=total))

    chunk_store = ChunkStore()
    chunk_store.build_from(extracted, fetch_full=fetch_fulltext, on_progress=_ground_progress)
    table = CitationTable(extracted)
    n_full = sum(
        1
        for p in extracted
        if any(c.kind.value == "fulltext" for c in chunk_store.chunks_for(p.id))
    )
    persistence.save_json(
        run_dir,
        "grounding.json",
        {
            "with_fulltext": n_full,
            "total": len(extracted),
            "papers": [
                {
                    "index": i,
                    "title": p.title,
                    "has_fulltext": any(
                        c.kind.value == "fulltext" for c in chunk_store.chunks_for(p.id)
                    ),
                    "n_chunks": len(chunk_store.chunks_for(p.id)),
                }
                for i, p in enumerate(extracted)
            ],
        },
    )
    # papers still missing full text -> fetch list for manual (browser) download
    missing = [
        p
        for p in extracted
        if not any(c.kind.value == "fulltext" for c in chunk_store.chunks_for(p.id))
    ]
    fetch_list_path = write_fetch_list(run_dir, missing)
    ground_msg = f"{n_full}/{len(extracted)} 篇获取到全文"
    if fetch_list_path:
        ground_msg += f" · {len(missing)} 篇待手动获取（见 fetch_list.md）"
    _emit(
        bus,
        StepCompleted(step="ground", message=ground_msg),
    )

    # write
    _emit(bus, StepStarted(step="write", title="综述撰写"))

    def _write_step(name, label=""):
        _emit(bus, StepProgress(step="write", message=label or name))

    body = write_review_body(extracted, themes, topic, synthesis=synthesis, on_step=_write_step)
    review = body + f"\n## {localized_heading('refs')}\n\n" + table.references_md() + "\n"
    review_path = persistence.save_text(run_dir, "review.md", review)
    persistence.save_text(run_dir, "references.bib", table.to_bibtex())

    claims = parse_claims_from_review(review)
    persistence.save_step(run_dir, "07_claims", claims)
    _emit(
        bus,
        StepCompleted(step="write", message=f"综述生成完成 · {len(claims)} 条带引用论断"),
    )

    # claim intent manifest: map claims to synthesis intents
    intent_to_claims = {}
    for i, claim in enumerate(claims):
        # Each claim implicitly supports one of the synthesis themes
        # For now, use a simple heuristic: map to the first consensus item mentioned
        for j, consensus in enumerate(synthesis.consensus):
            if any(word.lower() in claim.text.lower() for word in consensus.split()[:3]):
                intent_key = f"consensus_{j}"
                if intent_key not in intent_to_claims:
                    intent_to_claims[intent_key] = []
                intent_to_claims[intent_key].append(i)
                break
        else:
            # Default to gaps or contradictions if no consensus match
            if synthesis.gaps:
                intent_key = "gap_0"
            elif synthesis.contradictions:
                intent_key = "contradiction_0"
            else:
                intent_key = "general"
            if intent_key not in intent_to_claims:
                intent_to_claims[intent_key] = []
            intent_to_claims[intent_key].append(i)
    
    claim_manifest = ClaimIntentManifest(
        intents=synthesis.consensus + synthesis.gaps + synthesis.contradictions,
        intent_to_claims=intent_to_claims
    )
    persistence.save_step(run_dir, "07_claim_manifest", claim_manifest.model_dump())

    # verify (citation precision)
    ver_results = []
    precision = 0.0
    if claims:
        _emit(bus, StepStarted(step="verify", title="引用核验"))

        def _verify_progress(i, total):
            _emit(
                bus,
                StepProgress(step="verify", message=f"核验论断 {i}/{total}", current=i, total=total),
            )

        ver_results, precision = verify_claims(
            claims, table, chunk_store, on_progress=_verify_progress
        )
        persistence.save_json(
            run_dir,
            "verification.json",
            {
                "citation_precision": round(precision, 3),
                "total_claims": len(claims),
                "verifiable_claims": sum(
                    1 for r in ver_results if r.verdict.value != "unverifiable"
                ),
                "supported": sum(1 for r in ver_results if r.verdict.value == "supported"),
                "partial": sum(1 for r in ver_results if r.verdict.value == "partial"),
                "unsupported": sum(1 for r in ver_results if r.verdict.value == "unsupported"),
                "unverifiable": sum(
                    1 for r in ver_results if r.verdict.value == "unverifiable"
                ),
                "results": [r.model_dump() for r in ver_results],
            },
        )
        _emit(bus, StepCompleted(step="verify", message=f"引用精度 {precision * 100:.0f}%"))

        # bidirectional verification: defense lawyer challenges unsupported verdicts
        unsupported_count = sum(1 for r in ver_results if r.verdict.value == "unsupported")
        if unsupported_count > 0:
            _emit(bus, StepStarted(step="defense", title="双向核验（辩护律师）"))
            source_contexts = {}
            for i, claim in enumerate(claims):
                chunks = chunk_store.chunks_for(claim.citation_indices[0] if claim.citation_indices else "")
                source_contexts[i] = "\n\n".join(c.text for c in chunks[:3])
            
            defense_reviews = review_unsupported_claims(claims, ver_results, source_contexts)
            overturned = sum(1 for r in defense_reviews if r["overturned"])
            
            persistence.save_step(run_dir, "08_defense", defense_reviews)
            _emit(
                bus,
                StepCompleted(
                    step="defense",
                    message=f"辩护律师审查 {len(defense_reviews)} 条 unsupported，推翻 {overturned} 条"
                )
            )

        # health monitoring: detect systematic biases
        _emit(bus, StepStarted(step="health", title="对话健康监测"))
        theme_paper_counts = {theme.name: len(theme.paper_indices) for theme in themes.themes}
        absence_audit = audit_coverage(topic, [p.title for p in extracted])
        health_report = check_health(synthesis, ver_results, absence_audit, theme_paper_counts)
        persistence.save_step(run_dir, "08_health", health_report)
        
        issues = health_report.get("issues", [])
        if issues:
            _emit(
                bus,
                StepCompleted(
                    step="health",
                    message=f"检测到 {len(issues)} 个系统性偏差: {', '.join(issues)}"
                )
            )
        else:
            _emit(bus, StepCompleted(step="health", message="管线健康，未检测到系统性偏差"))

        # verifier feedback -> targeted supplementary retrieval for
        # unsupported claims (the "unsupported means missing evidence" loop)
        if on_supplement_queries:
            from litreview.agents.verifier_trigger import collect_unsupported_queries

            vq = collect_unsupported_queries(claims, ver_results, topic)
            if vq:
                persistence.save_step(run_dir, "08b_verify_trigger", vq)
                _emit(
                    bus,
                    StepProgress(
                        step="verify",
                        message=f"核验发现 {len(vq)} 条待补充检索（unsupported 论断溯源）",
                    ),
                )
                on_supplement_queries(vq, f"{len(vq)} 条核验触发补充检索")

    persistence.save_json(run_dir, "provenance.json", build_provenance(claims, table, ver_results))

    return ComposeResult(
        themes=themes,
        review_path=review_path,
        table=table,
        claims=claims,
        precision=precision,
        synthesis=synthesis,
    )


async def _supplement_search(
    keywords: list[str],
    existing_ids: set[str],
    options: RunOptions,
    topic: str,
) -> tuple[list, bool]:
    """Gap-targeted supplementary retrieval -> filter -> new papers only.

    Returns (fresh_papers, all_known): all_known=True means relevant papers
    WERE found but every one is already in the pool (typical on cache replay
    after the pool changed) — different from a genuine no-hit.
    """
    cache_key = {"keywords": keywords, "max_results": 20, "sources": options.sources, "supplement": True}
    papers = cache.get("search", cache_key) if options.use_cache else None
    if papers is None:
        papers = await search_papers(keywords, max_results=20, sources=options.sources)
        if options.use_cache:
            cache.put("search", cache_key, [p.model_dump() for p in papers])
    else:
        from litreview.models import Paper

        papers = [Paper(**p) for p in papers]
    papers = blend_pool(papers, cap=8)
    scored = filter_papers(papers, topic, filters=options.filters)
    fresh = [p for p in scored if p.id not in existing_ids][: options.max_supplement_papers]
    all_known = len(fresh) == 0 and len(scored) > 0
    return fresh, all_known


async def run_pipeline_async(
    topic: str,
    options: RunOptions | None = None,
    bus: EventBus | None = None,
) -> RunMeta:
    """Run the full pipeline asynchronously. Raises on unrecoverable failure."""
    options = options or RunOptions()
    max_results = options.resolved_max_results()
    max_papers = options.resolved_max_papers()

    run_dir = persistence.new_run_dir(topic)
    meta = RunMeta(topic=topic, run_dir=run_dir)
    _emit(bus, RunStarted(topic=topic))

    try:
        # Step 0: intent detection (auto-detect mode if not specified)
        if options.mode is None:
            _emit(bus, StepStarted(step="intent", title="检测用户意图"))
            mode_str = detect_intent(topic, options.filters)
            options.mode = RunMode(mode_str)
            persistence.save_step(run_dir, "00_intent", {"mode": mode_str})
            _emit(bus, StepCompleted(step="intent", message=f"运行模式: {mode_str}"))
        
        # Apply mode-specific defaults
        if options.mode == RunMode.QUICK_SCAN:
            options.allow_supplement = False  # skip reflection for quick scans
            if options.max_papers is None:
                options.max_papers = 5  # fewer papers for quick overview

        # Step 1: keywords
        _emit(bus, StepStarted(step="planner", title="生成检索关键词"))
        if options.filters:
            persistence.save_step(run_dir, "00_filters", options.filters)
        keywords = generate_keywords(topic, filters=options.filters)
        meta.keywords = keywords
        persistence.save_step(run_dir, "01_keywords", keywords)
        if options.filters:
            _emit(
                bus,
                StepProgress(step="planner", message=f"已应用 {len(options.filters)} 条澄清约束"),
            )
        _emit(bus, StepCompleted(step="planner", message=f"{len(keywords)} 条关键词"))

        # Step 2: search (iterative — refine if first round is thin)
        _emit(bus, StepStarted(step="search", title="检索论文"))
        cache_key = {"keywords": keywords, "max_results": max_results, "sources": options.sources}
        papers = cache.get("search", cache_key) if options.use_cache else None
        if papers is None:
            papers = await search_papers(keywords, max_results, sources=options.sources)
            if options.use_cache:
                cache.put("search", cache_key, [p.model_dump() for p in papers])
        else:
            from litreview.models import Paper

            papers = [Paper(**p) for p in papers]

        # Iterative refinement: if pool is thin, refine queries and search again
        min_pool = max_papers * 2 if max_papers else 15
        if len(papers) < min_pool and options.mode != RunMode.QUICK_SCAN:
            _emit(
                bus,
                StepProgress(step="search", message=f"首轮 {len(papers)} 篇不足，迭代扩展检索…"),
            )
            found_titles = [p.title for p in papers[:10]]
            refined = refine_queries(topic, keywords, found_titles, known_gaps=[])
            if refined:
                _emit(
                    bus,
                    StepProgress(step="search", message=f"补充查询: {', '.join(refined[:3])}"),
                )
                more = await search_papers(refined, max_results=min(max_results, 30), sources=options.sources)
                # Merge and deduplicate
                from litreview.search import deduplicate

                papers = deduplicate(papers + more)
                keywords = keywords + refined  # track all queries used
                meta.keywords = keywords

        if max_papers:
            papers = blend_pool(papers, cap=max(max_papers * 3, 12))
        meta.total_papers = len(papers)
        persistence.save_step(run_dir, "02_papers", papers)
        _emit(bus, StepCompleted(step="search", message=f"{len(papers)} 篇候选"))

        # Step 3: filter
        _emit(bus, StepStarted(step="filter", title="论文筛选"))

        def _filter_progress(i, total, title):
            _emit(bus, StepProgress(step="filter", message=title, current=i, total=total))

        scored, filter_log = filter_papers(
            papers, topic, filters=options.filters, on_progress=_filter_progress, return_log=True
        )
        # venue-aware composite ranking (relevance x citations x SJR quartile),
        # applied when deciding which papers survive the cap
        scored = rank_papers(scored)
        persistence.save_step(
            run_dir,
            "03c_ranking",
            [
                {
                    "title": p.title[:60],
                    "relevance": p.relevance_score,
                    "citations": p.citation_count,
                    "venue": p.venue,
                    "quartile": p.venue_quartile or "-",
                    "rank_score": p.rank_score,
                }
                for p in scored
            ],
        )
        # Save filter log with exclusion reasons
        persistence.save_step(run_dir, "03_filter_log", filter_log)
        if max_papers and len(scored) > max_papers:
            scored = scored[:max_papers]
        meta.filtered_papers = len(scored)
        persistence.save_step(run_dir, "03_filtered", scored)
        hist = quartile_histogram(scored)
        hist_msg = " · ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
        _emit(bus, StepCompleted(step="filter", message=f"{len(scored)} 篇通过（{hist_msg}）"))

        # Step 3.2: citation snowballing — expand pool via refs/citations of top papers
        if options.mode != RunMode.QUICK_SCAN:
            _emit(bus, StepStarted(step="snowball", title="引用滚雪球"))
            top_seeds = [p for p in scored[:3] if p.doi]
            existing = {p.id for p in scored}
            snowballed = await snowball(top_seeds, existing, limit_per_paper=6)
            if snowballed:
                _emit(
                    bus,
                    StepProgress(step="snowball", message=f"滚雪球发现 {len(snowballed)} 篇候选"),
                )
                # Filter the snowballed papers too
                snow_scored = filter_papers(snowballed, topic, filters=options.filters)
                snow_scored = rank_papers(snow_scored)
                # Only add papers that pass quality bar and aren't already in
                new_papers = [
                    p for p in snow_scored
                    if p.id not in existing and p.relevance_score >= 3
                ][:options.max_supplement_papers]
                if new_papers:
                    scored = scored + new_papers
                    meta.filtered_papers = len(scored)
                    persistence.save_step(
                        run_dir, "03s_snowball", {"added": new_papers, "total_found": len(snowballed)}
                    )
                    _emit(
                        bus,
                        StepCompleted(
                            step="snowball",
                            message=f"滚雪球补充 {len(new_papers)} 篇（来源: 引用图）"
                        ),
                    )
                else:
                    _emit(bus, StepCompleted(step="snowball", message="滚雪球无新增相关论文"))
            else:
                _emit(bus, StepCompleted(step="snowball", message="无可用引用图数据"))

        # Step 3.5: enrich missing abstracts via cross-source DOI lookup
        if options.enrich_abstracts:
            _emit(bus, StepStarted(step="enrich", title="摘要补全"))
            missing = sum(1 for p in scored if not p.abstract.strip())
            if missing:
                def _enrich_progress(i, n, t):
                    _emit(bus, StepProgress(step="enrich", message=t, current=i, total=n))

                filled, elog = enrich_abstracts(scored, on_progress=_enrich_progress)
                persistence.save_step(
                    run_dir,
                    "03b_enrichment",
                    {"missing": missing, "filled": filled, "log": elog},
                )
                _emit(bus, StepCompleted(step="enrich", message=f"补全 {filled}/{missing} 篇缺失摘要"))
            else:
                _emit(bus, StepCompleted(step="enrich", message="无缺失摘要"))

        # Step 4: extract
        _emit(bus, StepStarted(step="extract", title="信息抽取"))

        def _extract_progress(i, total, title):
            _emit(bus, StepProgress(step="extract", message=title, current=i, total=total))

        extracted = extract_papers(scored, topic, on_progress=_extract_progress)
        persistence.save_step(run_dir, "04_extracted", extracted)

        # Generate comparison matrix from deep extraction (quality signals)
        comp_rows = build_comparison_matrix(extracted)
        comp_md = render_comparison_md(comp_rows)
        persistence.save_step(run_dir, "04_comparison", comp_rows)
        if comp_md:
            persistence.save_text(run_dir, "comparison.md", comp_md)
        n_with_quality = sum(1 for p in extracted if p.quality.get("evidence_level"))
        _emit(
            bus,
            StepCompleted(
                step="extract",
                message=f"{len(extracted)} 篇抽取完成 · {n_with_quality} 篇含质量评估 · 对比矩阵已生成"
            ),
        )

        # Step 5: first composition (verifier feedback may raise supplement queries)
        verify_trigger_queries: list[str] = []

        def _collect_verify_queries(queries, _msg):
            verify_trigger_queries.extend(queries)

        result = _compose(
            extracted,
            topic,
            run_dir,
            bus,
            fetch_fulltext=options.fetch_fulltext,
            on_supplement_queries=_collect_verify_queries,
        )
        meta.themes = [t.name for t in result.themes.themes]

        # Step 6: reflect -> supplement -> recompose (one bounded loop)
        if options.allow_supplement:
            _emit(bus, StepStarted(step="reflect", title="反思与补充"))
            decision = reflect(result.synthesis, topic, len(extracted))

            # 6a: absence audit — canonical works the retrieved set is missing
            audit = audit_coverage(topic, [p.title for p in extracted])
            persistence.save_step(run_dir, "08a_absence_audit", audit)
            audit_queries = missing_to_queries(audit)
            absent_n = len(audit.get("absent_canonical_papers", []))
            if audit_queries:
                _emit(
                    bus,
                    StepProgress(
                        step="reflect",
                        message=f"缺席检测: {absent_n} 篇经典缺失，追加检索",
                    ),
                )

            # merge all three query sources: reflect gaps + absence audit +
            # verifier-triggered (unsupported-claim evidence)
            supplement_queries = list(dict.fromkeys(
                decision.get("supplementary_keywords", [])
                + audit_queries
                + verify_trigger_queries
            ))[:8]

            if decision["needs_supplement"] and supplement_queries:
                _emit(
                    bus,
                    StepProgress(step="reflect", message="补充检索: " + ", ".join(supplement_queries)),
                )
                fresh, all_known = await _supplement_search(
                    supplement_queries, {p.id for p in extracted}, options, topic
                )
                if fresh:
                    _emit(
                        bus,
                        StepProgress(step="reflect", message=f"补充到 {len(fresh)} 篇新论文，重新综合"),
                    )
                    new_extracted = extract_papers(fresh, topic)
                    extracted = extracted + new_extracted
                    meta.filtered_papers = len(extracted)
                    persistence.save_step(
                        run_dir,
                        "08_supplement",
                        {"new_papers": fresh, "decision": decision, "audit": audit},
                    )
                    result = _compose(
                        extracted,
                        topic,
                        run_dir,
                        bus,
                        fetch_fulltext=options.fetch_fulltext,
                        on_supplement_queries=_collect_verify_queries,
                    )
                    meta.themes = [t.name for t in result.themes.themes]
                    _emit(
                        bus,
                        StepCompleted(step="reflect", message=f"补充 {len(new_extracted)} 篇并重新综合完成"),
                    )
                else:
                    msg = (
                        "检索命中但均已在池内（可能是缓存回放），跳过重新综合"
                        if all_known
                        else "无新论文，跳过重新综合"
                    )
                    _emit(bus, StepCompleted(step="reflect", message=msg))
            else:
                _emit(
                    bus,
                    StepCompleted(step="reflect", message=decision["rationale"] or "覆盖充分，无需补充"),
                )

        # finalize
        meta.review_path = result.review_path
        meta.citation_precision = round(result.precision, 3)
        persistence.save_json(run_dir, "meta.json", meta)
        _emit(
            bus,
            RunCompleted(
                run_dir=run_dir,
                review_path=result.review_path,
                summary={
                    "topic": topic,
                    "total_papers": meta.total_papers,
                    "filtered_papers": meta.filtered_papers,
                    "themes": meta.themes,
                    "claims": len(result.claims),
                    "references": len(result.table),
                    "citation_precision": meta.citation_precision,
                },
            ),
        )
        return meta

    except Exception as e:  # noqa: BLE001
        _emit(bus, RunFailed(message=str(e), step="pipeline"))
        traceback.print_exc()
        raise


def run_pipeline(
    topic: str,
    options: RunOptions | None = None,
    bus: EventBus | None = None,
) -> RunMeta:
    """Sync entry point (for CLI). Runs the async pipeline via asyncio.run."""
    return asyncio.run(run_pipeline_async(topic, options, bus))
