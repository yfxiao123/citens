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
from typing import Optional

from litreview import cache, persistence
from litreview.agents import (
    extract_papers,
    filter_papers,
    generate_keywords,
    organize_themes,
    reflect,
    synthesize,
    verify_claims,
    write_review_body,
)
from litreview.config import settings
from litreview.events import (
    EventBus,
    Event,
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
from litreview.models import ExtractedPaper, RunMeta, ThemeStructure
from litreview.search import blend_pool, search_papers


@dataclass
class RunOptions:
    max_results: Optional[int] = None
    max_papers: Optional[int] = None
    sources: Optional[list[str]] = None
    use_cache: bool = True
    allow_supplement: bool = True
    max_supplement_papers: int = 4
    fetch_fulltext: bool = True
    enrich_abstracts: bool = True
    extra: dict = field(default_factory=dict)

    def resolved_max_results(self) -> int:
        return self.max_results or settings.default_max_results

    def resolved_max_papers(self) -> Optional[int]:
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


def _emit(bus: Optional[EventBus], event: Event) -> None:
    if bus is not None:
        bus.emit(event)


def _compose(
    extracted: list[ExtractedPaper],
    topic: str,
    run_dir: str,
    bus: Optional[EventBus],
    *,
    fetch_fulltext: bool = True,
) -> ComposeResult:
    """organize -> synthesize -> write -> verify, persisting all artifacts.

    Returns the final table / claims / precision so the outer flow can decide
    on reflection and assemble RunMeta.
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
    review = body + "\n## 参考文献 / References\n\n" + table.references_md() + "\n"
    review_path = persistence.save_text(run_dir, "review.md", review)
    persistence.save_text(run_dir, "references.bib", table.to_bibtex())

    claims = parse_claims_from_review(review)
    persistence.save_step(run_dir, "07_claims", claims)
    _emit(
        bus,
        StepCompleted(step="write", message=f"综述生成完成 · {len(claims)} 条带引用论断"),
    )

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
) -> list:
    """Gap-targeted supplementary retrieval -> filter -> new papers only."""
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
    scored = filter_papers(papers, topic)
    fresh = [p for p in scored if p.id not in existing_ids][: options.max_supplement_papers]
    return fresh


async def run_pipeline_async(
    topic: str,
    options: Optional[RunOptions] = None,
    bus: Optional[EventBus] = None,
) -> RunMeta:
    """Run the full pipeline asynchronously. Raises on unrecoverable failure."""
    options = options or RunOptions()
    max_results = options.resolved_max_results()
    max_papers = options.resolved_max_papers()

    run_dir = persistence.new_run_dir(topic)
    meta = RunMeta(topic=topic, run_dir=run_dir)
    _emit(bus, RunStarted(topic=topic))

    try:
        # Step 1: keywords
        _emit(bus, StepStarted(step="planner", title="生成检索关键词"))
        keywords = generate_keywords(topic)
        meta.keywords = keywords
        persistence.save_step(run_dir, "01_keywords", keywords)
        _emit(bus, StepCompleted(step="planner", message=f"{len(keywords)} 条关键词"))

        # Step 2: search
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
        if max_papers:
            papers = blend_pool(papers, cap=max(max_papers * 3, 12))
        meta.total_papers = len(papers)
        persistence.save_step(run_dir, "02_papers", papers)
        _emit(bus, StepCompleted(step="search", message=f"{len(papers)} 篇候选"))

        # Step 3: filter
        _emit(bus, StepStarted(step="filter", title="论文筛选"))

        def _filter_progress(i, total, title):
            _emit(bus, StepProgress(step="filter", message=title, current=i, total=total))

        scored = filter_papers(papers, topic, on_progress=_filter_progress)
        if max_papers and len(scored) > max_papers:
            scored = sorted(
                scored, key=lambda p: (p.relevance_score, p.citation_count), reverse=True
            )[:max_papers]
        meta.filtered_papers = len(scored)
        persistence.save_step(run_dir, "03_filtered", scored)
        _emit(bus, StepCompleted(step="filter", message=f"{len(scored)} 篇通过"))

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
        _emit(bus, StepCompleted(step="extract", message=f"{len(extracted)} 篇抽取完成"))

        # Step 5: first composition
        result = _compose(extracted, topic, run_dir, bus, fetch_fulltext=options.fetch_fulltext)
        meta.themes = [t.name for t in result.themes.themes]

        # Step 6: reflect -> supplement -> recompose (one bounded loop)
        if options.allow_supplement:
            _emit(bus, StepStarted(step="reflect", title="反思与补充"))
            decision = reflect(result.synthesis, topic, len(extracted))
            if decision["needs_supplement"] and decision["supplementary_keywords"]:
                _emit(
                    bus,
                    StepProgress(step="reflect", message="补充检索: " + ", ".join(decision["supplementary_keywords"])),
                )
                fresh = await _supplement_search(
                    decision["supplementary_keywords"], {p.id for p in extracted}, options, topic
                )
                if fresh:
                    _emit(
                        bus,
                        StepProgress(step="reflect", message=f"补充到 {len(fresh)} 篇新论文，重新综合"),
                    )
                    new_extracted = extract_papers(fresh, topic)
                    extracted = extracted + new_extracted
                    meta.filtered_papers = len(extracted)
                    persistence.save_step(run_dir, "08_supplement", {"new_papers": fresh, "decision": decision})
                    result = _compose(
                        extracted, topic, run_dir, bus, fetch_fulltext=options.fetch_fulltext
                    )
                    meta.themes = [t.name for t in result.themes.themes]
                    _emit(
                        bus,
                        StepCompleted(step="reflect", message=f"补充 {len(new_extracted)} 篇并重新综合完成"),
                    )
                else:
                    _emit(bus, StepCompleted(step="reflect", message="无新论文，跳过重新综合"))
            else:
                _emit(bus, StepCompleted(step="reflect", message=decision["rationale"] or "覆盖充分，无需补充"))

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
    options: Optional[RunOptions] = None,
    bus: Optional[EventBus] = None,
) -> RunMeta:
    """Sync entry point (for CLI). Runs the async pipeline via asyncio.run."""
    return asyncio.run(run_pipeline_async(topic, options, bus))
