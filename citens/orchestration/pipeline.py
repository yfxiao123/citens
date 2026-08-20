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
import contextlib
import traceback
from dataclasses import dataclass, field

from citens import cache, persistence
from citens.agents import (
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
    rewrite_unsupported_claims,
    synthesize,
    verify_claims,
    write_review_body,
)
from citens.agents.planner import (
    discover_terms,
    generate_facets,
    generate_seed_papers,
    refine_queries,
)
from citens.agents.quality import build_comparison_matrix, render_comparison_md
from citens.agents.verifier import claim_stack_stats
from citens.agents.writer import localized_heading
from citens.artifacts import write_review_browser
from citens.config import settings
from citens.events import (
    EventBus,
    RunCompleted,
    RunFailed,
    RunStarted,
    StepCompleted,
    StepProgress,
    StepStarted,
)
from citens.grounding import (
    ChunkStore,
    CitationTable,
    build_provenance,
    enrich_abstracts,
    parse_claims_from_review,
    write_fetch_list,
)
from citens.models import (
    ClaimIntentManifest,
    ExtractedPaper,
    RunMeta,
    RunMode,
    SynthesisResult,
    ThemeStructure,
    Verdict,
    VerificationResult,
)
from citens.orchestration.support import (  # noqa: F401 - re-exported
    StepClock,
    _emit,
    _gate_supplement_papers,
    _load_extracted_for_resume,
    _mode_from_run_dir,
    coverage_note_text,
    demote_blind_papers,
    facet_coverage_report,
    number_dense_excerpts,
    prune_citation_stacking,
    search_summary_text,
)
from citens.ranking import quartile_histogram, rank_papers
from citens.runlog import RunLog
from citens.search import (
    blend_pool,
    deduplicate,
    search_papers,
    search_papers_with_health,
)
from citens.search.seeds import resolve_seeds
from citens.search.snowball import snowball


@dataclass
class RunOptions:
    max_results: int | None = None
    max_papers: int | None = None
    # Supporting-reference layer size (None = settings default, 0 = off):
    # filtered-relevant papers beyond the core cap join the bibliography as
    # abstract-only citations instead of being thrown away.
    support_papers: int | None = None
    sources: list[str] | None = None
    use_cache: bool = True
    # Seed the candidate pool from `citens collect`'s persistent literature
    # pool when one exists (record-first workflow), and write new finds back.
    use_pool: bool = True
    # Domain profile name (overrides settings.profile); "" = generic.
    profile: str = ""
    allow_supplement: bool = True
    max_supplement_papers: int = 4
    fetch_fulltext: bool = True
    enrich_abstracts: bool = True
    extra: dict = field(default_factory=dict)
    filters: dict = field(default_factory=dict)  # pre-run clarifications (see clarify.py)
    mode: RunMode | None = None  # adaptive mode (auto-detected if None)
    resume_dir: str | None = None  # existing run dir: reuse its extracted papers

    def resolved_max_results(self) -> int:
        # The candidate pool must stay well above the final cap, or the filter
        # stage is a no-op (60 candidates for -n 100 selects nothing). When the
        # user raises the paper count without touching pool size, scale the
        # pool along (only when max_results was left at its default).
        if self.max_results:
            return self.max_results
        base = settings.default_max_results
        if self.max_papers:
            base = max(base, self.max_papers * 4)
        return base

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
    synthesis: SynthesisResult


def _compose(
    extracted: list[ExtractedPaper],
    topic: str,
    run_dir: str,
    bus: EventBus | None,
    *,
    fetch_fulltext: bool = True,
    on_supplement_queries=None,
    clock: StepClock | None = None,
    label: str = "compose",
    chunk_store: ChunkStore | None = None,
    terminology: dict[str, str] | None = None,
    supporting: list | None = None,
    facets: list | None = None,
    verdict_cache: dict | None = None,
    verdict_fuzzy: dict | None = None,
    n_candidates: int = 0,
    evidence_bias: str = "number_density",
) -> ComposeResult:
    """organize -> synthesize -> write -> verify, persisting all artifacts.

    Returns the final table / claims / precision so the outer flow can decide
    on reflection and assemble RunMeta. `on_supplement_queries`, if given, is
    invoked with (queries, message) when verification finds unsupported claims
    that could be resolved by targeted retrieval (the caller decides whether
    to actually supplement).

    ``supporting`` joins the bibliography beyond the deep-dive set: abstract-
    only citations the writer may use for background/comparison claims (the
    verifier checks them against the abstract like any other claim).
    """
    if clock is None:
        clock = StepClock()

    core_ids = {p.id for p in extracted}
    supporting = [p for p in (supporting or []) if p.id not in core_ids]
    table_papers = list(extracted) + supporting

    def _recompute_precision() -> float:
        verifiable = [r for r in ver_results if r.verdict.value != "unverifiable"]
        if not verifiable:
            return 0.0
        return sum(
            1 for r in verifiable if r.verdict.value in ("supported", "partial")
        ) / len(verifiable)

    # organize
    clock.mark(f"{label}:organize")
    _emit(bus, StepStarted(step="organize", title="主题组织"))
    themes = organize_themes(extracted, topic)
    persistence.save_step(run_dir, "05_themes", themes)
    _emit(bus, StepCompleted(step="organize", message=f"{len(themes.themes)} 个主题"))

    # synthesize (critical cross-paper analysis)
    clock.mark(f"{label}:synthesize")
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
    clock.mark(f"{label}:ground")
    _emit(bus, StepStarted(step="ground", title="获取全文（溯源）"))

    def _ground_progress(i, total, title):
        _emit(bus, StepProgress(step="ground", message=title, current=i, total=total))

    # shared across compose rounds so fulltext fetch/parse happens once
    chunk_store = chunk_store or ChunkStore()
    chunk_store.build_from(extracted, fetch_full=fetch_fulltext, on_progress=_ground_progress)
    # supporting layer: abstract-only ground text (no fetch, no extract)
    if supporting:
        chunk_store.build_from(supporting, fetch_full=False)
    table = CitationTable(table_papers)
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
            "supporting": len(supporting),
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
    clock.mark(f"{label}:write")
    _emit(bus, StepStarted(step="write", title="综述撰写"))

    def _write_step(name, label=""):
        _emit(bus, StepProgress(step="write", message=label or name))

    def _evidence_for(theme) -> str:
        """Full-text excerpts for a theme's papers (write-time grounding)."""
        return number_dense_excerpts(
            theme.paper_indices, extracted, chunk_store,
            query=f"{theme.name} {theme.description}",
            bias=evidence_bias,
        )

    # coverage honesty: hard numbers from retrieval feed the writer's
    # coverage paragraph (thin facets, thin themes, blind papers)
    cov_report = facet_coverage_report(facets or [], extracted)
    n_blind_now = sum(1 for p in extracted if not (p.abstract or "").strip())
    cov_note = coverage_note_text(cov_report, themes.themes, n_blind_now)

    ssum = search_summary_text(
        None, n_candidates, len(extracted), len(supporting or [])
    ) if n_candidates else ""
    body = write_review_body(
        extracted, themes, topic, synthesis=synthesis, on_step=_write_step,
        evidence_for=_evidence_for, terminology=terminology,
        supporting=[(len(extracted) + j, p) for j, p in enumerate(supporting)],
        coverage_note=cov_note,
        search_summary=ssum,
    )
    # hard citation-stacking cap: the prompt softens, this enforces (keep the
    # most BM25-relevant cites per overloaded sentence, strip the rest)
    body, stack_log = prune_citation_stacking(body, list(extracted) + list(supporting or []))
    if stack_log:
        persistence.save_step(
            run_dir, "06b_stack_lint",
            {"pruned_sentences": len(stack_log), "log": stack_log},
        )
        _emit(
            bus,
            StepProgress(
                step="write",
                message=f"引用堆砌剪枝: {len(stack_log)} 句超限，共剪除 "
                f"{sum(len(s['dropped']) for s in stack_log)} 个装饰性引用",
            ),
        )
    review = body + f"\n## {localized_heading('refs')}\n\n" + table.references_md() + "\n"
    review_path = persistence.save_text(run_dir, "review.md", review)
    persistence.save_text(run_dir, "references.bib", table.to_bibtex())
    # RIS for EndNote/Zotero users (nature-citation default export)
    persistence.save_text(run_dir, "references.ris", table.to_ris())

    claims = parse_claims_from_review(review)
    persistence.save_step(run_dir, "07_claims", claims)
    # citation coverage: which bibliography papers (core + supporting) never
    # got cited anywhere (breadth signal — pairs with the writer's
    # cite-broadly rule and the health issue below)
    n_table = len(table_papers)
    cited_papers = {
        i for c in claims for i in c.citation_indices if 0 <= i < n_table
    }
    _emit(
        bus,
        StepCompleted(step="write", message=f"综述生成完成 · {len(claims)} 条带引用论断"),
    )

    # claim intent manifest: map claims to synthesis intents
    intent_to_claims: dict[str, list[int]] = {}
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
    ver_results: list[VerificationResult] = []
    precision = 0.0
    if claims:
        clock.mark(f"{label}:verify")
        _emit(bus, StepStarted(step="verify", title="引用核验"))

        def _verify_progress(i, total):
            _emit(
                bus,
                StepProgress(step="verify", message=f"核验论断 {i}/{total}", current=i, total=total),
            )

        ver_results, precision = verify_claims(
            claims, table, chunk_store, on_progress=_verify_progress,
            verdict_cache=verdict_cache, verdict_fuzzy=verdict_fuzzy,
        )
        verifier_precision = precision

        # stage-specific extras (defense / rewrite / leniency / canary) merged
        # into every subsequent verification.json save, so no later save drops
        # the keys an earlier stage added
        payload_extras: dict = {}

        def _verification_payload() -> dict:
            n_verifiable = sum(
                1 for r in ver_results if r.verdict.value != "unverifiable"
            )
            payload = {
                "citation_precision": round(precision, 3),
                "total_claims": len(claims),
                "verifiable_claims": n_verifiable,
                "unverifiable_rate": round(
                    (len(claims) - n_verifiable) / len(claims), 3
                ) if claims else 0.0,
                "supported": sum(1 for r in ver_results if r.verdict.value == "supported"),
                "partial": sum(1 for r in ver_results if r.verdict.value == "partial"),
                "background": sum(1 for r in ver_results if r.verdict.value == "background"),
                "contradictory": sum(
                    1 for r in ver_results if r.verdict.value == "contradictory"
                ),
                "unsupported": sum(1 for r in ver_results if r.verdict.value == "unsupported"),
                "unverifiable": sum(
                    1 for r in ver_results if r.verdict.value == "unverifiable"
                ),
                "papers_cited": len(cited_papers),
                "papers_total": n_table,
                "citation_stacking": claim_stack_stats(claims),
                "results": [r.model_dump() for r in ver_results],
            }
            payload.update(payload_extras)
            return payload

        persistence.save_json(run_dir, "verification.json", _verification_payload())
        _emit(bus, StepCompleted(step="verify", message=f"引用精度 {precision * 100:.0f}%"))

        # canary honeypot: synthetic unsupported claims through the same judge —
        # a direct measurement of the false-accept rate behind the headline
        from citens.agents.verifier import canary_check

        canary = canary_check(table, chunk_store)
        payload_extras["canary"] = canary
        if canary.get("injected"):
            persistence.save_json(run_dir, "verification.json", _verification_payload())
            rate = canary.get("false_accept_rate")
            rate_msg = f"{rate * 100:.0f}%" if rate is not None else "n/a"
            _emit(
                bus,
                StepProgress(
                    step="verify",
                    message=(
                        f"蜜罐检测: {canary['injected']} 条无依据论断，"
                        f"漏判率 {rate_msg}（{canary.get('caught', 0)} 条被抓住）"
                    ),
                ),
            )

        # bidirectional verification: defense lawyer challenges defect verdicts
        defect_counts = {
            v: sum(1 for r in ver_results if r.verdict.value == v)
            for v in ("unsupported", "background", "contradictory")
        }
        if sum(defect_counts.values()) > 0:
            clock.mark(f"{label}:defense")
            _emit(bus, StepStarted(step="defense", title="双向核验（辩护律师）"))
            source_contexts: dict[int, str] = {}
            for i, claim in enumerate(claims):
                # citation_indices are [n] table indices — resolve to the
                # paper's chunk-store key (its id) before fetching ground text.
                pid = table.paper_id(claim.citation_indices[0]) if claim.citation_indices else ""
                chunks = chunk_store.chunks_for(pid)
                source_contexts[i] = "\n\n".join(c.text for c in chunks[:3])

            defense_reviews = review_unsupported_claims(claims, ver_results, source_contexts)
            overturned = sum(1 for r in defense_reviews if r["overturned"])

            # second read: an overturned verdict means the verifier misjudged
            # the claim; downgrade conservatively to partial (defended is not
            # the same as explicitly supported) and fold into the metric.
            for defense_review in defense_reviews:
                if defense_review["overturned"]:
                    idx = defense_review["claim_idx"]
                    if 0 <= idx < len(ver_results):
                        ver_results[idx].verdict = Verdict.PARTIAL
                        ver_results[idx].note = (
                            f"{ver_results[idx].note} "
                            f"[defense: {defense_review['defense_result'].get('rebuttal', '')[:200]}]"
                        ).strip()
            verifiable = [r for r in ver_results if r.verdict.value != "unverifiable"]
            if verifiable:
                precision = (
                    sum(1 for r in verifiable if r.verdict.value in ("supported", "partial"))
                    / len(verifiable)
                )

            payload_extras["verifier_precision"] = round(verifier_precision, 3)
            payload_extras["defense_overturned"] = overturned
            persistence.save_json(run_dir, "verification.json", _verification_payload())
            persistence.save_step(run_dir, "08_defense", defense_reviews)
            _emit(
                bus,
                StepCompleted(
                    step="defense",
                    message=(
                        f"辩护律师推翻 {overturned} 条 → 精度 "
                        f"{verifier_precision * 100:.0f}% ⇒ {precision * 100:.0f}%"
                    )
                ),
            )

        # rewrite loop: FIX (not just report) surviving defect claims —
        # unsupported (no basis), background (context-only citation) and
        # contradictory (source disagrees) each get a tailored rewrite. The
        # rewriter weakens/qualifies/re-aims each claim to match its sources;
        # the revised claims are re-verified and the review text is updated in
        # place — precision becomes a pipeline behavior, not a scorecard.
        pre_rewrite_precision = precision  # post-defense, pre-rewrite
        remaining_defects = [
            i
            for i, r in enumerate(ver_results)
            if r.verdict.value in ("unsupported", "background", "contradictory")
        ]
        if remaining_defects:
            clock.mark(f"{label}:rewrite")
            _emit(bus, StepStarted(step="rewrite", title="论断改写"))
            rewrites = rewrite_unsupported_claims(claims, ver_results, table, chunk_store)
            applied = []
            for i, rw in sorted(rewrites.items()):
                old_text = claims[i].text
                if old_text in review:
                    # rewritten text re-enters the same stacking discipline as
                    # the first draft (the rewriter's multi-sentence output can
                    # otherwise re-stack citations the post-write lint removed)
                    new_text, rw_lint = prune_citation_stacking(
                        rw["new_text"], table_papers
                    )
                    if rw_lint:
                        rw = {**rw, "new_text": new_text}
                    review = review.replace(old_text, rw["new_text"], 1)
                    claims[i].text = rw["new_text"]
                    ver_results[i].claim_text = rw["new_text"]
                    applied.append(
                        {
                            "claim_index": i,
                            "old_text": old_text,
                            "new_text": rw["new_text"],
                            "note": rw["note"],
                        }
                    )
            if applied:
                sub_claims = [claims[a["claim_index"]] for a in applied]
                sub_results, _ = verify_claims(
                    sub_claims, table, chunk_store, verdict_cache=verdict_cache,
                    verdict_fuzzy=verdict_fuzzy,
                )
                by_text = {r.claim_text: r for r in sub_results}
                fixed = 0
                for a in applied:
                    sr = by_text.get(a["new_text"])
                    if sr is None:
                        continue
                    idx = a["claim_index"]
                    if sr.verdict.value in ("supported", "partial"):
                        fixed += 1
                    ver_results[idx].verdict = sr.verdict
                    ver_results[idx].note = f"[rewrite] {sr.note}"[:400]
                precision = _recompute_precision()
                persistence.save_step(
                    run_dir,
                    "08c_rewrites",
                    {"rewrites": applied, "reverified_grounded": fixed},
                )
                persistence.save_step(run_dir, "07_claims", claims)
                persistence.save_text(run_dir, "review.md", review)
                # report BOTH sides of the rewrite: the conservative
                # pre-rewrite number and what the loop bought
                payload_extras["rewrite_fixed"] = fixed
                payload_extras["pre_rewrite_precision"] = round(pre_rewrite_precision, 3)
                payload_extras["post_rewrite_precision"] = round(precision, 3)
                persistence.save_json(run_dir, "verification.json", _verification_payload())
                _emit(
                    bus,
                    StepCompleted(
                        step="rewrite",
                        message=(
                            f"改写 {len(applied)} 条缺陷论断（无依据/背景引用/矛盾）· "
                            f"复验后 {fixed} 条转为有依据 · "
                            f"精度 {pre_rewrite_precision * 100:.0f}% ⇒ {precision * 100:.0f}%"
                        ),
                    ),
                )
            else:
                payload_extras["pre_rewrite_precision"] = round(pre_rewrite_precision, 3)
                payload_extras["post_rewrite_precision"] = round(pre_rewrite_precision, 3)
                persistence.save_json(run_dir, "verification.json", _verification_payload())
                _emit(
                    bus,
                    StepCompleted(step="rewrite", message="无可安全改写的论断，保持原判"),
                )

        # leniency audit: re-check a sample of "supported" verdicts under a
        # stricter standard, so a vague-claim rewrite cannot quietly inflate
        # the headline number unnoticed
        from citens.agents.verifier import spot_check_supported

        leniency = spot_check_supported(claims, ver_results, table, chunk_store)
        persistence.save_step(run_dir, "08d_leniency_check", leniency)
        if leniency.get("sampled"):
            ar = leniency.get("agreement_rate")
            ar_msg = f"{ar * 100:.0f}%" if ar is not None else "n/a"
            _emit(
                bus,
                StepProgress(
                    step="verify",
                    message=(
                        f"宽严抽检: {leniency['sampled']} 条 supported 复审，"
                        f"{leniency['downgraded']} 条应降级（一致率 {ar_msg}）"
                    ),
                ),
            )

        # the sprinkler behind the smoke detector: ADOPT the strict re-audit's
        # downgrades (supported -> partial) instead of merely reporting them,
        # so lenient first-pass verdicts cannot survive into the headline
        adopted = 0
        for item in leniency.get("results", []):
            i = item.get("claim_index")
            if (
                item.get("verdict") == "downgrade"
                and isinstance(i, int)
                and 0 <= i < len(ver_results)
                and ver_results[i].verdict == Verdict.SUPPORTED
            ):
                ver_results[i].verdict = Verdict.PARTIAL
                ver_results[i].note = (
                    f"{ver_results[i].note} [leniency: {item.get('note', '')}]".strip()
                )[:400]
                adopted += 1
        if adopted:
            precision = _recompute_precision()
            payload_extras["leniency_downgraded"] = adopted
            persistence.save_json(run_dir, "verification.json", _verification_payload())
            _emit(
                bus,
                StepProgress(
                    step="verify",
                    message=(
                        f"宽严纠偏: {adopted} 条 supported 降级为 partial · "
                        f"精度修正为 {precision * 100:.0f}%"
                    ),
                ),
            )

        # health monitoring: detect systematic biases
        clock.mark(f"{label}:health")
        _emit(bus, StepStarted(step="health", title="对话健康监测"))
        theme_paper_counts = {theme.name: len(theme.paper_indices) for theme in themes.themes}
        absence_audit = audit_coverage(topic, [p.title for p in extracted])
        health_report = check_health(
            synthesis, ver_results, absence_audit, theme_paper_counts, canary=canary
        )
        health_report["citation_coverage"] = {
            "cited": len(cited_papers),
            "total": n_table,
        }
        if n_table >= 8 and len(cited_papers) < 0.7 * n_table:
            health_report.setdefault("issues", []).append(
                f"thin_citation_coverage: only {len(cited_papers)}/{n_table} "
                "bibliography papers cited anywhere in the review"
            )
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
            from citens.agents.verifier_trigger import collect_unsupported_queries

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

    persistence.save_json(
        run_dir,
        "provenance.json",
        build_provenance(claims, table, ver_results, chunk_store=chunk_store),
    )

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
        from citens.models import Paper

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

    run_dir = options.resume_dir or persistence.new_run_dir(topic)
    meta = RunMeta(topic=topic, run_dir=run_dir)
    runlog = RunLog(run_dir)
    clock = StepClock(runlog=runlog)
    # tag every LLM usage record of this run (incl. thread-pool jobs) so
    # concurrent runs in one process attribute cleanly
    from citens.llm import run_scope

    scope = run_scope(runlog.run_id)
    scope.__enter__()
    runlog.append(
        "run_start",
        topic=topic,
        mode=str(options.mode) if options.mode else "auto",
        max_results=max_results,
        max_papers=max_papers,
    )
    _emit(bus, RunStarted(topic=topic))
    if options.resume_dir is None:
        # written before anything can fail, so `citens resume` can recover
        # the exact topic even when meta.json was never reached
        persistence.save_json(run_dir, "run.json", {"topic": topic})

    # bound once here so every later stage (including the resume path, which
    # skips the planner) can rely on it being defined
    from citens.profiles import load_profile, order_sources

    profile = load_profile(options.profile or settings.profile)
    # domain-preferred source order (dedup keeps the first reporter of a
    # preprint/published pair — finance wants the journal record to win)
    options.sources = order_sources(options.sources, profile)
    # venue whitelist (used by retrieval-side strict mode AND the ranking)
    _venue_boost = profile.venue_boost_set() if profile is not None else None

    # supporting-reference layer: bound here (like profile) because the
    # resume path skips the filter stage that fills it — empty is correct
    supporting: list = []
    # facet plan + cross-round verdict cache: bound for both paths (the
    # resume path skips the planner that fills facets — empty is correct)
    facets: list = []
    verdict_cache: dict = {}
    # (normalized text, citations) -> (grounding sig, verdict): lets the
    # post-supplement recompose reuse verdicts for reworded-but-same claims
    verdict_fuzzy: dict = {}
    search_health: dict[str, str] = {}

    try:
        resumed = options.resume_dir is not None
        if resumed:
            # Resume path: a previous run of this topic already extracted
            # its papers — reuse them and skip retrieval entirely.
            extracted = _load_extracted_for_resume(run_dir)
            if options.mode is None:
                options.mode = _mode_from_run_dir(run_dir)
            if options.mode == RunMode.QUICK_SCAN:
                options.allow_supplement = False
            _emit(
                bus,
                StepCompleted(
                    step="extract",
                    message=f"resume: reusing {len(extracted)} papers from {run_dir}",
                ),
            )
        if not resumed:
            # Step 0: intent detection (auto-detect mode if not specified)
            if options.mode is None:
                clock.mark("intent")
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
            clock.mark("planner")
            _emit(bus, StepStarted(step="planner", title="生成检索关键词"))
            if options.filters:
                persistence.save_step(run_dir, "00_filters", options.filters)
            keywords = generate_keywords(topic, filters=options.filters)
            # domain profile: curated terminology joins the keyword batches
            from citens.profiles import merge_profile_terms

            if profile is not None:
                before = set(keywords)
                keywords = merge_profile_terms(keywords, profile)
                added = [q for q in keywords if q not in before]
                if added:
                    _emit(
                        bus,
                        StepProgress(
                            step="planner",
                            message=f"金融 profile 注入 {len(added)} 条领域术语"
                            if profile.name == "finance"
                            else f"profile '{profile.name}' 注入 {len(added)} 条领域术语",
                        ),
                    )
            meta.keywords = keywords
            persistence.save_step(run_dir, "01_keywords", keywords)
            if options.filters:
                _emit(
                    bus,
                    StepProgress(step="planner", message=f"已应用 {len(options.filters)} 条澄清约束"),
                )
            _emit(bus, StepCompleted(step="planner", message=f"{len(keywords)} 条关键词"))

            # Step 1b: seed papers — landmark works the keywords may miss.
            # Their titles retrieve the canonical records themselves, and the
            # resolved seeds join the snowball seeds so their citation graph
            # (references + citing papers) is mined for neighbors.
            seed_papers: list = []
            if options.mode != RunMode.QUICK_SCAN:
                clock.mark("seeds")
                seed_titles, domain_terms = generate_seed_papers(topic, filters=options.filters)
                if domain_terms:
                    known = {q.lower() for q in keywords}
                    fresh_terms = [
                        t for t in domain_terms
                        if t.lower() not in known and not any(t.lower() in q for q in keywords)
                    ]
                    if fresh_terms:
                        keywords = keywords + fresh_terms
                        meta.keywords = keywords
                persistence.save_step(
                    run_dir,
                    "01b_seeds",
                    {"requested_titles": seed_titles, "domain_terms_added": domain_terms},
                )
                if seed_titles:
                    _emit(
                        bus,
                        StepProgress(
                            step="planner",
                            message=f"种子论文: {', '.join(t[:40] for t in seed_titles[:3])}…",
                        ),
                    )
                    seed_papers = await resolve_seeds(seed_titles)
                    _emit(
                        bus,
                        StepProgress(
                            step="planner",
                            message=f"种子论文解析 {len(seed_papers)}/{len(seed_titles)} 篇",
                        ),
                    )

                        # facet plan: coverage-by-design — the reflector targets thin
            # facets and the writer states them honestly
            facets = generate_facets(topic, filters=options.filters)
            if facets:
                persistence.save_step(run_dir, "01c_facets", facets)
                _emit(
                    bus,
                    StepProgress(
                        step="planner",
                        message=f"检索面规划: {len(facets)} 个面（"
                        + ", ".join(f["name"][:12] for f in facets[:4])
                        + "…）",
                    ),
                )

# Step 2: search (iterative — refine if first round is thin)
            clock.mark("search")
            _emit(bus, StepStarted(step="search", title="检索论文"))
            cache_key = {"keywords": keywords, "max_results": max_results, "sources": options.sources}
            cached = cache.get("search", cache_key) if options.use_cache else None
            search_health = {"cache": "hit"} if cached is not None else {}
            if cached is None:
                papers, search_health = await search_papers_with_health(
                    keywords, max_results, sources=options.sources
                )
                payload = [p.model_dump() for p in papers]
                if options.use_cache:
                    cache.put("search", cache_key, payload)
            else:
                from citens.models import Paper

                papers = [Paper(**p) for p in cached]

            failed_sources = [k for k, v in search_health.items() if v.startswith("failed")]
            persistence.save_step(run_dir, "02_search_health", search_health)
            if failed_sources:
                msg = f"检索源失败: {', '.join(failed_sources)}"
                print(f"  [search] {msg}")
                if len(failed_sources) * 2 >= max(len(search_health), 1):
                    _emit(bus, StepProgress(step="search", message=f"⚠ {msg}（过半源不可用，结果可能单薄）"))
                else:
                    _emit(bus, StepProgress(step="search", message=msg))

            # Iterative refinement: if pool is thin, refine queries and search again
            min_pool = max_papers * 2 if max_papers else 15
            if len(papers) < min_pool and options.mode != RunMode.QUICK_SCAN:
                _emit(
                    bus,
                    StepProgress(step="search", message=f"首轮 {len(papers)} 篇不足，迭代扩展检索…"),
                )
                found_titles = [p.title for p in papers[:10]]
                mined = discover_terms(papers)
                refined = refine_queries(
                    topic, keywords, found_titles, known_gaps=[], discovered_terms=mined
                )
                if mined:
                    _emit(
                        bus,
                        StepProgress(
                            step="search",
                            message=f"从结果中挖掘到领域术语: {', '.join(mined[:5])}",
                        ),
                    )
                if refined:
                    _emit(
                        bus,
                        StepProgress(step="search", message=f"补充查询: {', '.join(refined[:3])}"),
                    )
                    more = await search_papers(refined, max_results=min(max_results, 30), sources=options.sources)
                    # Merge and deduplicate
                    papers = deduplicate(papers + more)
                    keywords = keywords + refined  # track all queries used
                    meta.keywords = keywords

            # Merge resolved landmark seeds into the pool (they still go through
            # filter like everything else — no free pass, just guaranteed
            # candidacy for canonical works the keyword search missed).
            if seed_papers:
                papers = deduplicate(papers + seed_papers)

            # Literature pool (citens collect): inject accumulated records —
            # they carry subfield/keywords/author-engagement metadata — and
            # write this run's finds back so the pool grows with every run.
            if options.use_pool:
                from citens.collect import append_pool, pool_path, recall_from_pool
                from citens.search.filters import parse_constraints

                constraints = parse_constraints(options.filters)
                if pool_path(topic).is_file():
                    # pre-recall keeps LLM screening cost flat as the pool
                    # grows: BM25 picks the top slice (reviews always pass),
                    # the rest stays a deep reservoir
                    pooled = recall_from_pool(
                        topic, keywords, max_results * 2,
                        constraints=constraints,
                        venue_whitelist=_venue_boost,
                    )
                    if pooled:
                        papers = deduplicate(papers + pooled)
                        _emit(
                            bus,
                            StepProgress(
                                step="search",
                                message=(
                                    f"文献池预召回 {len(pooled)} 条"
                                    f"（citens collect 累积池）"
                                ),
                            ),
                        )

                # venue-strict clarification: fetch the top-journal papers the
                # pool lacks instead of filtering whatever it happens to have
                if constraints.venue_strict and _venue_boost:
                    from citens.search.openalex import (
                        resolve_source_ids,
                        search_venue_restricted,
                    )

                    source_ids = resolve_source_ids(
                        [v for v in profile.venue_whitelist if v][:30]
                        if profile is not None else []
                    )
                    if source_ids:
                        restricted = await search_venue_restricted(
                            keywords[:8], source_ids, constraints.year_from
                        )
                        if restricted:
                            papers = deduplicate(papers + restricted)
                            _emit(
                                bus,
                                StepProgress(
                                    step="search",
                                    message=(
                                        f"顶刊受限检索补充 {len(restricted)} 条"
                                        f"（{len(source_ids)} 本白名单期刊"
                                        + (
                                            f"，{constraints.year_from} 年起"
                                            if constraints.year_from
                                            else ""
                                        )
                                        + "）"
                                    ),
                                ),
                            )
                elif constraints.venue_strict:
                    _emit(
                        bus,
                        StepProgress(
                            step="search",
                            message=(
                                "⚠ 仅顶刊约束但未加载领域 profile（--profile finance），"
                                "无法在检索端限制期刊，将仅靠筛选"
                            ),
                        ),
                    )
                added_to_pool = append_pool(topic, papers)
                if added_to_pool:
                    runlog.snapshot("pool_writeback", added=added_to_pool)

            if max_papers:
                # Pool cap before LLM screening. 3×n starved the filter (a
                # -n 8 run screened 24 candidates, and blend_pool's citation
                # trim cut arXiv's zero-cited records hardest — the OA-richest
                # source). 8×n bounded by the retrieval target keeps screening
                # cost flat while giving the filter a real choice.
                papers = blend_pool(
                    papers, cap=min(max(max_papers * 8, 40), max_results)
                )
            meta.total_papers = len(papers)
            runlog.snapshot("search", papers=len(papers), queries=keywords)
            persistence.save_step(run_dir, "02_papers", papers)
            _emit(bus, StepCompleted(step="search", message=f"{len(papers)} 篇候选"))

            # Step 3: filter
            clock.mark("filter")
            _emit(bus, StepStarted(step="filter", title="论文筛选"))

            def _filter_progress(i, total, title):
                _emit(bus, StepProgress(step="filter", message=title, current=i, total=total))

            scored, filter_log = filter_papers(
                papers, topic, filters=options.filters, on_progress=_filter_progress, return_log=True
            )
            # venue-aware composite ranking (relevance x citations x SJR quartile),
            # applied when deciding which papers survive the cap
            scored = rank_papers(scored, venue_boost=_venue_boost)
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
            # supporting-reference layer: relevant papers beyond the core cap
            # become abstract-only citations instead of being discarded —
            # the bibliography no longer equals the deep-dive set
            support_n = (
                options.support_papers
                if options.support_papers is not None
                else settings.default_support_papers
            )
            supporting = []
            if support_n and max_papers and len(scored) > max_papers:
                supporting = [
                    p
                    for p in scored[max_papers:]
                    if p.relevance_score >= 3 and len(p.abstract.strip()) >= 80
                ][:support_n]
            if max_papers and len(scored) > max_papers:
                scored = scored[:max_papers]
            meta.filtered_papers = len(scored)
            runlog.snapshot("filter", papers=len(scored), supporting=len(supporting))
            persistence.save_step(run_dir, "03_filtered", scored)
            if supporting:
                persistence.save_step(
                    run_dir,
                    "03d_supporting",
                    [
                        {"title": p.title[:80], "venue": p.venue,
                         "citations": p.citation_count, "score": p.relevance_score}
                        for p in supporting
                    ],
                )
                _emit(
                    bus,
                    StepProgress(
                        step="filter",
                        message=f"另保留 {len(supporting)} 篇支持文献（仅摘要引用，充实参考文献）",
                    ),
                )
            hist = quartile_histogram(scored)
            hist_msg = " · ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
            _emit(bus, StepCompleted(step="filter", message=f"{len(scored)} 篇通过（{hist_msg}）"))
            # constraint-strictness warning: a tiny pass set under a strict
            # clarification (顶刊/近5年/实证) starves BOTH the core and the
            # supporting layer — the user should know why, not wonder
            if max_papers and len(scored) < min(6, max_papers):
                from citens.search.filters import parse_constraints as _pc

                c = _pc(options.filters)
                why = c.describe() or "澄清约束"
                _emit(
                    bus,
                    StepProgress(
                        step="filter",
                        message=(
                            f"⚠ 仅 {len(scored)} 篇通过（约束: {why}）。"
                            "候选池内符合约束的文献不足——建议放宽一档（如'顶刊+计算机顶会'、"
                            "'近10年'），或先 citens collect 补池"
                        ),
                    ),
                )

            # Step 3.2: citation snowballing — expand pool via refs/citations of top papers
            if options.mode != RunMode.QUICK_SCAN:
                clock.mark("snowball")
                _emit(bus, StepStarted(step="snowball", title="引用滚雪球"))
                top_seeds = [p for p in seed_papers if p.doi] + [
                    p for p in scored[:3] if p.doi
                ]
                # dedupe seeds by doi (landmarks first — they anchor the graph)
                _seen_dois: set[str] = set()
                deduped_seeds: list = []
                for p in top_seeds:
                    if p.doi not in _seen_dois:
                        _seen_dois.add(p.doi)
                        deduped_seeds.append(p)
                top_seeds = deduped_seeds[:5]
                existing = {p.id for p in scored}
                snowballed = await snowball(top_seeds, existing, limit_per_paper=6)
                if snowballed:
                    _emit(
                        bus,
                        StepProgress(step="snowball", message=f"滚雪球发现 {len(snowballed)} 篇候选"),
                    )
                    # Filter the snowballed papers too
                    snow_scored = filter_papers(snowballed, topic, filters=options.filters)
                    snow_scored = rank_papers(snow_scored, venue_boost=_venue_boost)
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
                clock.mark("enrich")
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

            # blind papers (no abstract even after enrichment, no OA pdf) can
            # be neither extracted nor verified — demote them to the
            # supporting layer and backfill the core from abstract-bearing
            # alternates, so -n means "-n verifiable papers"
            if max_papers:
                scored, supporting, swap_log = demote_blind_papers(scored, supporting)
                if swap_log:
                    meta.filtered_papers = len(scored)
                    persistence.save_step(run_dir, "03e_blind_demotion", swap_log)
                    n_demoted = sum(1 for s in swap_log if s["action"] == "demoted_to_supporting")
                    n_kept = len(swap_log) - n_demoted
                    if n_demoted:
                        msg = (
                            f"⚠ {n_demoted} 篇无摘要论文降级为支持文献（无法抽取/核验），由次序递补"
                        )
                        if n_kept:
                            msg += f"；{n_kept} 篇无递补者，暂留核心集"
                        _emit(bus, StepProgress(step="enrich", message=msg))

            # Step 4: extract
            clock.mark("extract")
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
        # one ChunkStore for ALL compose rounds — ground text for a paper is
        # fetched, parsed and chunked exactly once however often we recompose
        shared_store = ChunkStore()

        def _collect_verify_queries(queries, _msg):
            verify_trigger_queries.extend(queries)

        result = _compose(
            extracted,
            topic,
            run_dir,
            bus,
            fetch_fulltext=options.fetch_fulltext,
            on_supplement_queries=_collect_verify_queries,
            clock=clock,
            label="compose1",
            chunk_store=shared_store,
            terminology=profile.terminology if profile is not None else None,
            supporting=supporting,
            facets=facets,
            verdict_cache=verdict_cache,
            verdict_fuzzy=verdict_fuzzy,
            n_candidates=meta.total_papers,
            evidence_bias=profile.evidence_bias if profile is not None else "number_density",
        )
        meta.themes = [t.name for t in result.themes.themes]

        # Step 6: reflect -> supplement -> recompose, bounded loop with a
        # saturation stop (keep retrieving only while new papers keep arriving)
        if options.allow_supplement:
            clock.mark("reflect")
            _emit(bus, StepStarted(step="reflect", title="反思与补充"))
            # configurable ceiling (deep used to hardcode 2 — round 2 alone
            # cost ~37 min in the 08-19 run; see settings.reflect_max_rounds)
            max_rounds = max(1, settings.reflect_max_rounds)
            rounds_run = 0
            saturated = False
            while rounds_run < max_rounds:
                rounds_run += 1
                cov_lines = "; ".join(
                    f"{r['facet']}={r['papers']}" for r in facet_coverage_report(facets, extracted)
                )
                dead = [
                    k for k, v in search_health.items()
                    if v == "empty" or str(v).startswith("failed")
                ]
                channel = (
                    " | 渠道状态: " + ", ".join(f"{k}={search_health[k]}" for k in dead)
                    if dead else ""
                )
                decision = reflect(
                    result.synthesis, topic, len(extracted),
                    coverage=(cov_lines + channel),
                )

                # 6a: absence audit — canonical works the set is missing
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

                # merge the three query sources: reflect gaps + absence
                # audit + verifier-triggered (unsupported-claim evidence)
                supplement_queries = list(dict.fromkeys(
                    decision.get("supplementary_keywords", [])
                    + audit_queries
                    + verify_trigger_queries
                ))[:8]

                if not (decision["needs_supplement"] and supplement_queries):
                    _emit(
                        bus,
                        StepCompleted(
                            step="reflect",
                            message=decision["rationale"] or "覆盖充分，无需补充",
                        ),
                    )
                    saturated = True
                    break

                _emit(
                    bus,
                    StepProgress(
                        step="reflect",
                        message=f"round {rounds_run}/{max_rounds} 补充检索: "
                        + ", ".join(supplement_queries),
                    ),
                )
                fresh, all_known = await _supplement_search(
                    supplement_queries, {p.id for p in extracted}, options, topic
                )
                if options.enrich_abstracts and fresh:
                    enrich_abstracts(fresh)
                fresh, supporting, blind_supp = _gate_supplement_papers(fresh, supporting)
                if blind_supp:
                    _emit(
                        bus,
                        StepProgress(
                            step="reflect",
                            message=f"{len(blind_supp)} 篇补充论文无摘要，降级为支持文献（仅题录）",
                        ),
                    )
                if not fresh:
                    # saturation: no NEW relevant paper — extra rounds would
                    # only re-find what the pool already has
                    msg = (
                        "检索饱和（命中均已在池内），停止补充"
                        if all_known
                        else "无新论文，停止补充"
                    )
                    _emit(bus, StepCompleted(step="reflect", message=msg))
                    saturated = True
                    break

                _emit(
                    bus,
                    StepProgress(step="reflect", message=f"补充到 {len(fresh)} 篇新论文，重新综合"),
                )
                new_extracted = extract_papers(fresh, topic)
                extracted = extracted + new_extracted
                meta.filtered_papers = len(extracted)
                # unique name per round — the old fixed name lost round 1
                # when round 2 overwrote it
                persistence.save_step(
                    run_dir,
                    f"08_supplement_r{rounds_run}",
                    {
                        "round": rounds_run,
                        "new_papers": fresh,
                        "decision": decision,
                        "audit": audit,
                    },
                )
                runlog.snapshot(
                    "supplement",
                    round=rounds_run,
                    added=[p.title for p in fresh],
                    total=len(extracted),
                )
                result = _compose(
                    extracted,
                    topic,
                    run_dir,
                    bus,
                    fetch_fulltext=options.fetch_fulltext,
                    on_supplement_queries=_collect_verify_queries,
                    clock=clock,
                    label=f"compose{rounds_run + 1}",
                    chunk_store=shared_store,
                    terminology=profile.terminology if profile is not None else None,
                    supporting=supporting,
                    facets=facets,
                    verdict_cache=verdict_cache,
                    verdict_fuzzy=verdict_fuzzy,
                    n_candidates=meta.total_papers,
                    evidence_bias=profile.evidence_bias if profile is not None else "number_density",
                )
                meta.themes = [t.name for t in result.themes.themes]
                _emit(
                    bus,
                    StepProgress(
                        step="reflect",
                        message=f"round {rounds_run}: 补充 {len(new_extracted)} 篇并重新综合完成",
                    ),
                )
            else:
                # loop exhausted max_rounds without saturating
                _emit(
                    bus,
                    StepCompleted(
                        step="reflect",
                        message=f"达到补检轮数上限（{max_rounds}），仍有缺口可手动继续",
                    ),
                )
            persistence.save_step(
                run_dir,
                "09_saturation",
                {"rounds_run": rounds_run, "max_rounds": max_rounds, "saturated": saturated},
            )

        # finalize
        clock.mark("finalize")
        # the final paper set (post-supplement) was previously only in memory —
        # persist it so audit tools and resume don't have to reconstruct it
        persistence.save_step(run_dir, "09_final_papers", extracted)
        usage = runlog.finalize()
        meta.review_path = result.review_path
        meta.citation_precision = round(result.precision, 3)
        persistence.save_json(run_dir, "meta.json", meta)
        # self-contained HTML browser over the run's artifacts (claims,
        # verdicts, evidence anchors, downloads); regenerate-able any time
        # via `citens browse <run_dir>`
        browser_path = None
        with contextlib.suppress(Exception):
            browser_path = write_review_browser(run_dir)
        timings = clock.payload()
        timings["token_usage_by_stage"] = usage.get("token_usage_by_stage", {})
        timings["total_tokens"] = usage.get("total_tokens", 0)
        persistence.save_json(run_dir, "timings.json", timings)
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
                    "seconds": clock.total(),
                    **({"browser": browser_path} if browser_path else {}),
                },
            ),
        )
        return meta

    except Exception as e:  # noqa: BLE001
        clock.mark("failed")
        with contextlib.suppress(Exception):
            persistence.save_json(run_dir, "timings.json", clock.payload())
        _emit(bus, RunFailed(message=str(e), step="pipeline"))
        traceback.print_exc()
        raise
    finally:
        scope.__exit__(None, None, None)


def run_pipeline(
    topic: str,
    options: RunOptions | None = None,
    bus: EventBus | None = None,
) -> RunMeta:
    """Sync entry point (for CLI). Runs the async pipeline via asyncio.run."""
    return asyncio.run(run_pipeline_async(topic, options, bus))
