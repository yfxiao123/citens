"""Pipeline support layer: pure helper functions and small utilities that
surround the compose/run stages but don't belong in ``pipeline.py`` itself.

Everything here is either deterministic (no LLM, no network) or a thin
run-directory accessor, so it is safe to import from reverify / tests /
future stages. ``pipeline.py`` re-exports all public names — patch
``citens.orchestration.pipeline.*`` for LLM/search seams, not these.
"""

from __future__ import annotations

import re
import time

from citens.events import Event, EventBus
from citens.grounding import ChunkStore
from citens.models import ExtractedPaper, RunMode
from citens.runlog import RunLog

# --- writer evidence selection ------------------------------------------------

_NUMBER_DENSITY_RE = None  # compiled lazily; keep module import cheap


def _number_density(text: str) -> int:
    """Count effect-size-shaped tokens (37%, 0.92, 1.5M-style decimals)."""
    global _NUMBER_DENSITY_RE
    if _NUMBER_DENSITY_RE is None:
        _NUMBER_DENSITY_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\d+\.\d+")
    return len(_NUMBER_DENSITY_RE.findall(text))


def number_dense_excerpts(
    paper_indices: list[int],
    extracted: list[ExtractedPaper],
    chunk_store: ChunkStore,
    *,
    query: str,
    per_paper: int = 3,
    excerpt_chars: int = 900,
    budget: int = 7500,
    bias: str = "number_density",
) -> str:
    """Full-text excerpts for the writer, biased toward number-bearing chunks.

    Plain BM25 top-k favors intro/method prose; effect sizes live in results
    sections and tables deeper in the paper. Candidates are re-ranked with a
    number-density boost (BM25 order preserved within tiers) before the top
    `per_paper` chunks are excerpted — the writer's EVIDENCE RULE then carries
    the numbers into the review body.

    ``bias="none"`` (theoretical/mathematical domains, via the profile's
    ``evidence_bias``) keeps plain BM25 order — numbers don't mark the
    load-bearing passages there.
    """
    from citens.models import ChunkKind

    parts: list[str] = []
    for idx in paper_indices:
        if not 0 <= idx < len(extracted):
            continue
        p = extracted[idx]
        candidates = [
            c for c in chunk_store.retrieve(p.id, query, k=per_paper * 2)
            if c.kind == ChunkKind.FULLTEXT
        ]
        if bias == "number_density":
            candidates.sort(key=lambda c: -_number_density(c.text))
        for c in candidates[:per_paper]:
            excerpt = c.text[:excerpt_chars]
            parts.append(f"[{idx}] {p.title[:60]} — {excerpt}")
            budget -= len(excerpt)
            if budget <= 0:
                return "\n\n".join(parts)
    return "\n\n".join(parts)


# --- timing --------------------------------------------------------------------


class StepClock:
    """Ordered wall-time marks; a stage's duration is the delta to the next mark.

    Cheap enough to leave on every run — timings.json is how you answer
    "why was this run slow?" after the fact (see the 100-paper case).
    Also mirrors every mark into the run's append-only log when one is given.
    """

    def __init__(self, runlog: RunLog | None = None) -> None:
        self._marks: list[tuple[str, float]] = [("_start", time.monotonic())]
        self.runlog = runlog

    def mark(self, name: str) -> None:
        self._marks.append((name, time.monotonic()))
        if self.runlog is not None:
            self.runlog.mark(name)

    def durations(self) -> dict[str, float]:
        return {
            f"{prev}->{name}": round(t1 - t0, 1)
            for (prev, t0), (name, t1) in zip(self._marks, self._marks[1:], strict=False)
        }

    def total(self) -> float:
        return round(self._marks[-1][1] - self._marks[0][1], 1)

    def payload(self) -> dict:
        return {"total_seconds": self.total(), "stages": self.durations()}


# --- resume helpers --------------------------------------------------------------


def _load_extracted_for_resume(run_dir: str) -> list[ExtractedPaper]:
    """Load a previous run's 04_extracted.json, or raise with a clear message."""
    import json
    import os

    path = os.path.join(run_dir, "steps", "04_extracted.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"cannot resume {run_dir}: no steps/04_extracted.json "
            "(the run must have completed the extraction step)"
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [ExtractedPaper(**p) for p in raw]


def _mode_from_run_dir(run_dir: str) -> RunMode:
    """Recover the run mode persisted by step 0 (default: deep_review)."""
    import json
    import os

    path = os.path.join(run_dir, "steps", "00_intent.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return RunMode(json.load(f).get("mode", "deep_review"))
        except ValueError:
            pass
    return RunMode.DEEP_REVIEW


# --- events ---------------------------------------------------------------------


def _emit(bus: EventBus | None, event: Event) -> None:
    if bus is not None:
        bus.emit(event)


# --- pool hygiene -----------------------------------------------------------------


def demote_blind_papers(
    scored: list, supporting: list
) -> tuple[list, list, list[dict]]:
    """Swap abstract-less core papers for abstract-bearing supporting papers.

    A "blind" paper (no abstract and no OA pdf_url) can be neither extracted
    nor verified — a core slot spent on one is a wasted slot (2026-08-18 run:
    7/20 core papers were blind). Demotion is not deletion: blind papers join
    the supporting layer as bibliography entries; the next-ranked
    abstract-bearing alternates backfill the core so ``-n`` keeps meaning
    "-n verifiable papers". Returns (core, supporting, swap_log).
    """
    def _blind(p) -> bool:
        return not p.abstract.strip() and not (p.pdf_url or "").strip()

    blind = [p for p in scored if _blind(p)]
    if not blind:
        return scored, supporting, []
    alts = [p for p in supporting if not _blind(p)][: len(blind)]
    n_swap = min(len(blind), len(alts))
    if not n_swap:
        return scored, supporting, [
            {"title": p.title[:80], "action": "kept_blind_no_alternates"}
            for p in blind
        ]
    blind_out = blind[:n_swap]
    alts_in = alts[:n_swap]
    out_ids = {p.id for p in blind_out}
    in_ids = {p.id for p in alts_in}
    new_core = [p for p in scored if p.id not in out_ids] + alts_in
    new_core.sort(key=lambda p: getattr(p, "rank_score", 0.0), reverse=True)
    new_supporting = (
        [p for p in supporting if p.id not in in_ids] + blind_out
    )
    swap_log = [
        {"title": b.title[:80], "action": "demoted_to_supporting",
         "replaced_by": a.title[:80]}
        for b, a in zip(blind_out, alts_in, strict=False)
    ] + [
        {"title": p.title[:80], "action": "kept_blind_no_alternates"}
        for p in blind[n_swap:]
    ]
    return new_core, new_supporting, swap_log


def _gate_supplement_papers(
    fresh: list, supporting: list
) -> tuple[list, list, list]:
    """The reflect path's blind-paper gate (the main path demotes after
    enrichment; supplements used to skip it — 7/26 blind core papers in the
    2026-08-19 order-book run came in this way). Blind supplements (no
    abstract, no OA pdf) drop to the supporting layer as bibliography-only."""
    blind = [
        p for p in fresh
        if not p.abstract.strip() and not (p.pdf_url or "").strip()
    ]
    if not blind:
        return fresh, supporting, []
    blind_ids = {p.id for p in blind}
    return (
        [p for p in fresh if p.id not in blind_ids],
        supporting + blind,
        blind,
    )


# --- facet coverage (the coverage-by-design layer) ---------------------------


def facet_coverage_report(facets: list, papers: list) -> list[dict]:
    """Deterministic per-facet paper counts (heuristic term overlap).

    A facet "counts" a paper when any 4+ letter term from its queries appears
    in the paper's title or abstract. Crude but reproducible — good enough to
    name thin facets for the reflector and the writer's honesty paragraph.
    """
    out = []
    for f in facets or []:
        terms = {
            w
            for q in (f.get("queries") or [])
            for w in re.findall(r"[a-z]{4,}", q.lower())
        }
        n = sum(
            1
            for p in papers
            if terms
            & set(re.findall(r"[a-z]{4,}", f"{p.title} {getattr(p, 'abstract', '') or ''}".lower()))
        )
        out.append({"facet": f.get("name", ""), "papers": n})
    return out


def coverage_note_text(
    report: list[dict], themes: list, n_blind: int
) -> str:
    """Compose the writer's coverage-honesty paragraph from hard numbers."""
    parts = []
    thin = [r for r in report if r["papers"] < 3]
    if thin:
        parts.append(
            "检索面覆盖薄弱（<3篇）: "
            + ", ".join(f"{r['facet']}({r['papers']}篇)" for r in thin)
        )
    thin_themes = [t.name for t in themes if len(t.paper_indices) < 3]
    if thin_themes:
        parts.append("主题论文数偏少: " + ", ".join(thin_themes))
    if n_blind:
        parts.append(f"{n_blind} 篇论文无摘要（仅题录级信息，无法逐条核验其细节）")
    return "; ".join(parts)


# --- citation hygiene ---------------------------------------------------------


def prune_citation_stacking(
    review: str, papers: list, *, max_cites: int = 4
) -> tuple[str, list[dict]]:
    """Hard cap on citation stacking: sentences wearing more than ``max_cites``
    [n] markers keep only the most relevant cited papers (BM25 of the sentence
    against each cited paper's title+abstract); the rest of the markers are
    stripped. The prompt cap softens; this enforces. Returns (review, log)."""
    from citens.grounding.retrieval import bm25_rank_texts

    log: list[dict] = []
    out: list[str] = []
    for sent in re.split(r"(?<=。)", review):
        idxs = sorted({int(m) for m in re.findall(r"\[(\d+)\]", sent)})
        if len(idxs) <= max_cites or not any(0 <= i < len(papers) for i in idxs):
            out.append(sent)
            continue
        s_text = re.sub(r"\[\d+\]", "", sent)
        corpus = [
            f"{papers[i].title} {getattr(papers[i], 'abstract', '') or ''}"[:400]
            if 0 <= i < len(papers) else " "
            for i in idxs
        ]
        ranked = bm25_rank_texts(corpus, s_text)
        keep = {idxs[j] for j in ranked[:max_cites]}
        dropped = [i for i in idxs if i not in keep]
        if dropped:
            new_sent = re.sub(
                r"\[(\d+)\]",
                lambda m, _keep=keep: m.group(0) if int(m.group(1)) in _keep else "",
                sent,
            )
            out.append(new_sent)
            log.append({"kept": sorted(keep), "dropped": dropped})
        else:
            out.append(sent)
    return "".join(out), log


# --- methodology statement ----------------------------------------------------


def search_summary_text(
    sources: list | None,
    n_candidates: int,
    n_core: int,
    n_support: int,
    date_str: str = "",
) -> str:
    """The review's own retrieval-methodology statement — every number here is
    measured by this run (candidates screened, papers included), which is what
    the introduction reports PRISMA-style."""
    import datetime as _dt

    src = "、".join(sources) if sources else "arXiv、Semantic Scholar、OpenAlex、Crossref"
    parts = [
        f"检索源：{src}",
        f"共获得 {n_candidates} 篇候选论文",
        f"经相关性筛选与复合排序（相关性×引用×期刊分区）纳入核心文献 {n_core} 篇",
    ]
    if n_support:
        parts.append(f"另保留 {n_support} 篇支持文献（仅摘要引用）")
    parts.append(f"检索日期：{date_str or _dt.date.today().isoformat()}")
    return "；".join(parts) + "。"
