"""Relevance-filtering agent: score each candidate paper 1-5, keep >= 3.

Scoring runs in BATCHES (default 8 papers per LLM call) on the thread pool
(:func:`citens.llm.run_concurrent`) — a 300-paper candidate pool costs ~40
calls, not 300. Papers a batch fails to score (parse gaps, truncation) fall
back to a single-paper call, so a bad batch never drops papers silently.
"""

from __future__ import annotations

from typing import Literal, overload

from citens.agents.scoping import filters_block, min_year_from_filters
from citens.llm import chat_json, run_concurrent
from citens.models import Paper, ScoredPaper

SYSTEM_PROMPT = """You are an academic literature-screening expert. Given a research topic and \
one paper's metadata, you must:
1. Rate relevance 1-5:
   5 = directly on-topic, core literature
   4 = highly relevant
   3 = partially relevant
   2 = tangentially related
   1 = irrelevant
2. Give a 20-50 word justification.
3. Only papers scoring >= 3 are kept.

Output JSON:
{"score": 3, "reason": "This paper directly addresses the topic's core method..."}"""

BATCH_SYSTEM_PROMPT = """You are an academic literature-screening expert. You are given a research \
topic and SEVERAL candidate papers (numbered). For EACH paper, rate relevance 1-5:
   5 = directly on-topic, core literature
   4 = highly relevant
   3 = partially relevant
   2 = tangentially related
   1 = irrelevant
and give a 20-50 word justification. Judge every paper independently on its own merits.

Output JSON with one entry per paper (use the given paper_index):
{"results": [
  {"paper_index": 0, "score": 4, "reason": "..."},
  {"paper_index": 1, "score": 2, "reason": "..."}
]}"""

_BATCH_SIZE = 8


@overload
def filter_papers(
    papers: list[Paper],
    topic: str,
    *,
    filters: dict | None = ...,
    on_progress=...,
    return_log: Literal[False] = ...,
) -> list[ScoredPaper]: ...


@overload
def filter_papers(
    papers: list[Paper],
    topic: str,
    *,
    filters: dict | None = ...,
    on_progress=...,
    return_log: Literal[True],
) -> tuple[list[ScoredPaper], list[dict]]: ...


def filter_papers(
    papers: list[Paper],
    topic: str,
    *,
    filters: dict | None = None,
    on_progress=None,
    return_log: bool = False,
) -> list[ScoredPaper] | tuple[list[ScoredPaper], list[dict]]:
    """Score every paper (concurrently); return those with score >= 3.

    Args:
        papers: List of candidate papers
        topic: Research topic
        filters: Pre-run clarification answers — appended to the scoring
            prompt, plus a deterministic year floor (see scoping.py)
        on_progress: Progress callback ``fn(done, total, title)``
        return_log: If True, return (passed_papers, filter_log) where filter_log
                    contains detailed exclusion reasons for all papers

    Returns:
        List of ScoredPaper with score >= 3, or (passed, log) if return_log=True
    """
    constraints = filters_block(filters)
    min_year = min_year_from_filters(filters)
    total = len(papers)

    def _to_scored(paper: Paper, score: int, reason: str) -> tuple[ScoredPaper, dict]:
        # Deterministic enforcement of a stated timeframe: an LLM might let a
        # 1998 paper slip past "近5年"; the parse never does.
        if min_year and paper.year and paper.year < min_year:
            score = min(score, 1)
            reason = f"outside user timeframe (< {min_year}); {reason}".strip()

        scored_paper = ScoredPaper(
            **paper.model_dump(exclude={"id"}),
            relevance_score=score,
            filter_reason=reason,
        )
        log_entry = {
            "title": paper.title,
            "authors": paper.authors[:3],
            "year": paper.year,
            "doi": paper.doi,
            "score": score,
            "reason": reason,
            "passed": score >= 3,
        }
        return scored_paper, log_entry

    def _score_one(paper: Paper) -> tuple[ScoredPaper, dict]:
        user_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"{constraints}"
            f"\n论文信息 / Paper:\n{paper.brief()}\n\n"
            "Rate relevance and justify."
        )
        try:
            # A score + short reason is tiny; the small budget keeps reasoning
            # models from padding generation. chat_json retries larger if the
            # thinking squeezes the JSON out.
            result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=1024)
            score = int(result.get("score", 1))
            reason = str(result.get("reason", ""))
        except Exception as e:  # noqa: BLE001
            print(f"    score failed: {e}")
            score = 2
            reason = "scoring error, defaulting low"
        return _to_scored(paper, score, reason)

    def _score_batch(batch: list[Paper]) -> list[tuple[ScoredPaper, dict]]:
        """Score one batch in a single call; per-paper fallback for gaps."""
        paper_lines = "\n\n".join(
            f"--- Paper {i} ---\n{p.brief()}" for i, p in enumerate(batch)
        )
        user_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"{constraints}"
            f"\n候选论文 / Candidate papers:\n{paper_lines}\n\n"
            f"Rate ALL {len(batch)} papers."
        )
        by_index: dict[int, dict] = {}
        try:
            result = chat_json(BATCH_SYSTEM_PROMPT, user_prompt, max_tokens=4096)
            for entry in result.get("results", []):
                if isinstance(entry, dict) and "paper_index" in entry:
                    try:
                        by_index[int(entry["paper_index"])] = entry
                    except (TypeError, ValueError):
                        continue
        except Exception as e:  # noqa: BLE001
            print(f"    batch scoring failed ({e}); falling back to per-paper")

        out: list[tuple[ScoredPaper, dict]] = []
        for i, paper in enumerate(batch):
            entry = by_index.get(i)
            if entry is None:
                out.append(_score_one(paper))
                continue
            try:
                score = int(entry.get("score", 1))
            except (TypeError, ValueError):
                score = 1
            reason = str(entry.get("reason", ""))
            out.append(_to_scored(paper, score, reason))
        return out

    if not papers:
        return ([], []) if return_log else []

    batches = [papers[s : s + _BATCH_SIZE] for s in range(0, total, _BATCH_SIZE)]

    done = 0

    def _on_done(_i, batch, _result):
        nonlocal done
        done += len(batch)
        if on_progress:
            on_progress(done, total, f"批次 {_i + 1}/{len(batches)}")

    results = run_concurrent(
        lambda _i, b: _score_batch(b), batches, on_done=_on_done
    )
    scored = [r[0] for pair in results for r in pair]
    filter_log = [r[1] for pair in results for r in pair]

    passed = [p for p in scored if p.relevance_score >= 3]

    if return_log:
        return passed, filter_log
    return passed
