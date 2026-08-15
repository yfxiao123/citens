"""Relevance-filtering agent: score each candidate paper 1-5, keep >= 3."""

from __future__ import annotations

from litreview.llm import chat_json
from litreview.models import Paper, ScoredPaper

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


def filter_papers(
    papers: list[Paper],
    topic: str,
    *,
    on_progress=None,
    return_log: bool = False,
) -> list[ScoredPaper] | tuple[list[ScoredPaper], list[dict]]:
    """Score every paper; return those with score >= 3 as ScoredPaper.

    Args:
        papers: List of candidate papers
        topic: Research topic
        on_progress: Progress callback
        return_log: If True, return (passed_papers, filter_log) where filter_log
                    contains detailed exclusion reasons for all papers

    Returns:
        List of ScoredPaper with score >= 3, or (passed, log) if return_log=True
    """
    scored: list[ScoredPaper] = []
    filter_log: list[dict] = []
    total = len(papers)
    for i, paper in enumerate(papers):
        if on_progress:
            on_progress(i + 1, total, paper.title[:50])
        user_prompt = (
            f"研究主题 / Topic: {topic}\n\n"
            f"论文信息 / Paper:\n{paper.brief()}\n\n"
            "Rate relevance and justify."
        )
        try:
            result = chat_json(SYSTEM_PROMPT, user_prompt)
            score = int(result.get("score", 1))
            reason = result.get("reason", "")
        except Exception as e:  # noqa: BLE001
            print(f"    score failed: {e}")
            score = 2
            reason = "scoring error, defaulting low"
        
        scored_paper = ScoredPaper(
            **paper.model_dump(exclude={"id"}),
            relevance_score=score,
            filter_reason=reason,
        )
        scored.append(scored_paper)
        
        # Log the decision
        filter_log.append({
            "title": paper.title,
            "authors": paper.authors[:3],
            "year": paper.year,
            "doi": paper.doi,
            "score": score,
            "reason": reason,
            "passed": score >= 3,
        })
    
    passed = [p for p in scored if p.relevance_score >= 3]
    
    if return_log:
        return passed, filter_log
    return passed
