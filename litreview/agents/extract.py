"""Information-extraction agent: structured fields from each paper's abstract.

Per-paper calls are independent — they run on a thread pool
(:func:`litreview.llm.run_concurrent`) so a 9-paper extract stage takes one
round-trip instead of nine.
"""

from __future__ import annotations

from litreview.llm import chat_json, run_concurrent
from litreview.models import ExtractedPaper, ScoredPaper

SYSTEM_PROMPT = """You are an academic-paper analyst. Given a research topic and a paper, extract \
structured information from its abstract.

Output JSON (include every field):
{
  "research_question": "what problem the paper tackles",
  "methodology": "technical approach / method used",
  "key_findings": ["core conclusion 1", "core conclusion 2", "core conclusion 3"],
  "limitations": ["limitation 1", "limitation 2"],
  "relevance_to_topic": "this paper's value to the review"
}"""


def _extract_one(paper: ScoredPaper, topic: str) -> ExtractedPaper:
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"论文信息 / Paper:\n{paper.brief()}\n\n"
        "Extract the structured information."
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt)
    except Exception as e:  # noqa: BLE001
        print(f"    extract failed ({paper.title[:40]}): {e}")
        result = {}
    return ExtractedPaper(
        **paper.model_dump(exclude={"id"}),
        research_question=result.get("research_question", ""),
        methodology=result.get("methodology", ""),
        key_findings=result.get("key_findings", []),
        limitations=result.get("limitations", []),
        relevance_to_topic=result.get("relevance_to_topic", ""),
    )


def extract_papers(
    papers: list[ScoredPaper],
    topic: str,
    *,
    on_progress=None,
) -> list[ExtractedPaper]:
    """Extract structured fields for each filtered paper (concurrently)."""
    total = len(papers)

    def on_done(i, paper, _result):
        if on_progress:
            on_progress(i + 1, total, paper.title[:50])

    return run_concurrent(
        lambda i, p: _extract_one(p, topic), list(papers), on_done=on_done
    )
