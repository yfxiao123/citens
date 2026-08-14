"""Information-extraction agent: structured fields from each paper's abstract."""

from __future__ import annotations

from litreview.llm import chat_json
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


def extract_papers(
    papers: list[ScoredPaper],
    topic: str,
    *,
    on_progress=None,
) -> list[ExtractedPaper]:
    """Extract structured fields for each filtered paper."""
    extracted: list[ExtractedPaper] = []
    total = len(papers)
    for i, paper in enumerate(papers):
        if on_progress:
            on_progress(i + 1, total, paper.title[:50])
        user_prompt = (
            f"研究主题 / Topic: {topic}\n\n"
            f"论文信息 / Paper:\n{paper.brief()}\n\n"
            "Extract the structured information."
        )
        try:
            result = chat_json(SYSTEM_PROMPT, user_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"    extract failed: {e}")
            result = {}
        extracted.append(
            ExtractedPaper(
                **paper.model_dump(exclude={"id"}),
                research_question=result.get("research_question", ""),
                methodology=result.get("methodology", ""),
                key_findings=result.get("key_findings", []),
                limitations=result.get("limitations", []),
                relevance_to_topic=result.get("relevance_to_topic", ""),
            )
        )
    return extracted
