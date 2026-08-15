"""Theme-organization agent: cluster papers into 3-6 thematic dimensions."""

from __future__ import annotations

from citens.llm import chat_json
from citens.models import ExtractedPaper, ThemeInfo, ThemeStructure

SYSTEM_PROMPT = """You are an academic theme-organization expert. Given a research topic and the \
structured information of several papers, you must:
1. Identify 3-6 thematic dimensions covering the field's main directions.
2. Assign each paper to its best-fitting theme, with a grouping rationale.
3. Within each theme, articulate inter-paper logical relations (evolution, \
contrast, complementarity, etc.).

Output JSON:
{
  "themes": [
    {
      "name": "theme name",
      "description": "theme description",
      "paper_indices": [0, 1, 2],
      "grouping_reason": "why these papers belong here",
      "logical_relations": "logical relations among them"
    }
  ]
}"""


def organize_themes(papers: list[ExtractedPaper], topic: str) -> ThemeStructure:
    """Cluster papers into 3-6 themes."""
    parts = []
    for i, p in enumerate(papers):
        parts.append(
            f"\n--- Paper {i} ---\n"
            f"标题: {p.title}\n"
            f"研究问题: {p.research_question}\n"
            f"方法: {p.methodology}\n"
            f"发现: {'; '.join(p.key_findings)}\n"
            f"局限: {'; '.join(p.limitations)}\n"
            f"与主题关系: {p.relevance_to_topic}\n"
        )
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"论文列表 / Papers ({len(papers)}):\n{''.join(parts)}\n\n"
        "Identify 3-6 themes and assign papers to them."
    )
    result = chat_json(SYSTEM_PROMPT, user_prompt)
    themes = [ThemeInfo(**t) for t in result.get("themes", [])]
    return ThemeStructure(themes=themes)
