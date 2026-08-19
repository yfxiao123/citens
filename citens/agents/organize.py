"""Theme-organization agent: cluster papers into 3-6 thematic dimensions."""

from __future__ import annotations

from citens.config import settings
from citens.llm import chat_json
from citens.models import ExtractedPaper, ThemeInfo, ThemeStructure

SYSTEM_PROMPT = """You are an academic theme-organization expert. Given a research topic and the \
structured information of several papers, you must:
1. Identify 3-6 thematic dimensions covering the field's main directions.
2. Choose an ORGANIZING PRINCIPLE and stay consistent: group by mechanism, by method \
family, by application, or by historical evolution — pick the axis that best maps this \
field, and say which axis you chose in the theme descriptions.
3. Assign each paper to its best-fitting theme, with a grouping rationale.
4. Within each theme, articulate inter-paper logical relations (evolution, \
contrast, complementarity, etc.) — the theme description should read like part \
of an argument chain: scope -> organizing principle -> synthesis -> disagreements \
-> open questions.

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
    """Cluster papers into 3-6 themes.

    Degrade, never die: a truncated/garbled judge response (reasoning models
    occasionally blow the output budget mid-string) falls back to a
    deterministic rank-order grouping instead of killing the run — a
    mechanical theme structure still lets compose proceed.
    """
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
        "Identify 3-6 themes and assign papers to them. Keep each description "
        "under 80 characters so the JSON stays small.\n"
        + _localization_line()
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=8192, strong=True,
                           thinking=settings.judge_thinking)
        themes = [ThemeInfo(**t) for t in result.get("themes", [])]
    except Exception as e:  # noqa: BLE001
        print(f"    [organize] LLM theme organization failed ({e}); "
              "falling back to rank-order grouping")
        themes = []
    if themes:
        return ThemeStructure(themes=themes)
    return _fallback_themes(papers)


def _fallback_themes(papers: list[ExtractedPaper]) -> ThemeStructure:
    """Deterministic grouping: 3-4 contiguous rank-order chunks, generic names.

    The '(自动分组)' marker surfaces in the review headings so a reader can
    see that theme organization degraded for this run."""
    n = len(papers)
    if n == 0:
        return ThemeStructure(themes=[])
    k = 3 if n < 12 else 4
    size = (n + k - 1) // k
    themes = []
    for t in range(k):
        lo, hi = t * size, min((t + 1) * size, n)
        if lo >= hi:
            break
        themes.append(
            ThemeInfo(
                name=f"主题 {t + 1}（自动分组）",
                description="主题组织降级：按检索排序自动分组，非语义聚类",
                paper_indices=list(range(lo, hi)),
                grouping_reason="rank-order fallback",
                logical_relations="",
            )
        )
    return ThemeStructure(themes=themes)


def _localization_line() -> str:
    """Theme names must match the review's prose language — they become the
    section headings the writer renders as ``###``. Before this, a Chinese
    review got English theme titles like "Deep Learning Factor Models in
    Asset Pricing" while the body stayed Chinese."""
    v = (settings.review_language or "en").strip().lower()
    if v in ("zh", "cn", "chinese", "中文", "ch"):
        return (
            "输出语言要求：theme name 字段用中文书写（它将被用作综述的小节标题）；"
            "description 可用中英双语。"
        )
    return "Write theme names in English."
