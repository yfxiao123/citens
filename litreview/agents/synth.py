"""Synthesis agent: critical cross-paper analysis (consensus / contradictions / gaps).

This is what turns a "summary" review into a *critical* one. Instead of listing
per-paper findings, the Synth agent looks across all papers and surfaces where
they agree, where they conflict, and what the field has *not* covered. The
gaps feed the Reflector (supplementary retrieval), and the consensus /
contradictions feed the writer so sections argue rather than enumerate.
"""

from __future__ import annotations

from litreview.llm import chat_json
from litreview.models import ExtractedPaper, SynthesisResult, ThemeStructure

SYSTEM_PROMPT = """You are a critical-synthesis expert for academic literature. Given a set of \
papers (structured) and the themes they fall into, look ACROSS papers and identify:

1. "consensus": claims/observations on which multiple papers agree.
2. "contradictions": points where papers diverge, conflict, or reach opposite conclusions.
3. "gaps": aspects of the topic that are under-covered, unresolved, or missing entirely — \
be specific and concrete.

Be analytical and specific, cite papers by [index]. Do NOT just restate each paper.

Output JSON:
{
  "consensus": ["...", "..."],
  "contradictions": ["...", "..."],
  "gaps": ["...", "..."]
}"""


def synthesize(
    papers: list[ExtractedPaper],
    themes: ThemeStructure,
    topic: str,
) -> SynthesisResult:
    """Produce a cross-paper critical synthesis."""
    parts = []
    for i, p in enumerate(papers):
        parts.append(
            f"\n[{i}] {p.title}\n"
            f"  研究问题: {p.research_question}\n"
            f"  方法: {p.methodology}\n"
            f"  发现: {'; '.join(p.key_findings)}\n"
            f"  局限: {'; '.join(p.limitations)}\n"
        )
    theme_summary = "".join(f"- {t.name}: {t.description}\n" for t in themes.themes)
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"主题结构 / Themes:\n{theme_summary}\n"
        f"论文 / Papers:{''.join(parts)}\n\n"
        "Identify consensus, contradictions, and gaps across these papers."
    )
    result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=3072)
    return SynthesisResult(
        consensus=result.get("consensus", []) or [],
        contradictions=result.get("contradictions", []) or [],
        gaps=result.get("gaps", []) or [],
    )
