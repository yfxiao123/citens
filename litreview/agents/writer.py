"""Review-writing agent: assemble a cited, multi-section survey in Markdown.

Citations use ``[n]`` markers where ``n`` is the paper's GLOBAL index in the
final list — the same index the CitationTable renders in references.bib. The
references section itself is appended by the pipeline (from the CitationTable)
so prose markers and the bibliography can never drift.

Phase 3 replaces the section body with synthesis-driven, claim-grounded output
produced jointly with the Synth / Verifier agents.
"""

from __future__ import annotations

import re

from litreview.llm import chat
from litreview.models import ExtractedPaper, SynthesisResult, ThemeStructure

INTRO_PROMPT = """You are an academic survey writer. Write the "Introduction" (500-800 words) for \
the following research topic.
1. Establish background and significance.
2. Surface the open problems.
3. State the review's purpose and structure.
4. Use "本综述"/"this review", never "本研究".
5. Fluent, scholarly prose."""

SECTION_PROMPT = """You are an academic survey writer. Using the theme info and paper list below, \
write this theme section (600-900 words).
1. Synthesize across the papers — do NOT list them one by one.
2. Analyze inter-paper relations (agreement, contradiction, evolution, complementarity). Where the \
CROSS-PAPER SYNTHESIS notes consensus or contradictions, foreground them and argue a position.
3. **Cite a paper by writing its index in square brackets, e.g. [0] or [3].** Use the EXACT index \
shown before each paper below. A sentence making a claim about a paper MUST carry its [index].
4. **Ground every cited claim in what the paper's abstract actually says.** Do NOT invent specifics \
(methods, numbers, mechanisms) the abstract does not state. When the abstract is thin, make the \
claim appropriately general rather than fabricating detail. Prefer fewer, defensible claims over \
many speculative ones.
5. Do NOT write any heading line — start directly with prose.
6. Fluent, scholarly prose."""

CRIT_SYNTH_PROMPT = """You are an academic survey writer. Write a "Critical Synthesis" section \
(400-700 words) that takes a position across the whole literature.
Use the provided cross-paper consensus and contradictions. Argue, do not merely list.
Cite papers by [index], and keep claims grounded in what those papers' abstracts support.
Do NOT write a heading line. Fluent, scholarly prose."""

CONCLUSION_PROMPT = """You are an academic survey writer. Write "Conclusion & Outlook" (400-600 \
words) for the topic.
1. Summarize main findings and consensus.
2. Note current shortcomings.
3. Outlook on future directions.
4. Cite with [index] where appropriate.
5. Fluent, scholarly prose."""

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def _strip_leading_headings(text: str) -> str:
    """Drop heading lines the model re-emits at the top of a section body."""
    lines = text.splitlines()
    changed = True
    while changed and lines:
        changed = False
        # remove a leading blank line if a heading follows
        if not lines[0].strip() and len(lines) > 1 and _HEADING_RE.match(lines[1]):
            lines.pop(0)
            changed = True
        # remove a leading heading line
        if lines and _HEADING_RE.match(lines[0]):
            lines.pop(0)
            changed = True
    return "\n".join(lines).strip()


def _papers_block(indexed: list[tuple[int, ExtractedPaper]]) -> str:
    parts = []
    for idx, p in indexed:
        parts.append(
            f"\n[{idx}] {p.title}\n"
            f"  研究问题: {p.research_question}\n"
            f"  方法: {p.methodology}\n"
            f"  发现: {'; '.join(p.key_findings)}\n"
            f"  局限: {'; '.join(p.limitations)}\n"
        )
    return "".join(parts)


def write_review_body(
    papers: list[ExtractedPaper],
    themes: ThemeStructure,
    topic: str,
    *,
    synthesis: SynthesisResult | None = None,
    on_step=None,
) -> str:
    """Generate the review BODY (title + intro + theme sections + critical
    synthesis + conclusion).

    The references section is intentionally NOT included here; the pipeline
    appends it from the CitationTable.
    """
    sections: list[str] = [f"# {topic}\n"]

    synth_context = ""
    if synthesis and (synthesis.consensus or synthesis.contradictions):
        cons = "; ".join(synthesis.consensus)
        contra = "; ".join(synthesis.contradictions)
        synth_context = (
            f"\n跨论文综合 / Cross-paper synthesis:\n"
            f"  共识 / Consensus: {cons}\n"
            f"  矛盾 / Contradictions: {contra}\n"
        )

    if on_step:
        on_step("intro")
    intro = chat(INTRO_PROMPT, f"研究主题 / Topic: {topic}", max_tokens=4096)
    sections.append(f"## 引言 / Introduction\n\n{_strip_leading_headings(intro)}\n")

    for ti, theme in enumerate(themes.themes):
        if on_step:
            on_step(f"theme-{ti+1}", theme.name)
        idx_set = {i for i in theme.paper_indices if 0 <= i < len(papers)}
        indexed = [(i, papers[i]) for i in sorted(idx_set)]
        section_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"主题名称 / Theme: {theme.name}\n"
            f"主题描述 / Description: {theme.description}\n"
            f"逻辑关系 / Relations: {theme.logical_relations}\n"
            f"{synth_context}"
            f"包含论文 / Papers (index in [brackets]):{_papers_block(indexed)}\n"
        )
        body = chat(SECTION_PROMPT, section_prompt, max_tokens=6144)
        sections.append(f"### {theme.name}\n\n{_strip_leading_headings(body)}\n")

    if synthesis and (synthesis.consensus or synthesis.contradictions):
        if on_step:
            on_step("critical-synthesis")
        synth_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"共识 / Consensus:\n" + "".join(f"- {c}\n" for c in synthesis.consensus)
            + f"矛盾 / Contradictions:\n" + "".join(f"- {c}\n" for c in synthesis.contradictions)
        )
        crit = chat(CRIT_SYNTH_PROMPT, synth_prompt, max_tokens=4096)
        sections.append(
            f"## 批判性综合 / Critical Synthesis\n\n{_strip_leading_headings(crit)}\n"
        )

    if on_step:
        on_step("conclusion")
    summary = "".join(f"- {t.name}: {t.description}\n" for t in themes.themes)
    conclusion = chat(
        CONCLUSION_PROMPT,
        f"研究主题 / Topic: {topic}\n\n主题结构 / Themes:\n{summary}",
        max_tokens=4096,
    )
    sections.append(f"## 总结与展望 / Conclusion\n\n{_strip_leading_headings(conclusion)}\n")

    return "\n".join(sections)


# Backwards-compatible alias.
def write_review(papers, themes, topic, *, synthesis=None, on_step=None) -> str:
    return write_review_body(papers, themes, topic, synthesis=synthesis, on_step=on_step)
