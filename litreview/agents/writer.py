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

from litreview.config import settings
from litreview.llm import chat, run_concurrent
from litreview.models import ExtractedPaper, SynthesisResult, ThemeStructure

INTRO_PROMPT = """You are an academic survey writer. Write the "Introduction" (500-800 words) for \
the following research topic.
1. Establish background and significance.
2. Surface the open problems.
3. State the review's purpose and structure.
4. Refer to the survey as "this review", never "this paper"/"本研究".
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
3. Outlook on future directions — anchor it in the listed research gaps where possible.
4. Cite ONLY the papers listed below, using their EXACT [index]. Never invent papers, \
authors, or numbering not in the list — a conclusion citing unknown work is worthless.
5. Do NOT write any heading, bold title, or references list.
6. Fluent, scholarly prose."""

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_BOLD_TITLE_RE = re.compile(r"^\s{0,3}\*\*[^*\n]+\*\*\s*$")
# DOTALL: a model-appended references block spans multiple lines — without it
# only the first entry was stripped and the hallucinated rest survived.
_TAIL_REFS_RE = re.compile(
    r"\n\s*(\*{0,2}references\*{0,2}|参考文献)\s*\n.*$", re.IGNORECASE | re.DOTALL
)
_TERMINALS = "。．.!?！？；;”\"')]}】》…"

# --- output language (settings.review_language: "en" | "zh") -----------------
# The prose language is injected into every writer prompt; the section
# headings are localized to match, so a run is uniformly one language instead
# of the accidental Chinese-intro/English-body mix.
_ZH_ALIASES = {"zh", "cn", "chinese", "中文", "ch"}
_ZH_LANG_LINE = "输出语言：全文使用中文撰写，学术书面语；术语首次出现时在括号内附英文原文。"
_EN_LANG_LINE = "Output language: write all prose in English."

_HEADINGS = {
    "zh": {
        "intro": "引言",
        "crit": "批判性综合",
        "conclusion": "总结与展望",
        "refs": "参考文献",
    },
    "en": {
        "intro": "Introduction",
        "crit": "Critical Synthesis",
        "conclusion": "Conclusion and Outlook",
        "refs": "References",
    },
}


def _lang() -> str:
    v = (settings.review_language or "en").strip().lower()
    return "zh" if v in _ZH_ALIASES else "en"


def lang_instruction() -> str:
    """One line appended to every writer system prompt."""
    return _ZH_LANG_LINE if _lang() == "zh" else _EN_LANG_LINE


def localized_heading(key: str) -> str:
    """Localized section heading ("intro" | "crit" | "conclusion" | "refs")."""
    return _HEADINGS[_lang()][key]


def _strip_leading_headings(text: str) -> str:
    """Drop heading/bold-title lines the model re-emits at the top of a section."""
    lines = text.splitlines()
    changed = True
    while changed and lines:
        changed = False
        # remove a leading blank line if a heading follows
        if not lines[0].strip() and len(lines) > 1 and _HEADING_RE.match(lines[1]):
            lines.pop(0)
            changed = True
        # remove a leading heading line (## ...) or bold-only title (**...**)
        if lines and (_HEADING_RE.match(lines[0]) or _BOLD_TITLE_RE.match(lines[0])):
            lines.pop(0)
            changed = True
    return "\n".join(lines).strip()


def _strip_tail_references(text: str) -> str:
    """Drop a trailing References block the model sometimes appends (the real
    one is rendered by the CitationTable — a model-made one hallucinates)."""
    return _TAIL_REFS_RE.sub("", text).rstrip()


def _complete(text: str) -> bool:
    """True if the text ends in terminal punctuation (not mid-sentence)."""
    t = text.rstrip()
    return bool(t) and t[-1] in _TERMINALS


def _chat_section(system: str, user: str, max_tokens: int, label: str = "") -> str:
    """chat() with writer-grade retries: reasoning models sometimes return an
    empty body (thinking ate the budget) or truncate mid-sentence. Retry once
    with double the budget; still return whatever we have after that.

    Writer sections run on the STRONG model tier (see litreview.llm)."""
    text = ""
    for attempt in range(2):
        budget = max_tokens * (attempt + 1)
        try:
            text = chat(system, user, max_tokens=budget, strong=True)
        except Exception as e:  # noqa: BLE001
            print(f"    [writer:{label}] attempt {attempt + 1} failed: {e}")
            text = ""
        if len(text) >= 200 and _complete(text):
            return text
        print(
            f"    [writer:{label}] attempt {attempt + 1} "
            f"({'empty' if not text else 'truncated'}; retrying with more tokens)"
        )
    return text


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

    Sections are independent prompts — they are generated CONCURRENTLY on the
    thread pool and assembled in reading order. The references section is
    intentionally NOT included here; the pipeline appends it from the
    CitationTable.
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

    # --- build all section jobs (kind, label, system, user, budget) ----------
    lang_line = lang_instruction()
    jobs: list[dict] = [
        {"kind": "intro", "label": "intro", "system": INTRO_PROMPT + "\n" + lang_line,
         "user": f"研究主题 / Topic: {topic}", "budget": 4096}
    ]

    theme_jobs: list[dict] = []
    for ti, theme in enumerate(themes.themes):
        idx_set = {i for i in theme.paper_indices if 0 <= i < len(papers)}
        indexed = [(i, papers[i]) for i in sorted(idx_set)]
        if not indexed:
            continue
        section_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"主题名称 / Theme: {theme.name}\n"
            f"主题描述 / Description: {theme.description}\n"
            f"逻辑关系 / Relations: {theme.logical_relations}\n"
            f"{synth_context}"
            f"包含论文 / Papers (index in [brackets]):{_papers_block(indexed)}\n"
        )
        theme_jobs.append(
            {"kind": "theme", "label": f"theme-{ti+1}", "name": theme.name,
             "system": SECTION_PROMPT + "\n" + lang_line, "user": section_prompt, "budget": 6144}
        )
    jobs.extend(theme_jobs)

    if synthesis and (synthesis.consensus or synthesis.contradictions):
        synth_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"共识 / Consensus:\n" + "".join(f"- {c}\n" for c in synthesis.consensus)
            + "矛盾 / Contradictions:\n" + "".join(f"- {c}\n" for c in synthesis.contradictions)
        )
        jobs.append(
            {"kind": "crit", "label": "critical-synthesis",
             "system": CRIT_SYNTH_PROMPT + "\n" + lang_line, "user": synth_prompt, "budget": 4096}
        )

    summary = "".join(f"- {t.name}: {t.description}\n" for t in themes.themes)
    gaps = ""
    if synthesis and synthesis.gaps:
        gaps = "研究空白 / Research gaps:\n" + "".join(f"- {g}\n" for g in synthesis.gaps)
    jobs.append(
        {"kind": "conclusion", "label": "conclusion", "system": CONCLUSION_PROMPT + "\n" + lang_line,
         "user": (
             f"研究主题 / Topic: {topic}\n\n主题结构 / Themes:\n{summary}\n{gaps}\n"
             f"可引用论文 / Citable papers (use EXACT [index]):"
             f"{_papers_block(list(enumerate(papers)))}"
         ),
         "budget": 4096}
    )

    # --- generate concurrently, assemble in reading order --------------------
    def _run(_i, job):
        if on_step:
            on_step(job["label"], job.get("name", ""))
        return _chat_section(job["system"], job["user"], job["budget"], job["label"])

    bodies = run_concurrent(_run, jobs)

    intro = _strip_leading_headings(bodies[0])
    sections.append(f"## {localized_heading('intro')}\n\n{intro}\n")

    for job, body in zip(theme_jobs, bodies[1 : 1 + len(theme_jobs)], strict=False):
        body = _strip_tail_references(_strip_leading_headings(body))
        if body:
            sections.append(f"### {job['name']}\n\n{body}\n")
        else:
            print(f"    [writer] theme section empty after retries, skipped: {job['name']}")

    offset = 1 + len(theme_jobs)
    if offset < len(bodies):  # critical synthesis present
        crit = _strip_tail_references(_strip_leading_headings(bodies[offset]))
        sections.append(f"## {localized_heading('crit')}\n\n{crit}\n")
        offset += 1
    if offset < len(bodies):
        conclusion = _strip_tail_references(_strip_leading_headings(bodies[offset]))
        sections.append(f"## {localized_heading('conclusion')}\n\n{conclusion}\n")

    return "\n".join(sections)


# Backwards-compatible alias.
def write_review(papers, themes, topic, *, synthesis=None, on_step=None) -> str:
    return write_review_body(papers, themes, topic, synthesis=synthesis, on_step=on_step)
