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
import time

from citens.config import settings
from citens.llm import chat, run_concurrent
from citens.models import ExtractedPaper, Paper, SynthesisResult, ThemeStructure

INTRO_PROMPT = """You are an academic survey writer. Write the "Introduction" (for Chinese \
output: 900-1400 characters; English: 600-900 words) for \
the following research topic.
1. Establish background and significance.
2. Surface the open problems.
3. State the review's purpose and structure.
4. Refer to the survey as "this review", never "this paper"/"本研究".
5. If a SEARCH METHODOLOGY block with real funnel numbers is provided, report it in \
the introduction's closing paragraph (检索源、候选数、纳入数): these are measured facts \
of this review's own retrieval — state them as the review's methodology, PRISMA-style.
6. Fluent, scholarly prose."""

SECTION_PROMPT = """You are an academic survey writer. Using the theme info and paper list below, write this theme section (for Chinese output: 1500-2200 characters; English: 1000-1400 words).

STRUCTURE — unfold the literature, do not compress it:
- Opening paragraph (定位段): where this theme sits in the field and the organizing axis you will follow (mechanism / method family / evolution).
- 2-4 argument paragraphs: each advances ONE point and DEVELOPS its representative works (1-3 papers per paragraph) in depth — for each: motivation -> method mechanism (specific enough to retell: model class, data structure, key design) -> evidence (use the abstract's stated numbers, datasets, sample sizes whenever present) -> significance (what it established, what it leaves open).
- Closing paragraph (收束段): one-sentence synthesis of the theme's evolution logic and the hinge to the next theme.

RULES:
1. A review is NOT a survey list. Never write "Author A reported X. Author B reported Y." sequences — organize by mechanism / method / finding and cite papers INSIDE the synthesis.
2. Analyze inter-paper relations (agreement, contradiction, evolution, complementarity). Where the CROSS-PAPER SYNTHESIS notes consensus or contradictions, foreground them and argue a position.
3. NAMED SYSTEMS: use the 系统名 field — first mention as "作者等人[n]提出的 X"; afterwards refer to it as X. Named systems make a review read as a map of the field; bare indices read as a list.
4. **Cite BROADLY within the theme**: draw on every listed paper that bears on the argument — a section that cites only two or three of its listed papers is incomplete.
5. **Cite a paper by writing its index in square brackets, e.g. [0] or [3].** Use the EXACT index shown before each paper below. A sentence making a claim about a paper MUST carry its [index].
6. **Ground every cited claim in what the paper's abstract actually says.** Do NOT invent specifics (methods, numbers, mechanisms) the abstract does not state. When the abstract carries numbers, CARRY THEM INTO THE PROSE — a quantitative result must not be flattened into "显著提升" (name the metric and the magnitude).
7. **NO-ABSTRACT papers**: a paper marked "无摘要 / NO ABSTRACT" has no ground text. Cite it at most for title-level bibliographic context. NEVER describe its methods, findings, or design.
8. **Keep YOUR argument outside the citation brackets.** [n] marks only what the cited source itself supports; interpretive sentences of your own stand WITHOUT a citation. AT MOST 3 citation markers per sentence (a 4th only when each demonstrably backs a distinct part); NEVER 5+.
9. DENSITY: every paragraph must carry concrete information the previous paragraphs did not (机制、数字、对比、数据集). No filler, no pure-transition paragraphs, no restating the theme description.
10. Discursive academic narrative is ENCOURAGED where it earns its place — "从X到Y的演进"、"这一发现的意义在于"、"与之相对"、"值得注意的是" (colloquial/essayistic flourishes remain banned).
11. Do NOT write any heading line — start directly with prose.
12. Fluent, scholarly prose."""

CRIT_SYNTH_PROMPT = """You are an academic survey writer. Write a "Critical Synthesis" section \
(for Chinese output: 900-1300 characters; English: 600-900 words) that takes a position across \
the whole literature.
Use the provided cross-paper consensus and contradictions. Argue, do not merely list — \
a review may take a view, but it must SHOW its reasoning, not assert it. Where the evidence \
conflicts, map the disagreement (who claims what, on which data/method) instead of averaging it away.
Use named systems (系统名) where available — "作者等人[n]提出的 X" on first mention, X after.
Cite papers by [index], and keep claims grounded in what those papers' abstracts support. Your own \
argument and framing stand WITHOUT a citation — [n] marks only what the cited source itself says, \
and every [n] must back the specific statement it is attached to.
Do NOT write a heading line. Fluent, scholarly prose."""

CONCLUSION_PROMPT = """You are an academic survey writer. Write "Conclusion & Outlook" (for \
Chinese output: 900-1300 characters; English: 600-900 words) for the topic.
1. Close with a USABLE MAP of the field, not a replay of the sections: what is settled, \
where the live disagreements stand, and which open questions are most worth attacking next \
(anchor them in the listed research gaps where possible).
2. Note current shortcomings.
3. Cite ONLY the papers listed below, using their EXACT [index]. Never invent papers, \
authors, or numbering not in the list — a conclusion citing unknown work is worthless. Cite [n] \
only for claims that paper's abstract supports; your own outlook and framing need no citation.
4. Do NOT write any heading, bold title, or references list.
5. Fluent, scholarly prose."""

ABSTRACT_PROMPT = """You are an academic survey writer. Write the structured ABSTRACT and \
KEYWORDS for the FINISHED review text provided below (do not rewrite the review).

- Abstract (Chinese output: 280-380 characters; English: 180-240 words), structured as: \
background (1 sentence) -> scope and method (dimensions organized, number of papers if the \
review states it) -> main findings along the review's OWN argument (2-3 sentences) -> the \
review's position and outlook (1 sentence).
- Keywords: 5-6 terms, separated by "；", most specific first.
- State ONLY what the review below actually says — no new claims, NO [n] citation markers, \
no numbers the review does not contain.

Output format (exactly this shape, no headings of your own):
摘要：<abstract text>
关键词：<k1>；<k2>；<k3>；<k4>；<k5>"""

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

# Formal-register rules distilled from the nature-writing / nature-polishing
# skills (sentence discipline, no essayistic commentary, verb-evidence
# calibration). The 2026-08-18 review averaged 98 chars/sentence with 31
# sentences over 150 chars — comma-chained multi-proposition sentences were
# the dominant informality, not colloquial vocabulary.
_ZH_FORMALITY_LINE = """学术语体规范（严格执行）：
- 一句只承载一个主要命题。禁止用逗号/分号串联多个命题的超长句（中文句子超过约60字即应拆分，或改用"因此/与之相反/这一局限在于"等显式逻辑连接）。
- 禁用评述性、随笔式表达与修辞设问："耐人寻味""有趣的是""值得玩味""我们不妨""可以说""不难发现"等一律不得出现；综述者的判断必须以论证呈现，不得以修辞断言。
- 动词与证据强度校准："证明/表明"仅用于强证据，"提示/与……一致"用于间接证据；禁止无比较对象的"显著提升/大幅改善"。
- 每段首句为该段主题句；段内各句与前一句保持显式逻辑关系（因果、对比、限定、例证）。
- 避免泛化断言（"许多研究表明""大量工作"）——要么落到具体引用，要么删除。"""

_EN_FORMALITY_LINE = """Formal-register rules (strict): one main proposition per sentence — split
comma-chained multi-proposition sentences (target ≤30 words); no essayistic
commentary or rhetorical questions; calibrate verbs to evidence strength
("show/demonstrate" only for strong evidence, "suggest/consistent with" for
indirect); every paragraph opens with its topic sentence and each sentence
bears an explicit logical relation to the previous one; replace vague
quantifiers ("many studies show") with specific citations or delete them."""

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


def formality_instruction() -> str:
    """Formal-register rules appended to every writer system prompt."""
    return "\n" + (_ZH_FORMALITY_LINE if _lang() == "zh" else _EN_FORMALITY_LINE)


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
    empty body (thinking ate the budget) or truncate mid-sentence.

    Ladder: normal budget WITHOUT thinking → double budget without thinking →
    double budget WITH thinking as the last resort. No-thinking goes first:
    hybrid models share the completion budget between deliberation and body,
    and with large evidence prompts the thinking prefix is exactly what
    starved the body (the 08-19 run: 9 sections came back empty on the
    thinking attempt and were rescued by the no-thinking retry — so skip the
    wasted round and its ~60-90s per section). A short pause between attempts
    also gives a throttling provider room.

    Writer sections run on the STRONG model tier (see citens.llm)."""
    text = ""
    for attempt in range(3):
        budget = max_tokens * (2 if attempt else 1)
        try:
            text = chat(
                system, user, max_tokens=budget, strong=True,
                thinking=attempt == 2,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    [writer:{label}] attempt {attempt + 1} failed: {e}")
            text = ""
        if len(text) >= 200 and _complete(text):
            return text
        if attempt < 2:
            time.sleep(2.0)
        print(
            f"    [writer:{label}] attempt {attempt + 1} "
            f"({'empty' if not text else 'truncated'}; "
            f"{'retrying with more tokens' if attempt == 0 else 'retrying WITH thinking'})"
        )
    return text


def _papers_block(indexed: list[tuple[int, ExtractedPaper]]) -> str:
    parts = []
    for idx, p in indexed:
        parts.append(
            f"\n[{idx}] {p.title}\n"
            + (f"  系统名: {p.system_name}\n" if (p.system_name or "").strip() else "")
            + f"  研究问题: {p.research_question}\n"
            + f"  方法: {p.methodology}\n"
            + f"  发现: {'; '.join(p.key_findings)}\n"
            + f"  局限: {'; '.join(p.limitations)}\n"
        )
        if not (p.abstract or "").strip():
            parts.append(
                "  ⚠ 无摘要 / NO ABSTRACT — cite for title-level context ONLY; "
                "never describe its methods, findings, or design\n"
            )
    return "".join(parts)


def _supporting_block(supporting: list[tuple[int, Paper]] | None) -> str:
    """Render the abstract-only supporting layer for the writer's prompts.

    These papers are real bibliography entries (verifiable against their
    abstracts) but carry no deep extraction — the writer may cite them for
    background, context, and comparisons, never for primary method/result
    claims. This is what lets a review cite far more than it dissects.
    """
    if not supporting:
        return ""
    lines = [
        "\n支持文献 / Supporting references (abstract-only; cite with their [index] "
        "for BACKGROUND, CONTEXT or COMPARISON claims — NOT for primary claims "
        "about methods, findings or magnitudes):\n"
    ]
    for idx, p in supporting[:18]:
        abstract = " ".join(p.abstract.split())[:180]
        venue = p.venue or ""
        year = p.year or ""
        lines.append(
            f"[{idx}] {p.title}"
            + (f" — {venue} ({year})" if venue or year else "")
            + (
                f": {abstract}…"
                if abstract
                else " — 无摘要 / NO ABSTRACT: bibliography presence only; "
                "do not cite it for any specific claim\n"
            )
        )
    return "\n".join(lines) + "\n"


def write_review_body(
    papers: list[ExtractedPaper],
    themes: ThemeStructure,
    topic: str,
    *,
    synthesis: SynthesisResult | None = None,
    on_step=None,
    evidence_for=None,
    terminology: dict[str, str] | None = None,
    supporting: list[tuple[int, Paper]] | None = None,
    coverage_note: str = "",
    search_summary: str = "",
) -> str:
    """Generate the review BODY (title + intro + theme sections + critical
    synthesis + conclusion).

    Sections are independent prompts — they are generated CONCURRENTLY on the
    thread pool and assembled in reading order. The references section is
    intentionally NOT included here; the pipeline appends it from the
    CitationTable.

    ``evidence_for(theme) -> str``, when given, supplies full-text evidence
    excerpts for a theme's papers (from the ChunkStore): the writer grounds
    its claims in them instead of abstract extracts alone — fewer
    unsupported claims at the SOURCE, before any verifier pass.

    ``terminology`` (from the domain profile) is the EN->ZH ledger appended
    to every prompt so field terms get ONE consistent Chinese rendering
    across all sections (nature-reader's Terminology Ledger).
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
    term_line = ""
    if terminology:
        pairs = "; ".join(f"{en}={zh}" for en, zh in list(terminology.items())[:40])
        term_line = (
            "\nTERMINOLOGY LEDGER (use these exact renderings, consistently, "
            f"on first and later occurrence): {pairs}\n"
        )
    support_block = _supporting_block(supporting)
    coverage_block = (
        "\n覆盖性声明素材 / COVERAGE HONESTY (deterministic, from retrieval):\n"
        f"{coverage_note}\n"
        "RULE: in the conclusion, state these coverage weaknesses in one honest "
        "paragraph（如\"公开检索对X方向的覆盖有限（本综述仅纳入N篇），相关结论应视为阶段性\"），"
        "naming thin directions explicitly. Never silently narrow the scope; never "
        "claim coverage you do not have.\n"
        if coverage_note
        else ""
    )
    jobs: list[dict] = [
        {"kind": "intro", "label": "intro",
         "system": INTRO_PROMPT + "\n" + lang_line + formality_instruction() + term_line,
         "user": f"研究主题 / Topic: {topic}"
                 + (f"\n\nSEARCH METHODOLOGY (measured facts of this run):\n{search_summary}"
                    if search_summary else ""),
         "budget": 8192}
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
            f"{support_block}"
        )
        system_prompt = SECTION_PROMPT + "\n" + lang_line + formality_instruction() + term_line
        if evidence_for is not None:
            excerpts = evidence_for(theme)
            if excerpts:
                section_prompt += (
                    "\n全文证据摘录 / Full-text evidence excerpts (verbatim from the "
                    "papers' full texts where available):\n"
                    f"{excerpts}\n"
                )
                system_prompt += (
                    "\nEVIDENCE RULE: the excerpts above are verbatim source text. "
                    "Ground factual claims in them whenever they cover the point — "
                    "specifics (numbers, methods, findings) must come from the "
                    "excerpts or the paper extracts, never invented. Cite the "
                    "paper's [index] for every such claim.\n"
                )
        theme_jobs.append(
            {"kind": "theme", "label": f"theme-{ti+1}", "name": theme.name,
             "system": system_prompt, "user": section_prompt, "budget": 10240}
        )
    jobs.extend(theme_jobs)

    if synthesis and (synthesis.consensus or synthesis.contradictions):
        synth_prompt = (
            f"研究主题 / Topic: {topic}\n"
            f"共识 / Consensus:\n" + "".join(f"- {c}\n" for c in synthesis.consensus)
            + "矛盾 / Contradictions:\n" + "".join(f"- {c}\n" for c in synthesis.contradictions)
            + support_block
        )
        jobs.append(
            {"kind": "crit", "label": "critical-synthesis",
             "system": CRIT_SYNTH_PROMPT + "\n" + lang_line + formality_instruction() + term_line,
             "user": synth_prompt + coverage_block, "budget": 8192}
        )

    summary = "".join(f"- {t.name}: {t.description}\n" for t in themes.themes)
    gaps = ""
    if synthesis and synthesis.gaps:
        gaps = "研究空白 / Research gaps:\n" + "".join(f"- {g}\n" for g in synthesis.gaps)
    jobs.append(
        {"kind": "conclusion", "label": "conclusion",
         "system": CONCLUSION_PROMPT + "\n" + lang_line + formality_instruction() + term_line,
         "user": (
             f"研究主题 / Topic: {topic}\n\n主题结构 / Themes:\n{summary}\n{gaps}\n"
             f"可引用论文 / Citable papers (use EXACT [index]):"
             f"{_papers_block(list(enumerate(papers)))}"
             f"{support_block}"
             f"{coverage_block}"
         ),
         "budget": 8192}
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

    # --- second wave: the abstract abstracts the WRITTEN review -------------
    # generated after the body so it states what the review actually says;
    # carries no [n] markers, so it never enters the verification pipeline
    body_so_far = "\n".join(sections[1:])
    if body_so_far:
        if on_step:
            on_step("abstract", "")
        abstract_md = _strip_leading_headings(
            _chat_section(
                ABSTRACT_PROMPT + "\n" + lang_line,
                f"研究主题 / Topic: {topic}\n\n综述全文 / Finished review text:\n{body_so_far}",
                4096,
                "abstract",
            )
        )
        if abstract_md:
            sections.insert(1, f"{abstract_md}\n")

    return "\n".join(sections)


# Backwards-compatible alias.
def write_review(papers, themes, topic, *, synthesis=None, on_step=None) -> str:
    return write_review_body(papers, themes, topic, synthesis=synthesis, on_step=on_step)
