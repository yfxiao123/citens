"""Information-extraction agent: deep structured fields from each paper.

Per-paper calls are independent — they run on a thread pool
(:func:`litreview.llm.run_concurrent`) so a 9-paper extract stage takes one
round-trip instead of nine.

Deep extraction (v2) adds quality assessment: study type, evidence level,
method rigor, effect direction — feeding both the ranking (retrieval quality)
and the writer (extraction quality → better-grounded claims).
"""

from __future__ import annotations

from litreview.agents.quality import assess_paper_quality
from litreview.llm import chat_json, run_concurrent
from litreview.models import ExtractedPaper, ScoredPaper

SYSTEM_PROMPT = """You are an academic-paper analyst. Given a research topic and a paper, extract \
DEEP structured information from its abstract. Go beyond surface-level summary — extract \
specific, verifiable details.

Output JSON (include every field):
{
  "research_question": "what specific problem the paper tackles (not generic — \
include the specific aspect, gap, or twist)",
  "methodology": "exact technical approach: model class, algorithm, data structure, \
theoretical framework (be specific: 'CNN with LSTM on 40-level LOB data' not 'deep learning')",
  "key_findings": [
    "finding 1 with specifics: direction of effect, magnitude, statistical significance if mentioned",
    "finding 2",
    "finding 3"
  ],
  "limitations": [
    "limitation 1 (what the authors acknowledge OR what is evident from the abstract)",
    "limitation 2"
  ],
  "relevance_to_topic": "this paper's SPECIFIC value to the review — what unique \
perspective, method, or evidence does it contribute that others don't",
  "study_type": "one of: meta_analysis, systematic_review, rct, cohort, case_control, \
cross_sectional, survey, theoretical, simulation, empirical, review, other",
  "sample_or_data": "sample size, dataset name, or data description from abstract",
  "effect_direction": "one of: positive, negative, mixed, null, not_applicable"
}

CRITICAL RULES:
1. Findings must be SPECIFIC — include effect direction, magnitude, or statistical \
significance when the abstract mentions them. "Method X outperforms Y by 15% on \
dataset Z" is good; "Method X is effective" is bad.
2. Methodology must name the specific technique, not just the broad category.
3. Limitations include both author-acknowledged and evident-from-abstract issues.
4. Only extract what the abstract actually says — do NOT infer or fabricate."""


def _extract_one(paper: ScoredPaper, topic: str, assess_quality: bool = True) -> ExtractedPaper:
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"论文信息 / Paper:\n{paper.brief()}\n\n"
        "Extract the deep structured information."
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=3072)
    except Exception as e:  # noqa: BLE001
        print(f"    extract failed ({paper.title[:40]}): {e}")
        result = {}

    quality = {}
    if assess_quality:
        # Quick quality assessment from the same abstract
        ep_preview = ExtractedPaper(
            **paper.model_dump(exclude={"id"}),
            research_question=result.get("research_question", ""),
            methodology=result.get("methodology", ""),
            key_findings=result.get("key_findings", []),
        )
        quality = assess_paper_quality(ep_preview)

    return ExtractedPaper(
        **paper.model_dump(exclude={"id"}),
        research_question=result.get("research_question", ""),
        methodology=result.get("methodology", ""),
        key_findings=result.get("key_findings", []),
        limitations=result.get("limitations", []),
        relevance_to_topic=result.get("relevance_to_topic", ""),
        quality=quality,
    )


def extract_papers(
    papers: list[ScoredPaper],
    topic: str,
    *,
    on_progress=None,
    assess_quality: bool = True,
) -> list[ExtractedPaper]:
    """Extract deep structured fields for each filtered paper (concurrently).

    Args:
        papers: Filtered papers to extract from
        topic: Research topic
        on_progress: Progress callback
        assess_quality: If True, also assess evidence level and method rigor
    """
    total = len(papers)

    def on_done(i, paper, _result):
        if on_progress:
            on_progress(i + 1, total, paper.title[:50])

    return run_concurrent(
        lambda i, p: _extract_one(p, topic, assess_quality), list(papers), on_done=on_done
    )
