"""Quality assessment and evidence grading for extracted papers.

Implements methodology-inspired quality signals:
- Study type classification (RCT / cohort / survey / theoretical / simulation)
- Evidence level grading (Level 1-4, adapted from standard hierarchies)
- Method rigor scoring from abstract signals
- Comparison matrix generation for structured synthesis

These feed into the ranking (retrieval quality) and the writer (extraction
quality → better-grounded claims).
"""

from __future__ import annotations

from litreview.llm import chat_json
from litreview.models import ExtractedPaper

QUALITY_PROMPT = """You are an evidence-quality assessor for academic papers. Given a paper's \
abstract, assess its methodological quality and evidence level.

Output JSON (include every field):
{
  "study_type": "one of: meta_analysis, systematic_review, rct, cohort, case_control, \
cross_sectional, survey, theoretical, simulation, empirical, review, other",
  "evidence_level": 1-4 where:
    1 = meta-analysis or systematic review of primary studies
    2 = randomized controlled trial or high-quality experimental study
    3 = observational study (cohort, case-control, cross-sectional, survey)
    4 = theoretical/model/opinion/case-report
  "method_rigor": 1-5 where:
    5 = rigorous: clear methodology, adequate sample/data, statistical tests, controls
    4 = good: mostly rigorous with minor gaps
    3 = adequate: methodology described but some concerns
    2 = weak: significant methodological concerns
    1 = poor: methodology unclear or flawed
  "sample_or_data": "sample size, dataset name, or data description (from abstract)",
  "effect_direction": "one of: positive, negative, mixed, null, not_applicable",
  "temporal_scope": "time period covered by the study (from abstract, if mentioned)",
  "quality_note": "one-line assessment of methodological strengths/weaknesses"
}"""


def assess_paper_quality(paper: ExtractedPaper) -> dict:
    """Assess methodological quality and evidence level from the paper's abstract.

    Returns a dict with study_type, evidence_level, method_rigor, etc.
    """
    user_prompt = (
        f"论文标题: {paper.title}\n"
        f"摘要: {paper.abstract[:1500]}\n"
        f"方法: {paper.methodology}\n"
        f"发现: {'; '.join(paper.key_findings[:3])}\n\n"
        "Assess the evidence quality."
    )
    try:
        result = chat_json(QUALITY_PROMPT, user_prompt, max_tokens=1536)

        # Validate and clamp values
        evidence_level = result.get("evidence_level", 4)
        if not isinstance(evidence_level, int) or evidence_level < 1 or evidence_level > 4:
            evidence_level = 4

        method_rigor = result.get("method_rigor", 3)
        if not isinstance(method_rigor, int) or method_rigor < 1 or method_rigor > 5:
            method_rigor = 3

        valid_study_types = {
            "meta_analysis", "systematic_review", "rct", "cohort", "case_control",
            "cross_sectional", "survey", "theoretical", "simulation", "empirical",
            "review", "other"
        }
        study_type = result.get("study_type", "other")
        if study_type not in valid_study_types:
            study_type = "other"

        valid_directions = {"positive", "negative", "mixed", "null", "not_applicable"}
        effect_direction = result.get("effect_direction", "not_applicable")
        if effect_direction not in valid_directions:
            effect_direction = "not_applicable"

        return {
            "study_type": study_type,
            "evidence_level": evidence_level,
            "method_rigor": method_rigor,
            "sample_or_data": result.get("sample_or_data", ""),
            "effect_direction": effect_direction,
            "temporal_scope": result.get("temporal_scope", ""),
            "quality_note": result.get("quality_note", ""),
        }
    except Exception:  # noqa: BLE001
        return {
            "study_type": "other",
            "evidence_level": 4,
            "method_rigor": 3,
            "sample_or_data": "",
            "effect_direction": "not_applicable",
            "temporal_scope": "",
            "quality_note": "",
        }


def build_comparison_matrix(papers: list[ExtractedPaper]) -> list[dict]:
    """Build a structured comparison of the extracted papers.

    Returns a list of row dicts suitable for rendering as a Markdown table:
    study, year, type, evidence_level, rigor, sample, key_finding, direction.
    """
    rows = []
    for p in papers:
        q = getattr(p, "quality", None) or {}
        rows.append({
            "study": p.title[:60],
            "year": p.year or "",
            "type": q.get("study_type", ""),
            "evidence": f"L{q.get('evidence_level', '?')}",
            "rigor": f"{q.get('method_rigor', '?')}/5",
            "sample": q.get("sample_or_data", "")[:40],
            "key_finding": (p.key_findings[0] if p.key_findings else "")[:80],
            "direction": q.get("effect_direction", ""),
        })
    return rows


def render_comparison_md(rows: list[dict]) -> str:
    """Render the comparison matrix as a Markdown table."""
    if not rows:
        return ""
    headers = ["Study", "Year", "Type", "Evidence", "Rigor", "Sample/Data", "Key Finding", "Direction"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [
            r.get("study", ""), str(r.get("year", "")), r.get("type", ""),
            r.get("evidence", ""), r.get("rigor", ""), r.get("sample", ""),
            r.get("key_finding", ""), r.get("direction", ""),
        ]
        # Escape pipes in cells
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
