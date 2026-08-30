"""Information-extraction agent: deep structured fields from each paper.

Papers are extracted in BATCHES (default 4 per LLM call) on the thread pool
(:func:`citens.llm.run_concurrent`) — a 100-paper extract stage costs ~25
calls instead of 100. A paper the batch response omits (parse gaps,
truncation) falls back to a single-paper call, so extraction never silently
drops fields.

Quality assessment (study type, evidence level, method rigor) is folded into
the SAME call: two prompts over the same abstract doubled the calls without
adding information.
"""

from __future__ import annotations

from citens.llm import chat_json, run_concurrent
from citens.models import ExtractedPaper, ScoredPaper

SYSTEM_PROMPT = """You are an academic-paper analyst. Given a research topic and a paper, extract \
DEEP structured information from its abstract. Go beyond surface-level summary — extract \
specific, verifiable details.

Output JSON (include every field):
{
  "system_name": "the NAMED framework/model/method the paper proposes, exactly \
as the community calls it ('TALLRec', 'FactorVAE', 'Avellaneda-Stoikov', \
'Kyle model'); empty string if the paper proposes no named system",
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
  "effect_direction": "one of: positive, negative, mixed, null, not_applicable",
  "evidence_level": 1-4 where: 1 = meta-analysis or systematic review, 2 = RCT or \
high-quality experimental, 3 = observational (cohort/case-control/cross-sectional/survey), \
4 = theoretical/model/opinion/case-report",
  "method_rigor": 1-5 where: 5 = rigorous (clear methodology, adequate data, tests, \
controls), 3 = adequate with concerns, 1 = unclear or flawed",
  "temporal_scope": "time period covered by the study (from abstract, if mentioned)",
  "quality_note": "one-line assessment of methodological strengths/weaknesses"
}

CRITICAL RULES:
1. Findings must be SPECIFIC — include effect direction, magnitude, or statistical \
significance when the abstract mentions them. "Method X outperforms Y by 15% on \
dataset Z" is good; "Method X is effective" is bad.
1b. NUMBERS ARE MANDATORY CARGO: if the abstract states ANY quantitative result \
(percentages, basis points, sample sizes, dataset scale, error reductions, \
out-of-sample gains), that number MUST survive verbatim inside a finding, with \
its metric name and comparator. A finding that drops the number it came from is \
a failed extraction. Produce 3-5 findings when the abstract supports them.
2. Methodology must name the specific technique, not just the broad category.
3. Limitations include both author-acknowledged and evident-from-abstract issues.
4. Only extract what the abstract actually says — do NOT infer or fabricate.

When given SEVERAL numbered papers at once, extract EVERY paper independently \
and return one entry per paper:
{"papers": [{"paper_index": 0, ...same fields as above...},
            {"paper_index": 1, ...}]}"""

_BATCH_SIZE = 4
_BATCH_MAX_TOKENS = 12288


def _s(v) -> str:
    """LLM JSON fields come back as null / non-strings often enough to matter."""
    return v if isinstance(v, str) else ""


def _lst(v) -> list:
    return v if isinstance(v, list) else []


def _build_extracted(paper: ScoredPaper, result: dict, assess_quality: bool) -> ExtractedPaper:
    quality: dict = {}
    if assess_quality:
        from citens.agents.quality import _validated_quality

        quality = _validated_quality(result)

    return ExtractedPaper(
        **paper.model_dump(exclude={"id"}),
        system_name=_s(result.get("system_name")),
        research_question=_s(result.get("research_question")),
        methodology=_s(result.get("methodology")),
        key_findings=_lst(result.get("key_findings")),
        limitations=_lst(result.get("limitations")),
        relevance_to_topic=_s(result.get("relevance_to_topic")),
        quality=quality,
    )


def _extract_one(paper: ScoredPaper, topic: str, assess_quality: bool = True) -> ExtractedPaper:
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"论文信息 / Paper:\n{paper.brief()}\n\n"
        "Extract the deep structured information."
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=4096, cheap=True)
    except Exception as e:  # noqa: BLE001
        print(f"    extract failed ({paper.title[:40]}): {e}")
        result = {}
    return _build_extracted(paper, result, assess_quality)


def _extract_batch(
    batch: list[ScoredPaper], topic: str, assess_quality: bool = True
) -> list[ExtractedPaper]:
    """Extract one batch in a single call; per-paper fallback for gaps."""
    paper_lines = "\n\n".join(
        f"--- Paper {i} ---\n{p.brief()}" for i, p in enumerate(batch)
    )
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"候选论文 / Papers:\n{paper_lines}\n\n"
        f"Extract the deep structured information for ALL {len(batch)} papers."
    )
    by_index: dict[int, dict] = {}
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=_BATCH_MAX_TOKENS, cheap=True)
        for entry in result.get("papers", []):
            if isinstance(entry, dict) and "paper_index" in entry:
                try:
                    by_index[int(entry["paper_index"])] = entry
                except (TypeError, ValueError):
                    continue
    except Exception as e:  # noqa: BLE001
        print(f"    batch extract failed ({e}); falling back to per-paper")

    out: list[ExtractedPaper] = []
    for i, paper in enumerate(batch):
        entry = by_index.get(i)
        if entry is None:
            out.append(_extract_one(paper, topic, assess_quality))
        else:
            out.append(_build_extracted(paper, entry, assess_quality))
    return out


def extract_papers(
    papers: list[ScoredPaper],
    topic: str,
    *,
    on_progress=None,
    assess_quality: bool = True,
) -> list[ExtractedPaper]:
    """Extract deep structured fields for each filtered paper (batched, concurrent).

    Args:
        papers: Filtered papers to extract from
        topic: Research topic
        on_progress: Progress callback
        assess_quality: If True, also assess evidence level and method rigor
    """
    total = len(papers)
    if total == 0:
        return []
    batches = [papers[s : s + _BATCH_SIZE] for s in range(0, total, _BATCH_SIZE)]

    done = 0

    def on_done(_i, batch, _result):
        nonlocal done
        done += len(batch)
        if on_progress:
            on_progress(done, total, f"批次 {_i + 1}/{len(batches)}（{len(batch)} 篇）")

    results = run_concurrent(
        lambda _i, b: _extract_batch(b, topic, assess_quality), batches, on_done=on_done
    )
    return [p for pair in results for p in pair]
