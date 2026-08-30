"""Coverage evaluation against a human survey's reference list (AutoSurvey
protocol, adapted): the one quality axis internal metrics never measured —
RECALL. Precision we know (verifier grounding); what we miss was invisible.

Given a run directory and a gold reference list (bench_data/coverage/*.json,
fetched from Semantic Scholar), compute:

* survey_recall   |our citations ∩ survey refs| / |survey refs| — how much of
                  the human survey's literature base our review covers
* core50_recall   same, restricted to the survey's 50 most-cited refs —
                  the "every RAG survey must cite these" core (an XRD-style
                  expected-docs cut that is fair to a 20-paper review)
* overlap_precision  |∩| / |our citations| — AutoSurvey's precision analog;
                  unlike the verifier's grounding precision this punishes
                  citing off-base literature the human survey ignored
* judge (optional, cheap tier): coverage / coherence / relevance of
  review.md scored 1-10 against the survey's reference titles, plus the
  top missing titles it names — the actionable part of the report

Matching is DOI-first, normalized-title fallback (arXiv preprint == published
version counts as covered).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from citens.eval.litsearch import _norm_doi, _title_tokens


def load_survey_refs(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_bib(run_dir: str | Path) -> list[dict]:
    """Minimal bib scrape of the run's references.bib: title + doi per entry."""
    text = (Path(run_dir) / "references.bib").read_text(encoding="utf-8")
    out: list[dict] = []
    for chunk in re.split(r"(?m)^@", text)[1:]:  # entries start with "@" at line start
        m_title = re.search(r"title\s*=\s*[{\"](.+?)[}\"],\s*\n", chunk + "\n")
        m_doi = re.search(r"doi\s*=\s*[{\"](.+?)[}\"],", chunk)
        if not m_title:
            continue
        out.append({
            "title": m_title.group(1).strip(),
            "doi": _norm_doi(m_doi.group(1)) if m_doi else "",
        })
    return out


def _same(a: dict, b: dict) -> bool:
    if a.get("doi") and b.get("doi") and a["doi"] == b["doi"]:
        return True
    ta, tb = _title_tokens(a.get("title", "")), _title_tokens(b.get("title", ""))
    # >=3 informative tokens: numbered/short titles ("Survey Ref 0") reduce to
    # a 2-token skeleton that matches everything — refuse to match on that
    return bool(len(ta) >= 3 and len(tb) >= 3 and ta == tb)


def coverage_metrics(ours: list[dict], survey: list[dict]) -> dict:
    """Overlap numbers. `ours` = what our review cites; `survey` = gold refs."""
    matched_gold = [g for g in survey if any(_same(o, g) for o in ours)]
    matched_ours = [o for o in ours if any(_same(o, g) for g in survey)]
    core = sorted(survey, key=lambda g: -(g.get("citations") or 0))[:50]
    core_hit = [g for g in core if any(_same(o, g) for o in ours)]
    return {
        "our_citations": len(ours),
        "survey_refs": len(survey),
        "overlap": len(matched_gold),
        "survey_recall": round(len(matched_gold) / len(survey), 4) if survey else 0.0,
        "core50_hit": len(core_hit),
        "core50_recall": round(len(core_hit) / len(core), 4) if core else 0.0,
        "overlap_precision": round(len(matched_ours) / len(ours), 4) if ours else 0.0,
        "missing_top": [
            g["title"] for g in core if g not in core_hit
        ][:10],
    }


def judge_review(
    review_md: str, topic: str, survey: list[dict], max_titles: int = 120
) -> dict:
    """Cheap-tier LLM judge (AutoSurvey-style dimensions), 1-10 each.

    The judge sees the survey's most-cited reference titles as the
    literature base and names what the review is missing — a reproducible
    stand-in for a human rater, not a replacement.
    """
    from citens import llm

    core = sorted(survey, key=lambda g: -(g.get("citations") or 0))[:max_titles]
    titles = "\n".join(f"- {g['title']}" for g in core)
    sys_p = (
        "You are a critical survey reviewer. Given a topic, the reference "
        "list of an authoritative human survey, and a generated survey, "
        'reply ONLY JSON: {"coverage": 1-10, "coherence": 1-10, '
        '"relevance": 1-10, "missing": [up to 8 reference titles the '
        "generated survey should have covered]}."
    )
    user_p = (
        f"Topic: {topic}\n\n"
        f"Human survey's literature base (top {len(core)} by citations):\n"
        f"{titles}\n\nGenerated survey:\n{review_md[:24000]}"
    )
    raw = llm.chat(sys_p, user_p, cheap=True, temperature=0.0, response_json=True)
    data = json.loads(raw)
    for k in ("coverage", "coherence", "relevance"):
        data[k] = max(1, min(10, int(data.get(k) or 0)))
    data.setdefault("missing", [])
    return data


def evaluate_run(
    run_dir: str | Path, gold_path: str | Path, *, with_judge: bool = True
) -> dict:
    run_dir = Path(run_dir)
    ours = parse_bib(run_dir)
    survey = load_survey_refs(gold_path)
    result = coverage_metrics(ours, survey)
    with open(run_dir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    result["topic"] = meta.get("topic", "")
    ver_path = run_dir / "verification.json"
    if ver_path.exists():
        with open(ver_path, encoding="utf-8") as fh:
            ver = json.load(fh)
        result["verifier_precision"] = ver.get("citation_precision")
    if with_judge:
        review = (run_dir / "review.md").read_text(encoding="utf-8")
        result["judge"] = judge_review(review, result["topic"], survey)
    return result
