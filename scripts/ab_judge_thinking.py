"""Golden-set A/B: judge thinking ON vs OFF.

Holds ground text constant (abstracts only — the same base the human audit
judged against) and compares each configuration's verdicts against the
golden human labels (tests/golden/verifier_calibration_201038.json), plus
the canary false-accept rate. Also reports wall time per configuration.

Usage: python scripts/ab_judge_thinking.py
"""

from __future__ import annotations

import json
import time

from citens.agents.verifier import canary_check, verify_claims
from citens.config import settings
from citens.grounding import ChunkStore, CitationTable, parse_claims_from_review
from citens.models import ExtractedPaper

RUN = "runs/机器学习股票收益预测-20260818_201038"
GOLDEN = "tests/golden/verifier_calibration_201038.json"


def norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def main() -> None:
    with open(f"{RUN}/steps/04_extracted.json", encoding="utf-8") as f:
        extracted = [ExtractedPaper(**p) for p in json.load(f)]
    with open(f"{RUN}/review.md", encoding="utf-8") as f:
        review = f.read()
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    # golden labels attach to paragraph-sized claims from the OLD sentence
    # parser; the current parser yields sentence-level claims. Each sentence
    # inherits the human label of the golden blob that contains it.
    human_blobs = [
        (norm(c["claim_text"]), c["human"]) for c in golden["claims"]
    ]

    def human_label(claim_text: str) -> str | None:
        n = norm(claim_text)
        return next((h for blob, h in human_blobs if n in blob), None)

    claims = parse_claims_from_review(review)
    table = CitationTable(extracted)
    store = ChunkStore()
    store.build_from(extracted, fetch_full=False)  # abstracts: audit-era base

    report = {}
    for label, flag in (
        ("thinking=HIGH(default)", True),
        ("thinking=LOW", "low"),
        ("thinking=NONE", False),
    ):
        settings.judge_thinking = flag
        t0 = time.monotonic()
        results, precision = verify_claims(claims, table, store)
        elapsed = time.monotonic() - t0
        canary = canary_check(table, store)

        matched = agree = 0
        lenient = strict = 0
        rank = {"unsupported": 0, "background": 1, "partial": 2, "supported": 3}
        for r in results:
            h = human_label(r.claim_text)
            if h is None:
                continue
            matched += 1
            m = r.verdict.value
            if m == h:
                agree += 1
            elif rank.get(m, 0) > rank.get(h, 0):
                lenient += 1
            else:
                strict += 1
        report[label] = {
            "wall_seconds": round(elapsed, 1),
            "precision": round(precision, 3),
            "matched": matched,
            "agreement": round(agree / matched, 3) if matched else None,
            "lenient": lenient,
            "strict": strict,
            "canary_false_accept_rate": canary["false_accept_rate"],
            "canary_caught": canary["caught"],
            "canary_injected": canary["injected"],
        }
        print(label, report[label], flush=True)

    with open("judge_thinking_ab.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("saved judge_thinking_ab.json")


if __name__ == "__main__":
    main()
