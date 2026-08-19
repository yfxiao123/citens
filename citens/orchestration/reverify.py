"""Re-verify an existing run against newly available full text.

The typical loop: a run finishes with N papers on abstract-only grounding;
`fetch_list.md` tells you which PDFs to download; you drop them into
`papers/`; then `citens reverify runs/<dir>` re-grounds every claim WITHOUT
re-running retrieval, extraction or writing — just the verifier (and the
defense pass over unsupported claims).

Updates in place: verification.json, provenance.json, meta.json precision.
"""

from __future__ import annotations

import json
import os

from citens import persistence
from citens.agents import review_unsupported_claims, verify_claims
from citens.events import EventBus, StepCompleted, StepStarted
from citens.grounding import ChunkStore, CitationTable, parse_claims_from_review
from citens.models import ExtractedPaper, Verdict
from citens.orchestration.support import _emit


def reverify(run_dir: str, bus: EventBus | None = None) -> dict:
    """Re-run verification for an existing run. Returns the summary dict."""
    review_path = os.path.join(run_dir, "review.md")
    extracted_path = os.path.join(run_dir, "steps", "04_extracted.json")
    for p, why in ((review_path, "review.md"), (extracted_path, "steps/04_extracted.json")):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"cannot re-verify {run_dir}: missing {why}")

    with open(extracted_path, encoding="utf-8") as f:
        extracted = [ExtractedPaper(**p) for p in json.load(f)]
    with open(review_path, encoding="utf-8") as f:
        review = f.read()

    old_precision = None
    ver_path = os.path.join(run_dir, "verification.json")
    if os.path.isfile(ver_path):
        with open(ver_path, encoding="utf-8") as f:
            old_precision = json.load(f).get("citation_precision")

    # re-ground: picks up PDFs dropped into papers/ since the original run
    _emit(bus, StepStarted(step="ground", title="重建溯源（含新放入的 PDF）"))
    chunk_store = ChunkStore()
    chunk_store.build_from(extracted, fetch_full=True)
    table = CitationTable(extracted)
    n_full = sum(
        1 for p in extracted
        if any(c.kind.value == "fulltext" for c in chunk_store.chunks_for(p.id))
    )
    persistence.save_json(
        run_dir, "grounding.json",
        {"with_fulltext": n_full, "total": len(extracted),
         "papers": [{"index": i, "title": p.title,
                     "has_fulltext": any(c.kind.value == "fulltext"
                                         for c in chunk_store.chunks_for(p.id)),
                     "n_chunks": len(chunk_store.chunks_for(p.id))}
                    for i, p in enumerate(extracted)]},
    )
    _emit(bus, StepCompleted(step="ground", message=f"{n_full}/{len(extracted)} 篇全文"))

    _emit(bus, StepStarted(step="verify", title="重新核验"))
    claims = parse_claims_from_review(review)
    ver_results, precision = verify_claims(claims, table, chunk_store)
    persistence.save_json(
        run_dir, "verification.json",
        {"citation_precision": precision, "total_claims": len(claims),
         "verifiable_claims": sum(
             1 for r in ver_results if r.verdict.value != "unverifiable"),
         **{v: sum(1 for r in ver_results if r.verdict.value == v)
            for v in ("supported", "partial", "background", "contradictory",
                      "unsupported", "unverifiable")},
         "results": [r.model_dump() for r in ver_results],
         "previous_precision": old_precision},
    )
    persistence.save_json(
        run_dir, "provenance.json",
        [{"claim": c.text, "section": c.section,
          "citations": [{"index": i, "label": table.label(i),
                         "paper_id": table.paper_id(i)}
                        for i in c.citation_indices],
          "verdict": r.verdict.value, "verdict_note": r.note}
         for c, r in zip(claims, ver_results, strict=False)],
    )

    # defense pass over defect claims (now with better ground text),
    # then fold overturns into the final verdicts like the main pipeline
    unsupported = [
        r
        for r in ver_results
        if r.verdict.value in ("unsupported", "background", "contradictory")
    ]
    defense = []
    if unsupported:
        _emit(bus, StepStarted(step="defense", title="双向核验（辩护律师）"))
        claims_by_text = {c.text: c for c in claims}
        unsupported_claims = [claims_by_text[r.claim_text] for r in unsupported]
        source_contexts = {}
        for i, c in enumerate(unsupported_claims):
            pid = table.paper_id(c.citation_indices[0]) if c.citation_indices else ""
            chunks = chunk_store.chunks_for(pid)
            source_contexts[i] = "\n\n".join(ch.text for ch in chunks[:3])
        defense = review_unsupported_claims(unsupported_claims, unsupported, source_contexts)
        persistence.save_step(run_dir, "08_defense", defense)

        by_text = {r.claim_text: r for r in ver_results}
        for defense_review in defense:
            if not defense_review["overturned"]:
                continue
            match = unsupported_claims[defense_review["claim_idx"]]
            target = by_text.get(match.text)
            if target is not None:
                target.verdict = Verdict.PARTIAL
                target.note = (
                    f"{target.note} "
                    f"[defense: {defense_review['defense_result'].get('rebuttal', '')[:200]}]"
                ).strip()
        verifiable = [r for r in ver_results if r.verdict.value != "unverifiable"]
        if verifiable:
            precision = (
                sum(1 for r in verifiable if r.verdict.value in ("supported", "partial"))
                / len(verifiable)
            )
        # rewrite artifacts with the post-defense verdicts
        with open(ver_path, "w", encoding="utf-8") as f:
            json.dump(
                {"citation_precision": round(precision, 3),
                 "previous_precision": old_precision,
                 "defense_overturned": sum(1 for d in defense if d["overturned"]),
                 "results": [r.model_dump() for r in ver_results]},
                f, ensure_ascii=False, indent=2,
            )
        persistence.save_json(
            run_dir, "provenance.json",
            [{"claim": c.text, "section": c.section,
              "citations": [{"index": i, "label": table.label(i),
                             "paper_id": table.paper_id(i)}
                            for i in c.citation_indices],
              "verdict": r.verdict.value, "verdict_note": r.note}
             for c, r in zip(claims, ver_results, strict=False)],
        )

    # refresh meta.json precision if the run ever wrote one
    meta_path = os.path.join(run_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["citation_precision"] = round(precision, 3)
        persistence.save_json(run_dir, "meta.json", meta)

    _emit(
        bus,
        StepCompleted(
            step="verify",
            message=(
                f"precision {old_precision:.1%} -> {precision:.1%}"
                if old_precision is not None else f"precision {precision:.1%}"
            ),
        ),
    )
    return {
        "claims": len(claims),
        "fulltext": n_full,
        "precision": precision,
        "previous_precision": old_precision,
    }
