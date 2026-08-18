"""Citation-verification agent (the "trustworthy" signature).

For every claim the writer made that carries a citation, the Verifier checks it
against the cited paper's ground text (full text when fetched, else abstract)
using an LLM-as-judge on the STRONG model tier. It returns a verdict per claim
and an aggregate citation-precision metric — the headline number for the README
("X% of claims are supported by their cited sources"). Unsupported claims are
surfaced for rewriting / removal.

Claims are processed in batches to keep the call count (and cost) bounded;
batches are independent and run on a thread pool.
"""

from __future__ import annotations

from citens.grounding import ChunkStore, CitationTable
from citens.llm import chat_json, run_concurrent
from citens.models import Claim, Verdict, VerificationResult

SYSTEM_PROMPT = """You are a citation-verification expert. You are given several CLAIMS (each citing \
one or more papers by [index]) and the GROUND TEXT of the cited papers (abstracts, or full-text \
excerpts when available).

This is a literature *synthesis* review: claims are often interpretations or syntheses, not \
verbatim quotes. So judge whether each claim is GROUNDED IN the cited sources, on this \
five-grade scale:

- "supported":     the claim is consistent with, and supported by, the cited ground text — with \
no material overstatement. It does not need to be stated verbatim.
- "partial":       the claim's core is grounded but it overstates, adds specifics the ground \
text does not back, or mixes grounded and ungrounded elements.
- "background":    the cited source supports the field's CONTEXT only — it does not address the \
specific relationship, method, or magnitude the claim asserts. Typical cause: citing a survey or \
an adjacent paper as if it were primary evidence.
- "contradictory": the cited source conflicts with, or materially narrows, the claim. A \
disagreement is content the review must acknowledge — do NOT grade it away.
- "unsupported":   the claim attributes a finding, method, or detail the cited ground text gives \
NO plausible basis for, or cites the wrong paper.

REVIEW-SOURCE RULE: a paper tagged [REVIEW] is a survey, not primary evidence. It may back \
background/context claims, but a claim about experimental findings, methods, or magnitudes \
cited ONLY to [REVIEW] papers is "background" at best — primary claims need primary sources.

CALIBRATION RULES (derived from a human audit of real verdicts — apply strictly):

1. NO-GROUND-TEXT RULE: a citation whose context says "NO GROUND TEXT" contributes NOTHING to \
the claim's support. Judge only against the sources that have ground text. If the claim's core \
assertion is ABOUT such a paper (its method, findings, or design), the verdict is "unsupported" \
— citing a paper you cannot see is not evidence.
2. INTERPRETIVE FRAMING RULE: when the factual core is supported but the claim adds interpretive \
framing the source does not make ("开创了范式 / opened a new paradigm", "回答了不同层次的问题", \
"the field's central question has shifted"), the verdict is "partial", not "supported".
3. MULTI-CITATION RULE: check EVERY [index] attached to a claim, not just the closest one. The \
claim is "supported" only if each cited paper's ground text backs the part it is attached to; \
citations that back nothing make the claim "partial" at best.

Judge only against the provided ground text. Grade what the text actually shows — do not \
resolve doubt toward the lenient side.

Output JSON only:
{"results": [
  {"claim_index": 0, "verdict": "supported", "note": "ground text supports ..."},
  {"claim_index": 1, "verdict": "background", "note": "cited source is a survey; no primary evidence"}
]}"""

_BATCH_SIZE = 10


def _claim_ground_key(
    claim: Claim, table: CitationTable, chunk_store: ChunkStore
) -> str:
    """Stable identity of (claim text, citations, their ground text).

    A compose round that re-states an unchanged claim against unchanged
    ground text will re-judge it identically — so later rounds can reuse the
    verdict instead of paying for it again."""
    import hashlib

    parts = [claim.text, ",".join(str(i) for i in sorted(set(claim.citation_indices)))]
    for i in sorted(set(claim.citation_indices)):
        pid = table.paper_id(i)
        chunks_text = "|".join(c.text for c in chunk_store.chunks_for(pid))
        parts.append(f"{pid}:{hashlib.sha1(chunks_text.encode()).hexdigest()[:12]}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()


def _build_context(
    indices: list[int],
    table: CitationTable,
    chunk_store: ChunkStore,
    query: str,
) -> str:
    """For each cited paper, expose its abstract + the chunks most relevant to
    the claims being checked (RAG-lite over full text when available).

    Review-type papers are tagged ``[REVIEW]`` so the judge can apply the
    review-source rule (a survey is context, not primary evidence)."""
    lines = []
    for idx in indices:
        pid = table.paper_id(idx)
        retrieved = chunk_store.retrieve(pid, query, k=4)
        if not retrieved:
            lines.append(
                f"[{idx}] {table.label(idx)}\n"
                "(NO GROUND TEXT — abstract unavailable; this citation contributes "
                "nothing to any claim)\n"
            )
            continue
        kinds = {c.kind.value for c in retrieved}
        body = "\n".join(c.text for c in retrieved)
        review_tag = " [REVIEW]" if _is_review(table, idx) else ""
        lines.append(f"[{idx}] {table.label(idx)}{review_tag} [{','.join(kinds)}]\n{body}\n")
    return "\n".join(lines)


def _is_review(table: CitationTable, index: int) -> bool:
    papers = getattr(table, "papers", None) or []
    return bool(0 <= index < len(papers) and getattr(papers[index], "is_review", False))


def claim_stack_stats(claims: list[Claim]) -> dict:
    """Citation-stacking lint: a claim wearing many [n] markers is hard to
    ground (the multi-citation rule) and hard to read. >4 citations on one
    claim is a defect per the writer's stacking cap."""
    counts = [len(c.citation_indices) for c in claims]
    return {
        "max_citations_per_claim": max(counts) if counts else 0,
        "stacked_claims": sum(1 for n in counts if n > 4),
    }


def verify_claims(
    claims: list[Claim],
    table: CitationTable,
    chunk_store: ChunkStore,
    *,
    batch_size: int = _BATCH_SIZE,
    on_progress=None,
    verdict_cache: dict | None = None,
) -> tuple[list[VerificationResult], float]:
    """Verify every cited claim.

    Claims whose cited source has no ground text are marked ``unverifiable``
    and excluded from the precision denominator. Returns (results, precision)
    where precision = (supported + partial) / (verifiable claims); background,
    contradictory and unsupported all count against it.

    ``verdict_cache`` (threaded across compose rounds by the pipeline) reuses
    verdicts for claims whose text, citations, and cited ground text are all
    unchanged — the re-compose rounds only pay for what actually changed.
    """
    results: list[VerificationResult] = []
    total = len(claims)
    # results placed by CLAIM index — the rewriter / spot-check index into
    # ver_results parallel to claims, so order must never drift (unverifiable
    # used to be appended at the end, misaligning every later index)
    placed: list[VerificationResult | None] = [None] * total
    to_check: list[tuple[int, Claim]] = []

    # First pass: split out unverifiable claims (no abstract for any cited source).
    for slot, claim in enumerate(claims):
        has_ground = any(
            chunk_store.has(table.paper_id(i)) for i in claim.citation_indices
        )
        if not has_ground:
            placed[slot] = VerificationResult(
                claim_text=claim.text,
                verdict=Verdict.UNVERIFIABLE,
                citation_indices=claim.citation_indices,
                note="cited source(s) have no abstract available",
            )
        else:
            to_check.append((slot, claim))

    # Cache hits: unchanged claims skip the judge entirely.
    n_cached = 0
    if verdict_cache is not None:
        remaining: list[tuple[int, Claim]] = []
        for slot, claim in to_check:
            hit = verdict_cache.get(_claim_ground_key(claim, table, chunk_store))
            if hit is not None:
                placed[slot] = hit.model_copy()
                n_cached += 1
            else:
                remaining.append((slot, claim))
        to_check = remaining

    # Second pass: judge the verifiable claims in concurrent batches.
    verifiable = [c for _, c in to_check]
    batches = [
        (start, verifiable[start : start + batch_size])
        for start in range(0, len(verifiable), batch_size)
    ]
    done = 0

    def _verify_batch(_i: int, pair: tuple[int, list[Claim]]) -> list[VerificationResult]:
        _, batch = pair
        cited_indices = sorted(
            {i for c in batch for i in c.citation_indices if chunk_store.has(table.paper_id(i))}
        )
        batch_query = " ".join(c.text for c in batch)
        context = _build_context(cited_indices, table, chunk_store, batch_query)
        claim_lines = "\n".join(f"Claim {j}: {c.text}" for j, c in enumerate(batch))
        user_prompt = (
            f"Ground text of cited papers:\n{context}\n\n"
            f"Claims to verify:\n{claim_lines}\n\n"
            "Return a verdict for each claim_index (0-based within this batch)."
        )
        try:
            raw = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=8192, strong=True)
            verdicts = {int(r.get("claim_index", -1)): r for r in raw.get("results", [])}
        except Exception as e:  # noqa: BLE001
            print(f"    verify batch failed: {e}")
            verdicts = {}

        out: list[VerificationResult] = []
        for j, claim in enumerate(batch):
            entry = verdicts.get(j, {})
            verdict_str = str(entry.get("verdict", "")).lower().strip()
            if verdict_str not in {v.value for v in Verdict}:
                # a missing/malformed judge response is "we could not verify",
                # never a free pass into the precision numerator
                verdict_str = "unverifiable"
                entry = {
                    **entry,
                    "note": (str(entry.get("note", "")) or "")
                    + " [judge returned no usable verdict]"
                }
            out.append(
                VerificationResult(
                    claim_text=claim.text,
                    verdict=Verdict(verdict_str),
                    citation_indices=claim.citation_indices,
                    note=entry.get("note", ""),
                )
            )
        return out

    def on_done(_i, _pair, batch_results):
        nonlocal done
        done += len(batch_results)
        if on_progress:
            on_progress(done, total)

    for batch_results in run_concurrent(_verify_batch, batches, on_done=on_done):
        results.extend(batch_results)

    # place fresh verdicts at their claim slots + store for later rounds
    for (slot, claim), res in zip(to_check, results, strict=False):
        placed[slot] = res
        if verdict_cache is not None:
            verdict_cache[_claim_ground_key(claim, table, chunk_store)] = res
    if n_cached and on_progress:
        on_progress(done + n_cached, total)

    results = [r for r in placed if r is not None]

    verifiable_results = [r for r in results if r.verdict != Verdict.UNVERIFIABLE]
    if verifiable_results:
        ok = sum(1 for r in verifiable_results if r.verdict in {Verdict.SUPPORTED, Verdict.PARTIAL})
        precision = ok / len(verifiable_results)
    else:
        precision = 0.0
    return results, precision


SPOT_CHECK_PROMPT = """You are a STRICT citation auditor doing a second pass. A previous verifier \
judged these claims "supported". Your job is to catch LENIENT judgments: claims that \
were passed but should have been "partial" or "unsupported".

Apply a HARSHER standard than a first pass:
- "confirm":   the claim is solidly grounded — no overstatement at all.
- "downgrade": the claim overstates, adds ungrounded specifics, or stretches the \
source; it should have been "partial" or "unsupported".

Output JSON only:
{"results": [{"claim_index": 0, "verdict": "confirm", "note": "..."}]}"""


def spot_check_supported(
    claims: list[Claim],
    ver_results: list[VerificationResult],
    table: CitationTable,
    chunk_store: ChunkStore,
    *,
    sample_size: int = 8,
) -> dict:
    """Re-audit a sample of "supported" verdicts under a stricter standard.

    The rewrite loop can inflate precision by making claims vaguer and having
    the same judge pass them — this is the counterweight. Returns a summary
    dict (never raises; a failed audit is reported, not fatal).
    """
    import random

    supported_idx = [
        i for i, r in enumerate(ver_results)
        if r.verdict.value == "supported" and i < len(claims)
    ]
    if not supported_idx:
        return {"sampled": 0, "downgraded": 0, "agreement_rate": None}

    sample = sorted(random.sample(supported_idx, min(sample_size, len(supported_idx))))
    context_lines: list[str] = []
    claim_lines: list[str] = []
    for j, i in enumerate(sample):
        claim = claims[i]
        claim_lines.append(f"Claim {j}: {claim.text}")
        for cite in claim.citation_indices[:3]:
            pid = table.paper_id(cite)
            chunks = chunk_store.chunks_for(pid)[:3]
            if chunks:
                body = "\n".join(c.text for c in chunks)
                context_lines.append(f"[{cite}] {table.label(cite)}\n{body}\n")

    try:
        raw = chat_json(
            SPOT_CHECK_PROMPT,
            "Ground text of cited papers:\n"
            + "\n".join(context_lines[:20])
            + "\n\nClaims previously judged 'supported':\n"
            + "\n".join(claim_lines)
            + f"\n\nAudit all {len(sample)} claims.",
            max_tokens=8192,
            strong=True,
        )
        verdicts = {int(r.get("claim_index", -1)): r for r in raw.get("results", [])}
    except Exception as e:  # noqa: BLE001
        return {"sampled": 0, "downgraded": 0, "agreement_rate": None, "error": str(e)[:200]}

    downgraded = sum(
        1
        for j in range(len(sample))
        if verdicts.get(j, {}).get("verdict", "confirm") == "downgrade"
    )
    per_claim = [
        {
            "claim_index": i,
            "verdict": verdicts.get(j, {}).get("verdict", "confirm"),
            "note": str(verdicts.get(j, {}).get("note", ""))[:200],
        }
        for j, i in enumerate(sample)
    ]
    return {
        "sampled": len(sample),
        "downgraded": downgraded,
        "agreement_rate": round(1 - downgraded / len(sample), 3) if sample else None,
        "claim_indices": sample,
        "results": per_claim,
    }


# --- canary claims: a honeypot that measures the judge's false-accept rate ---
# Deliberately unsupported claims, verified in a separate call so they never
# contaminate the real results. A judge that lets canaries pass as
# supported/partial is lenient by construction — this turns "verifier too
# lenient" from a suspicion into a measured number.
_CANARY_CLAIMS = (
    "The cited paper reports a 47.3% reduction in average patient mortality "
    "across three clinical trials.",
    "According to the cited paper, its proposed method reduces GPU training "
    "cost by an order of magnitude compared with all baselines.",
    "The cited paper proves its main theorem under the assumption that markets "
    "are frictionless and informationally efficient at all times.",
)


def canary_check(table: CitationTable, chunk_store: ChunkStore) -> dict:
    """Verify synthetic unsupported claims against real ground text.

    Each canary cites one paper that HAS ground text, so the correct verdict is
    unambiguously "unsupported". Returns the catch rate; never raises (a failed
    canary call is reported, not fatal).
    """
    indices = [
        i for i in range(len(getattr(table, "papers", []) or []))
        if chunk_store.has(table.paper_id(i))
    ]
    if not indices:
        return {"injected": 0, "caught": 0, "false_accept_rate": None}

    picks = indices[: len(_CANARY_CLAIMS)]
    claims = [
        Claim(text=text, citation_indices=[i])
        for text, i in zip(_CANARY_CLAIMS, picks, strict=False)
    ]
    context = _build_context(
        sorted({c.citation_indices[0] for c in claims}), table, chunk_store,
        "canary unsupported claims",
    )
    claim_lines = "\n".join(f"Claim {j}: {c.text}" for j, c in enumerate(claims))
    try:
        raw = chat_json(
            SYSTEM_PROMPT,
            f"Ground text of cited papers:\n{context}\n\n"
            f"Claims to verify:\n{claim_lines}\n\n"
            "Return a verdict for each claim_index (0-based within this batch).",
            max_tokens=4096,
            strong=True,
        )
        verdicts = {int(r.get("claim_index", -1)): str(r.get("verdict", "")).lower()
                    for r in raw.get("results", [])}
    except Exception as e:  # noqa: BLE001
        return {"injected": len(claims), "caught": 0, "false_accept_rate": None,
                "error": str(e)[:200]}

    details = [
        {"claim": j, "verdict": verdicts.get(j, "missing")}
        for j in range(len(claims))
    ]
    caught = sum(
        1 for d in details
        if d["verdict"] in {"unsupported", "contradictory", "background"}
    )
    return {
        "injected": len(claims),
        "caught": caught,
        "false_accept_rate": round(1 - caught / len(claims), 3) if claims else None,
        "verdicts": {str(d["claim"]): d["verdict"] for d in details},
    }
