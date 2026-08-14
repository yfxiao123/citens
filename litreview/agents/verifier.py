"""Citation-verification agent (the "trustworthy" signature).

For every claim the writer made that carries a citation, the Verifier checks it
against the cited paper's abstract (ground text) using an LLM-as-judge. It
returns a verdict per claim and an aggregate citation-precision metric — the
headline number for the README ("X% of claims are supported by their cited
sources"). Unsupported claims are surfaced for rewriting / removal.

Claims are processed in batches to keep the call count (and cost) bounded.
"""

from __future__ import annotations

from litreview.grounding import ChunkStore, CitationTable
from litreview.llm import chat_json
from litreview.models import Claim, VerificationResult, Verdict

SYSTEM_PROMPT = """You are a citation-verification expert. You are given several CLAIMS (each citing \
one or more papers by [index]) and the ABSTRACTS of the cited papers.

This is a literature *synthesis* review: claims are often reasonable interpretations or \
syntheses, not verbatim quotes. So judge whether each claim is GROUNDED IN the cited abstract(s):

- "supported":   the claim is consistent with, and reasonably supported or inferable from, the \
cited abstract(s). It does not need to be stated verbatim.
- "partial":     the claim is broadly consistent but overstates, adds specifics the abstract does \
not back, or mixes supported and unsupported elements.
- "unsupported": the claim CONTRADICTS the cited abstract, attributes a finding/method the abstract \
gives no plausible basis for, or cites the wrong paper. Reserve this for genuine mis-grounding.

Judge only against the provided abstracts. When in doubt between supported and partial, prefer \
"supported"; between partial and unsupported, prefer "partial".

Output JSON only:
{"results": [
  {"claim_index": 0, "verdict": "supported", "note": "abstract supports ..."},
  {"claim_index": 1, "verdict": "unsupported", "note": "abstract contradicts ..."}
]}"""

_BATCH_SIZE = 6


def _build_context(
    indices: list[int],
    table: CitationTable,
    chunk_store: ChunkStore,
    query: str,
) -> str:
    """For each cited paper, expose its abstract + the chunks most relevant to
    the claims being checked (RAG-lite over full text when available)."""
    lines = []
    for idx in indices:
        pid = table.paper_id(idx)
        retrieved = chunk_store.retrieve(pid, query, k=4)
        if not retrieved:
            lines.append(f"[{idx}] {table.label(idx)}\n(ABSTRACT UNAVAILABLE — cannot verify)\n")
            continue
        kinds = {c.kind.value for c in retrieved}
        body = "\n".join(c.text for c in retrieved)
        lines.append(f"[{idx}] {table.label(idx)} [{','.join(kinds)}]\n{body}\n")
    return "\n".join(lines)


def verify_claims(
    claims: list[Claim],
    table: CitationTable,
    chunk_store: ChunkStore,
    *,
    batch_size: int = _BATCH_SIZE,
    on_progress=None,
) -> tuple[list[VerificationResult], float]:
    """Verify every cited claim.

    Claims whose cited source has no ground text are marked ``unverifiable``
    and excluded from the precision denominator. Returns (results, precision)
    where precision = (supported + partial) / (verifiable claims).
    """
    results: list[VerificationResult] = []
    total = len(claims)
    unverifiable: list[VerificationResult] = []
    to_check: list[tuple[int, Claim]] = []

    # First pass: split out unverifiable claims (no abstract for any cited source).
    for claim in claims:
        has_ground = any(
            chunk_store.has(table.paper_id(i)) for i in claim.citation_indices
        )
        if not has_ground:
            unverifiable.append(
                VerificationResult(
                    claim_text=claim.text,
                    verdict=Verdict.UNVERIFIABLE,
                    citation_indices=claim.citation_indices,
                    note="cited source(s) have no abstract available",
                )
            )
        else:
            to_check.append((len(results) + len(unverifiable), claim))

    # Second pass: judge the verifiable claims in batches.
    verifiable = [c for _, c in to_check]
    for start in range(0, len(verifiable), batch_size):
        batch = verifiable[start : start + batch_size]
        if on_progress:
            on_progress(start + len(batch), total)
        cited_indices = sorted(
            {i for c in batch for i in c.citation_indices if chunk_store.has(table.paper_id(i))}
        )
        batch_query = " ".join(c.text for c in batch)
        context = _build_context(cited_indices, table, chunk_store, batch_query)
        claim_lines = "\n".join(f"Claim {j}: {c.text}" for j, c in enumerate(batch))
        user_prompt = (
            f"Source abstracts of cited papers:\n{context}\n\n"
            f"Claims to verify:\n{claim_lines}\n\n"
            "Return a verdict for each claim_index (0-based within this batch)."
        )
        try:
            raw = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=2048)
            verdicts = {int(r.get("claim_index", -1)): r for r in raw.get("results", [])}
        except Exception as e:  # noqa: BLE001
            print(f"    verify batch failed: {e}")
            verdicts = {}

        for j, claim in enumerate(batch):
            entry = verdicts.get(j, {})
            verdict_str = str(entry.get("verdict", "partial")).lower().strip()
            if verdict_str not in {v.value for v in Verdict}:
                verdict_str = "partial"
            results.append(
                VerificationResult(
                    claim_text=claim.text,
                    verdict=Verdict(verdict_str),
                    citation_indices=claim.citation_indices,
                    note=entry.get("note", ""),
                )
            )

    # Append unverifiable claims at the end.
    results.extend(unverifiable)

    verifiable_results = [r for r in results if r.verdict != Verdict.UNVERIFIABLE]
    if verifiable_results:
        ok = sum(1 for r in verifiable_results if r.verdict in {Verdict.SUPPORTED, Verdict.PARTIAL})
        precision = ok / len(verifiable_results)
    else:
        precision = 0.0
    return results, precision
