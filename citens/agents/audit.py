"""Absence-detection agent (retrieval audit).

One-shot retrieval pipelines cannot see what they missed. This agent looks at
the papers that WERE retrieved and names the canonical works (classic papers,
landmark journals) that a survey of this topic would be expected to cover but
which are absent — a retrieval audit. The pipeline turns its output into
targeted supplementary queries (the missing work's title/venue + author), so
absence drives retrieval rather than being silently accepted.

Canonical-invention guardrail: the agent is told the missing entries must be
real, well-known works (NOT generated); the orchestrator additionally requires
the supplementary round to retrieve actual papers matching those queries — a
hallucinated canonical paper simply returns nothing and is discarded.
"""

from __future__ import annotations

from citens.llm import chat_json

SYSTEM_PROMPT = """You are a literature-retrieval auditor. You are given a research topic and the \
list of papers currently included in a survey of that topic.

Your job: identify CANONICAL WORK that is missing. Think of the classic papers, \
landmark methods, and foundational authors that ANY credible survey of this topic would be \
expected to cover, and check whether they are absent from the provided list.

CRITICAL RULES:
1. ONLY name works you are CERTAIN are real, well-known publications in this field (title, \
author, year). NEVER invent, guess, or generate plausible-sounding papers — a hallucinated \
"missing classic" poisons the audit.
2. Do NOT name works already present in the list.
3. Distinguish the missing canonical papers from the retrieved ones.
4. Also flag missing JOURNAL VENUES only if a whole sub-area of the topic is evidently \
uncovered (e.g. the list contains only CS venues but the topic has a physics tradition).

Output JSON:
{
  "absent_canonical_papers": [
    {"title": "exact title", "authors": "lead author or 'Unknown'", "year": 2010, "note": "why this is canonical"}
  ],
  "missing_venue_areas": ["sub-area that appears entirely uncovered"],
  "audit_note": "one-line overall assessment"
}"""


def audit_coverage(
    topic: str,
    included_titles: list[str],
) -> dict:
    """Audit the retrieved set against the expected canonical literature.

    Returns a dict with ``absent_canonical_papers`` (list of dicts with
    title/authors/year/note) and ``missing_venue_areas``.
    """
    titles_text = "\n".join(f"- {t}" for t in included_titles) or "(none)"
    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"当前纳入综述的论文 / Papers currently included:\n{titles_text}\n\n"
        "Audit: which canonical works are MISSING from this set? "
        "Only real, well-known works — never invent."
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=2048)
    except Exception:  # noqa: BLE001
        # reasoning models occasionally emit unparseable JSON; degrade gracefully
        return {"absent_canonical_papers": [], "missing_venue_areas": [], "audit_note": "audit failed, skipped"}
    papers = result.get("absent_canonical_papers", [])
    papers = [p for p in papers if isinstance(p, dict) and str(p.get("title", "")).strip()]
    return {
        "absent_canonical_papers": papers,
        "missing_venue_areas": result.get("missing_venue_areas", []),
        "audit_note": result.get("audit_note", ""),
    }


def missing_to_queries(audit: dict) -> list[str]:
    """Convert the audit output into targeted English search queries.

    Each missing paper becomes a title+author query (high precision for the
    absent canonical work); missing venue areas become topical queries.
    """
    queries: list[str] = []
    for p in audit.get("absent_canonical_papers", [])[:5]:
        title = str(p.get("title", "")).strip()
        if not title:
            continue
        q = title
        author = str(p.get("authors", "")).strip()
        if author and author.lower() not in {"unknown", "n/a"}:
            q = f"{q} {author}"
        queries.append(q[:200])
    for area in audit.get("missing_venue_areas", [])[:2]:
        area = str(area).strip()
        if area:
            queries.append(area[:200])
    return queries[:6]
