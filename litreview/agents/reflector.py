"""Reflector agent: detect coverage gaps and trigger supplementary retrieval.

Most literature agents are one-shot. The Reflector closes the loop: after a
first pass, it inspects the synthesis gaps and — if the coverage is thin —
derives targeted supplementary search queries to fill them. The pipeline then
searches, filters, and extracts the new papers and folds them in for a second
composition pass.

The agent itself only decides *whether* to supplement and *what to query*; the
actual retrieval/extraction lives in the orchestration layer.
"""

from __future__ import annotations

from litreview.llm import chat_json
from litreview.models import SynthesisResult

SYSTEM_PROMPT = """You are a literature-review methodology critic. Given a research topic, the \
identified COVERAGE GAPS, and how many papers are currently in the review, decide whether a \
supplementary retrieval round is warranted.

- If the gaps are concrete and addressable by more literature (not just "future work"), set \
"needs_supplement" to true and produce 3-5 targeted ENGLISH search queries aimed specifically at \
those gaps.
- If the coverage is already adequate, or the gaps are genuinely open research questions that no \
retrieval will fix, set "needs_supplement" to false.

Output JSON:
{
  "needs_supplement": true,
  "rationale": "why supplement (or not)",
  "supplementary_keywords": ["english query 1", "english query 2", ...]
}"""


def reflect(
    synthesis: SynthesisResult,
    topic: str,
    current_paper_count: int,
) -> dict:
    """Decide whether to supplement and produce gap-targeted queries."""
    gaps_text = "\n".join(f"- {g}" for g in synthesis.gaps) or "(none identified)"
    user_prompt = (
        f"研究主题 / Topic: {topic}\n"
        f"当前论文数 / Current papers: {current_paper_count}\n\n"
        f"已识别的覆盖空白 / Coverage gaps:\n{gaps_text}\n\n"
        "Decide whether a supplementary retrieval round is warranted."
    )
    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=2048)
    except Exception:  # noqa: BLE001
        print("    reflect failed: unparseable LLM JSON")
        result = {}
    return {
        "needs_supplement": bool(result.get("needs_supplement", False)),
        "rationale": result.get("rationale", ""),
        "supplementary_keywords": [
            q for q in result.get("supplementary_keywords", []) if isinstance(q, str) and q.strip()
        ][:5],
    }
