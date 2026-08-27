"""Intent detection agent.

Classifies user requests into run modes to enable adaptive pipeline behavior:
- QUICK_SCAN: Fast overview, 1-2 rounds, generate summary not full review
- DEEP_REVIEW: Multi-round reflection, absence detection, verification-driven supplement
- INTERACTIVE: User-driven exploration with mid-stream adjustments

The detector analyzes the topic string and optional user hints to infer intent.
"""

from __future__ import annotations

from citens.llm import chat_json

SYSTEM_PROMPT = """You are a research-intent classifier. Given a research topic and optional user hints, \
classify the request into one of three modes:

1. "quick_scan": User wants a fast overview (1-2 rounds of retrieval). Indicators: \
brief topic, words like "overview", "summary", "quick look", "brief", time pressure hints.

2. "deep_review": User wants a comprehensive literature review with full methodology \
(multi-round reflection, absence detection, verification). Indicators: words like \
"comprehensive", "systematic", "thorough", "detailed", academic context, no time pressure.

3. "interactive": User wants to explore iteratively with mid-stream adjustments. \
Indicators: words like "help me understand", "let's explore", "guide me", open-ended \
questions, conversational tone.

Output JSON:
{"mode": "quick_scan" | "deep_review" | "interactive", "confidence": 0.0-1.0, \
"reasoning": "why this mode was chosen"}"""


def detect_intent(topic: str, user_hints: dict | None = None) -> str:
    """Detect user intent and return the recommended run mode.

    Args:
        topic: The research topic string
        user_hints: Optional dict with clarifying answers (focus, scope, etc.)

    Returns:
        One of: "quick_scan", "deep_review", "interactive"
    """
    hints_text = ""
    if user_hints:
        hints_text = "\n".join(f"{k}: {v}" for k, v in user_hints.items())

    user_prompt = (
        f"研究主题 / Topic: {topic}\n\n"
        f"用户提示 / User hints:\n{hints_text}\n\n"
        "Classify the intent into quick_scan / deep_review / interactive."
    )

    try:
        result = chat_json(SYSTEM_PROMPT, user_prompt, max_tokens=1024, thinking=False, cheap=True)
        mode = result.get("mode", "deep_review")
        if mode not in {"quick_scan", "deep_review", "interactive"}:
            mode = "deep_review"
        return mode
    except Exception:  # noqa: BLE001
        return "deep_review"
