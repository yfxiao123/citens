"""Pipeline agents."""

from __future__ import annotations

from litreview.agents.audit import audit_coverage, missing_to_queries
from litreview.agents.clarify import generate_clarifying_questions
from litreview.agents.extract import extract_papers
from litreview.agents.filter import filter_papers
from litreview.agents.organize import organize_themes
from litreview.agents.planner import generate_keywords
from litreview.agents.reflector import reflect
from litreview.agents.synth import synthesize
from litreview.agents.verifier import verify_claims
from litreview.agents.verifier_trigger import collect_unsupported_queries
from litreview.agents.writer import write_review, write_review_body

__all__ = [
    "audit_coverage",
    "collect_unsupported_queries",
    "extract_papers",
    "filter_papers",
    "generate_clarifying_questions",
    "generate_keywords",
    "missing_to_queries",
    "organize_themes",
    "reflect",
    "synthesize",
    "verify_claims",
    "write_review",
    "write_review_body",
]
