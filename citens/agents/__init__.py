"""Pipeline agents."""

from __future__ import annotations

from citens.agents.audit import audit_coverage, missing_to_queries
from citens.agents.clarify import generate_clarifying_questions
from citens.agents.defense import challenge_verdict, review_unsupported_claims
from citens.agents.extract import extract_papers
from citens.agents.filter import filter_papers
from citens.agents.health import check_health
from citens.agents.intent import detect_intent
from citens.agents.organize import organize_themes
from citens.agents.planner import generate_keywords
from citens.agents.reflector import reflect
from citens.agents.rewriter import rewrite_unsupported_claims
from citens.agents.synth import synthesize
from citens.agents.verifier import verify_claims
from citens.agents.verifier_trigger import collect_unsupported_queries
from citens.agents.writer import write_review, write_review_body

__all__ = [
    "audit_coverage",
    "challenge_verdict",
    "check_health",
    "collect_unsupported_queries",
    "detect_intent",
    "extract_papers",
    "filter_papers",
    "generate_clarifying_questions",
    "generate_keywords",
    "missing_to_queries",
    "organize_themes",
    "reflect",
    "review_unsupported_claims",
    "rewrite_unsupported_claims",
    "synthesize",
    "verify_claims",
    "write_review",
    "write_review_body",
]
