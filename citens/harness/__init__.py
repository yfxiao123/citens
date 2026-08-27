"""Agentic retrieval harness (Phase 1).

The pipeline's retrieval waves (facet gap / zero-hit synonyms / thin-pool
refine) are agent decisions hardcoded in Python. This package moves the
DECIDING into the model: a budgeted tool-calling loop that perceives the
pool state and drives :func:`citens.search.search_round` /
:func:`citens.search.snowball.snowball` itself. The deterministic pipeline
remains the default; ``RunOptions.agentic_retrieval`` opts a run in.
"""

from citens.harness.loop import HarnessBudget, HarnessResult, run_retrieval_harness

__all__ = ["HarnessBudget", "HarnessResult", "run_retrieval_harness"]
