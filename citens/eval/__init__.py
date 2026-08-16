"""Citation-precision evaluation harness.

Makes the README's precision claims reproducible: run a fixed topic set
through the pipeline, collect per-run metrics from the run directory, and
render a comparison table. Live LLM + network required — this is a
maintainer tool (``citens eval``), not a CI step. The offline parts
(collect/render) are unit-tested.
"""

from citens.eval.precision import collect_metrics, render_table

__all__ = ["collect_metrics", "render_table"]
