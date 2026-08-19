"""Profile evidence_bias controls writer-excerpt ranking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from citens.orchestration.support import number_dense_excerpts  # noqa: E402
from citens.profiles import load_profile  # noqa: E402


class _Chunk:
    def __init__(self, text):
        from citens.models import ChunkKind

        self.text = text
        self.kind = ChunkKind.FULLTEXT


class _P:
    id = "p1"
    title = "T"


class _Store:
    def retrieve(self, pid, query, k=6):
        return [
            _Chunk("plain prose about methodology without any figures"),
            _Chunk("accuracy improved to 92.5% with Sharpe 1.37"),
        ][:k]


def test_default_bias_prefers_number_dense_chunks():
    out = number_dense_excerpts([0], [_P()], _Store(), query="q")
    assert "92.5%" in out.split("—")[1][:60]


def test_bias_none_keeps_bm25_order():
    out = number_dense_excerpts([0], [_P()], _Store(), query="q", bias="none")
    assert out.split("—")[1].strip().startswith("plain prose")


def test_profile_defaults_and_override():
    finance = load_profile("finance")
    assert finance.evidence_bias == "number_density"
