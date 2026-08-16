"""Pluggable chunk retrievers for citation grounding.

The verifier retrieves the chunks most relevant to a claim before judging it;
how "relevant" is computed is a swappable capability (harness-style), because
the stages want different things:

* ``bm25`` (default) — classic lexical scoring over the paper's few chunks.
  Strictly sharper than raw token-overlap, zero dependencies.
* ``keyword`` — the original token-overlap ranking (kept for A/B comparison).
* ``embedding`` — vector similarity via an OpenAI-compatible /embeddings
  endpoint (set EMBEDDING_MODEL). Falls back to BM25 on any failure — a
  missing embedding API must never kill a run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Protocol

from citens import cache
from citens.config import settings
from citens.models import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class Retriever(Protocol):
    def rank(self, chunks: Sequence[Chunk], query: str, k: int) -> list[Chunk]: ...


class KeywordRetriever:
    """Token-overlap ranking (the original behavior)."""

    def rank(self, chunks: Sequence[Chunk], query: str, k: int) -> list[Chunk]:
        qtokens = set(_terms(query))
        if not qtokens:
            return list(chunks[:k])
        ranked = sorted(
            chunks,
            key=lambda c: len(qtokens & set(_terms(c.text))),
            reverse=True,
        )
        return ranked[:k]


class BM25Retriever:
    """BM25 over one paper's chunks (corpus is small; per-call is cheap)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rank(self, chunks: Sequence[Chunk], query: str, k: int) -> list[Chunk]:
        docs = [_terms(c.text) for c in chunks]
        n = len(docs)
        if n == 0:
            return []
        avgdl = sum(len(d) for d in docs) / n
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        qterms = _terms(query)
        scores: list[float] = []
        for d in docs:
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in qterms:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (
                    tf[t] + self.k1 * (1 - self.b + self.b * len(d) / avgdl)
                )
            scores.append(s)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:k]]


class EmbeddingRetriever:
    """Cosine similarity via an OpenAI-compatible embeddings endpoint.

    Embeddings are disk-cached by text hash. Any failure (no model configured,
    API error, dimension mismatch) silently degrades to BM25 — the run's
    correctness never depends on this API being up.
    """

    def __init__(self) -> None:
        self._fallback = BM25Retriever()

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        model = settings.embedding_model
        if not model:
            return None
        cached: list[list[float] | None] = []
        missing: list[int] = []
        for i, t in enumerate(texts):
            v = cache.get("embed", {"model": model, "text": t})
            if isinstance(v, list):
                cached.append(v)
            else:
                cached.append(None)
                missing.append(i)
        if missing:
            try:
                from openai import OpenAI

                kwargs: dict = {"api_key": settings.llm_api_key}
                if settings.llm_api_base:
                    kwargs["base_url"] = settings.llm_api_base
                client = OpenAI(**kwargs)
                resp = client.embeddings.create(
                    model=model, input=[texts[i] for i in missing]
                )
                for i, item in zip(missing, resp.data, strict=False):
                    vec = list(item.embedding)
                    cached[i] = vec
                    cache.put("embed", {"model": model, "text": texts[i]}, vec)
            except Exception:  # noqa: BLE001
                return None
        return [c for c in cached if c is not None] if all(c is not None for c in cached) else None

    def rank(self, chunks: Sequence[Chunk], query: str, k: int) -> list[Chunk]:
        texts = [c.text for c in chunks] + [query]
        vecs = self._embed(texts)
        if vecs is None:
            return self._fallback.rank(chunks, query, k)

        def cos(a: list[float], b: list[float]) -> float:
            num = sum(x * y for x, y in zip(a, b, strict=False))
            da = math.sqrt(sum(x * x for x in a)) or 1e-9
            db = math.sqrt(sum(y * y for y in b)) or 1e-9
            return num / (da * db)

        qv = vecs[-1]
        order = sorted(range(len(chunks)), key=lambda i: cos(vecs[i], qv), reverse=True)
        return [chunks[i] for i in order[:k]]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """The configured retriever (constructed once, like the LLM backends)."""
    global _retriever
    if _retriever is None:
        name = settings.retriever.strip().lower()
        if name == "embedding":
            _retriever = EmbeddingRetriever()
        elif name == "keyword":
            _retriever = KeywordRetriever()
        else:
            _retriever = BM25Retriever()
    return _retriever
