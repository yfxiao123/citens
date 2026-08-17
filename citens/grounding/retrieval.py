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
from typing import Protocol, cast

from citens import cache
from citens.config import settings
from citens.models import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")

_CJK_RUN_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def _terms(text: str) -> list[str]:
    """Lexical terms: latin/digit words, CJK bigrams (Lucene CJKAnalyzer style).

    Bigrams, not single chars: single CJK chars are so common they drown the
    IDF signal; before this, ``[a-z0-9]+`` matched NOTHING in Chinese text —
    BM25 over a Chinese pool silently returned index order.
    """
    out: list[str] = []
    for m in _WORD_RE.finditer(text.lower()):
        tok = m.group(0)
        if not _CJK_RUN_RE.match(tok) or len(tok) == 1:
            out.append(tok)
        else:
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
    return out


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
        order = bm25_rank_texts([c.text for c in chunks], query, k1=self.k1, b=self.b)
        return [chunks[i] for i in order[:k]]


def bm25_rank_texts(
    texts: list[str], query: str, *, k1: float = 1.5, b: float = 0.75
) -> list[int]:
    """Indices of ``texts`` sorted by BM25 relevance to ``query`` (desc).

    Generic text-level BM25 shared by chunk retrieval and the literature
    pool's deterministic pre-recall (rank pool records before LLM screening).

    Corpora past ``_FTS5_MIN`` texts go through an in-memory SQLite FTS5
    index (C speed, stdlib) — the pure-Python scorer stays as the small-
    corpus path and the fallback when FTS5 is unavailable.
    """
    if len(texts) >= _FTS5_MIN:
        order = _fts5_rank(texts, query)
        if order is not None:
            return order
    docs = [_terms(t) for t in texts]
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
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return sorted(range(n), key=lambda i: scores[i], reverse=True)


# SQLite FTS5 is C-speed and in the stdlib; it only pays off once the corpus
# outgrows pure-Python scoring (the pool's ~140 records are instant either way)
_FTS5_MIN = 400


def _fts5_rank(texts: list[str], query: str) -> list[int] | None:
    """BM25 ordering via an in-memory FTS5 index; None on ANY failure (FTS5
    not compiled in, query syntax, …) so the caller falls back silently.

    Texts are pre-tokenized with :func:`_terms` (CJK bigrams included) and
    inserted space-joined, so the FTS tokenizer never sees raw Chinese —
    unicode61 would otherwise swallow a whole CJK run as one token.
    """
    import sqlite3

    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        conn.executemany(
            "INSERT INTO t(rowid, body) VALUES (?, ?)",
            [(i, " ".join(_terms(t))) for i, t in enumerate(texts)],
        )
        qtokens = list(dict.fromkeys(_terms(query)))
        if not qtokens:
            return list(range(len(texts)))
        match = " OR ".join(qtokens)
        # bm25() is lower-is-better (more negative = better match)
        rows = conn.execute(
            "SELECT rowid FROM t WHERE t MATCH ? ORDER BY bm25(t)", (match,)
        ).fetchall()
        matched = [r[0] for r in rows]
        # FTS omits zero-match docs entirely; the pure-Python scorer ranks
        # them last-but-present — keep that contract so callers can rely on
        # getting every index back
        seen = set(matched)
        return matched + [i for i in range(len(texts)) if i not in seen]
    except Exception:  # noqa: BLE001
        return None
    finally:
        if conn is not None:
            conn.close()


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed texts via the configured model (disk-cached); None when
    embedding is unavailable (no model set or API failure)."""
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
            resp = client.embeddings.create(model=model, input=[texts[i] for i in missing])
            for i, item in zip(missing, resp.data, strict=False):
                vec = list(item.embedding)
                cached[i] = vec
                cache.put("embed", {"model": model, "text": texts[i]}, vec)
        except Exception:  # noqa: BLE001
            return None
    for c in cached:
        if c is None:
            return None
    return cast("list[list[float]]", cached)


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a)) or 1e-9
    db = math.sqrt(sum(y * y for y in b)) or 1e-9
    return num / (da * db)


class EmbeddingRetriever:
    """Cosine similarity via an OpenAI-compatible embeddings endpoint.

    Embeddings are disk-cached by text hash. Any failure (no model configured,
    API error, dimension mismatch) silently degrades to BM25 — the run's
    correctness never depends on this API being up.
    """

    def __init__(self) -> None:
        self._fallback = BM25Retriever()

    def rank(self, chunks: Sequence[Chunk], query: str, k: int) -> list[Chunk]:
        vecs = embed_texts([c.text for c in chunks] + [query])
        if vecs is None:
            return self._fallback.rank(chunks, query, k)
        qv = vecs[-1]
        order = sorted(
            range(len(chunks)), key=lambda i: cosine(vecs[i], qv), reverse=True
        )
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
