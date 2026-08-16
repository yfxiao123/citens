"""Source-text store for citation grounding.

Each paper contributes one or more chunks of "ground text" that a claim is
verified against:

* if an open-access full text is available (PDF -> MarkItDown), we store several
  FULLTEXT chunks and the verifier retrieves the ones most relevant to each
  claim (cheap keyword-overlap RAG);
* otherwise we fall back to a single ABSTRACT chunk.

Keeping the abstraction uniform lets callers ignore whether a paper has full
text or only an abstract.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from citens.models import Chunk, ChunkKind, Paper


class ChunkStore:
    """Maps paper_id -> list of ground-text chunks."""

    def __init__(self) -> None:
        self._by_paper: dict[str, list[Chunk]] = {}

    def build_from(
        self,
        papers: Sequence[Paper],
        *,
        fetch_full: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> None:
        from citens.grounding.fulltext import chunk_text, fetch_fulltext

        n_full = 0
        for i, paper in enumerate(papers):
            if on_progress:
                on_progress(i + 1, len(papers), paper.title[:40])
            if self.has(paper.id):
                # already grounded in a previous compose round (the reflect
                # loop re-runs build over the whole augmented set) — keep the
                # existing chunks; re-fetching and re-parsing PDFs per round
                # was pure repeated work
                if any(c.kind == ChunkKind.FULLTEXT for c in self._by_paper[paper.id]):
                    n_full += 1
                continue
            chunks: list[Chunk] = []
            if fetch_full:
                try:
                    full = fetch_fulltext(paper)
                except Exception as e:  # noqa: BLE001
                    print(f"    fulltext fetch failed for {paper.id}: {e}")
                    full = None
                if full:
                    n_full += 1
                    for j, piece in enumerate(chunk_text(full)):
                        chunks.append(
                            Chunk(
                                paper_id=paper.id,
                                chunk_id=f"{paper.id}-ft-{j}",
                                text=piece,
                                kind=ChunkKind.FULLTEXT,
                            )
                        )
            if not chunks and paper.abstract:
                chunks.append(
                    Chunk(
                        paper_id=paper.id,
                        chunk_id=f"{paper.id}-abs",
                        text=paper.abstract,
                        kind=ChunkKind.ABSTRACT,
                    )
                )
            if chunks:
                self._by_paper[paper.id] = chunks
        if fetch_full:
            print(f"  [ground] {n_full}/{len(papers)} 篇获取到全文")

    def has(self, paper_id: str) -> bool:
        return bool(self._by_paper.get(paper_id))

    def chunks_for(self, paper_id: str) -> list[Chunk]:
        return self._by_paper.get(paper_id, [])

    def ground_text(self, paper_id: str) -> str:
        """Single best-effort ground string (abstract if present, else first chunk)."""
        for c in self._by_paper.get(paper_id, []):
            if c.kind == ChunkKind.ABSTRACT:
                return c.text
        chunks = self._by_paper.get(paper_id, [])
        return chunks[0].text if chunks else ""

    def retrieve(self, paper_id: str, query: str, k: int = 4) -> list[Chunk]:
        """Top-k chunks relevant to `query`; always leads with the abstract
        chunk when present (ranking itself is the pluggable retriever — see
        citens.grounding.retrieval)."""
        chunks = self._by_paper.get(paper_id, [])
        if not chunks:
            return []
        from citens.grounding.retrieval import get_retriever

        ranked = get_retriever().rank(chunks, query, k) if len(chunks) > 1 else chunks[:k]
        result: list[Chunk] = []
        abs_chunk = next((c for c in chunks if c.kind == ChunkKind.ABSTRACT), None)
        if abs_chunk:
            result.append(abs_chunk)
        for c in ranked:
            if c not in result:
                result.append(c)
            if len(result) >= k:
                break
        return result
