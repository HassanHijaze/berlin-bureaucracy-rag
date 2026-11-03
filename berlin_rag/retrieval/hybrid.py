"""Hybrid search: fuse semantic and BM25 scores and pick the top chunks.

Final score = 70% semantic similarity + 30% BM25, where the BM25 part
itself combines the original question (60%) and its German expansion (40%).
Every score is min-max normalised before weighting.
"""

import numpy as np

from ..config import (
    BM25_EXPANDED_WEIGHT,
    BM25_ORIGINAL_WEIGHT,
    BM25_WEIGHT,
    MAX_PER_SOURCE,
    SEMANTIC_WEIGHT,
    TOP_K,
)
from .bm25 import BM25Index
from .embeddings import embed_query
from .expansion import expand_query

SCORE_EPSILON = 1e-12

Hit = tuple[dict, float, float, float]
"""One search result: (chunk, final_score, semantic_score, bm25_score)."""


def minmax_normalize(scores: np.ndarray | list[float]) -> np.ndarray:
    """Scale scores to the range 0-1.

    Args:
        scores: Raw scores, one per chunk.

    Returns:
        The scaled scores, or all zeros if every score is equal.
    """
    scores = np.asarray(scores, dtype=float)
    minimum, maximum = scores.min(), scores.max()

    if maximum - minimum < SCORE_EPSILON:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def cosine_similarities(vectors: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity between a query and every chunk vector.

    Args:
        vectors: Chunk embeddings, one row per chunk.
        query_vector: The query embedding.

    Returns:
        One similarity per chunk, between -1 and 1.
    """
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)
    return vectors @ query_vector / (vector_norms * query_norm + SCORE_EPSILON)


def select_top_k(
    chunks: list[dict],
    final_scores: np.ndarray,
    semantic_scores: np.ndarray,
    bm25_scores: np.ndarray,
    k: int = TOP_K,
    max_per_source: int = MAX_PER_SOURCE,
) -> list[Hit]:
    """Pick the highest-scoring chunks, with at most a few per source.

    The per-source cap keeps one long document from filling every slot.

    Args:
        chunks: Corpus chunks, each with a ``source`` field.
        final_scores: Fused score per chunk, used for ranking.
        semantic_scores: Semantic score per chunk, reported in each hit.
        bm25_scores: Combined BM25 score per chunk, reported in each hit.
        k: Maximum number of hits.
        max_per_source: Maximum number of hits from one source file.

    Returns:
        Up to ``k`` hits, best first.
    """
    results = []
    source_counts = {}

    for index in np.argsort(final_scores)[::-1]:
        source = chunks[index]["source"]
        count = source_counts.get(source, 0)

        if count >= max_per_source:
            continue

        source_counts[source] = count + 1
        results.append(
            (
                chunks[index],
                float(final_scores[index]),
                float(semantic_scores[index]),
                float(bm25_scores[index]),
            )
        )

        if len(results) >= k:
            break

    return results


class Retriever:
    """Hybrid semantic + keyword search over the corpus chunks."""

    def __init__(self, chunks: list[dict], vectors: np.ndarray) -> None:
        """Build the keyword index next to the chunk embeddings.

        Args:
            chunks: Corpus chunks with ``text`` and ``source`` fields.
            vectors: One embedding per chunk, in the same order.
        """
        self._chunks = chunks
        self._vectors = vectors
        self._bm25_index = BM25Index(chunks)

    @property
    def chunks(self) -> list[dict]:
        """The searchable corpus chunks, in index order."""
        return self._chunks

    @property
    def vectors(self) -> np.ndarray:
        """The chunk embeddings, one row per chunk."""
        return self._vectors

    def search(
        self, question: str, k: int = TOP_K, max_per_source: int = MAX_PER_SOURCE
    ) -> list[Hit]:
        """Rank chunks by weighted semantic + BM25 score.

        Args:
            question: The user's question.
            k: Maximum number of hits.
            max_per_source: Maximum number of hits from one source file.

        Returns:
            Up to ``k`` hits, best first.

        Raises:
            RuntimeError: If no OpenAI API key is configured.
        """
        semantic = self._semantic_scores(question)
        bm25_combined = self._bm25_scores(question)

        final_scores = (
            SEMANTIC_WEIGHT * minmax_normalize(semantic) + BM25_WEIGHT * bm25_combined
        )

        return select_top_k(
            self._chunks, final_scores, semantic, bm25_combined, k, max_per_source
        )

    def _semantic_scores(self, question: str) -> np.ndarray:
        """Cosine similarity between the embedded question and every chunk."""
        return cosine_similarities(self._vectors, embed_query(question))

    def _bm25_scores(self, question: str) -> np.ndarray:
        """BM25 on the original question, combined with the German expansion."""
        bm25_original = self._bm25_index.score(question)

        german_keywords = expand_query(question)
        if german_keywords:
            bm25_expanded = self._bm25_index.score(german_keywords)
        else:
            bm25_expanded = np.zeros(len(self._chunks))

        return BM25_ORIGINAL_WEIGHT * minmax_normalize(
            bm25_original
        ) + BM25_EXPANDED_WEIGHT * minmax_normalize(bm25_expanded)
