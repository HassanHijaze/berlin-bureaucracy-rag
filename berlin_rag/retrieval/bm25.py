"""Okapi BM25 keyword search over corpus chunks, implemented from scratch."""

import re

import numpy as np

from ..config import BM25_B, BM25_K1

TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into word tokens.

    Args:
        text: Any text, in English or German.

    Returns:
        The word tokens in order. Punctuation and symbols such as § are dropped.
    """
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Keyword index that scores every chunk against a query with Okapi BM25.

    The index is built once from the chunk texts; ``score()`` can then be
    called for any number of queries.
    """

    def __init__(
        self, chunks: list[dict], k1: float = BM25_K1, b: float = BM25_B
    ) -> None:
        """Build the index.

        Args:
            chunks: Corpus chunks, each with a ``text`` field.
            k1: Term-frequency saturation. Higher values reward repeated terms more.
            b: Length normalisation, from 0 (none) to 1 (full).
        """
        self._k1 = k1
        self._b = b
        self._tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self._counts = []
        document_frequency = {}

        for document in self._tokens:
            counts = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            self._counts.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        n = len(self._tokens)
        self._idf = {
            token: np.log(1 + (n - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

        lengths = [len(document) for document in self._tokens]
        self._avg_len = sum(lengths) / n if n else 1

    def score(self, query: str) -> np.ndarray:
        """Score every chunk against a query.

        Args:
            query: The search text, e.g. the question or its German expansion.

        Returns:
            One BM25 score per chunk, in chunk order. Chunks sharing no word
            with the query score 0.
        """
        query_tokens = tokenize(query)
        scores = np.zeros(len(self._tokens))

        for i, counts in enumerate(self._counts):
            if not counts:
                continue

            document_length = len(self._tokens[i])
            score = 0.0

            for token in query_tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue

                idf = self._idf.get(token, 0)
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * document_length / self._avg_len
                )
                score += idf * frequency * (self._k1 + 1) / denominator

            scores[i] = score

        return scores
