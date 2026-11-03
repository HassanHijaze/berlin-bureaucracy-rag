"""Answer generation: retrieve chunks and ask the language model for a cited answer."""

import logging

import numpy as np

from ..config import (
    BM25_EXPANDED_WEIGHT,
    BM25_ORIGINAL_WEIGHT,
    BM25_WEIGHT,
    CHAT_MODEL,
    MAX_COMPLETION_TOKENS_ANSWER,
    MAX_PER_SOURCE,
    SEMANTIC_WEIGHT,
    TOP_K,
)
from ..corpus.loading import build_corpus
from ..openai_client import get_client
from ..retrieval.embeddings import cached_vectors
from ..retrieval.hybrid import Hit, Retriever
from .prompts import PROMPT

logger = logging.getLogger(__name__)


def build_context(hits: list[Hit]) -> str:
    """Format retrieved chunks as the SOURCES section of the answer prompt.

    Args:
        hits: Search results, best first.

    Returns:
        The chunks numbered from 1, each headed by its source filename.
    """
    return "\n".join(
        f"\n--- CHUNK {index + 1} [{chunk['source']}] ---\n{chunk['text']}\n"
        for index, (chunk, _, _, _) in enumerate(hits)
    )


class AnswerGenerator:
    """Answers questions from retrieved sources, with citations or a refusal."""

    def __init__(self, retriever: Retriever) -> None:
        """Use the given retriever to find the sources for every answer.

        Args:
            retriever: The hybrid search over the corpus.
        """
        self.retriever = retriever

    def ask(self, question: str, k: int = TOP_K) -> tuple[str, list[Hit]]:
        """Retrieve chunks and generate a cited answer.

        Args:
            question: The user's question, in English or German.
            k: Number of chunks to give the model as sources.

        Returns:
            The model's answer (with inline citations, or the refusal
            sentence) and the hits it was given.

        Raises:
            RuntimeError: If no OpenAI API key is configured.
        """
        hits = self.retriever.search(question, k=k)

        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS_ANSWER,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(
                        context=build_context(hits), question=question
                    ),
                }
            ],
        )

        return response.choices[0].message.content, hits


def load_answer_generator(verbose: bool = False) -> AnswerGenerator:
    """Load the corpus and cached vectors, and wire up retrieval and generation.

    Args:
        verbose: Log corpus statistics and retrieval settings.

    Returns:
        A ready-to-use AnswerGenerator.

    Raises:
        RuntimeError: If the vector cache is missing and no OpenAI API key
            is configured to rebuild it.
    """
    documents, chunks = build_corpus()
    vectors = cached_vectors(chunks)

    if verbose:
        logger.info("%s", describe_index(documents, chunks, vectors))

    return AnswerGenerator(Retriever(chunks, vectors))


def describe_index(
    documents: list[dict], chunks: list[dict], vectors: np.ndarray
) -> str:
    """Summarise the corpus statistics and retrieval settings.

    Args:
        documents: The loaded documents.
        chunks: The chunks built from them.
        vectors: The chunk embeddings.

    Returns:
        A multi-line description for the startup log.
    """
    sizes = [len(chunk["text"]) for chunk in chunks]

    lines = [
        f"{len(documents)} documents -> {len(chunks)} chunks",
        f"embedding index: {vectors.shape}",
        f"chunk size: min={min(sizes)}, "
        f"median={int(np.median(sizes))}, max={max(sizes)}",
        "RETRIEVAL SETTINGS",
        f"semantic weight: {SEMANTIC_WEIGHT:.0%}",
        f"BM25 weight: {BM25_WEIGHT:.0%}",
        f"BM25 original weight: {BM25_ORIGINAL_WEIGHT:.0%}",
        f"BM25 expanded weight: {BM25_EXPANDED_WEIGHT:.0%}",
        f"K: {TOP_K}",
        f"MAX_PER_SOURCE: {MAX_PER_SOURCE}",
    ]
    return "\n".join(lines)
