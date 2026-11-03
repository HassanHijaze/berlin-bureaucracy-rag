"""Embedding calls to OpenAI and the on-disk cache of chunk vectors."""

import hashlib
import logging
from pathlib import Path

import numpy as np

from ..config import CACHE_DIR, EMBED_BATCH_SIZE, EMBED_MODEL, FINGERPRINT_LENGTH
from ..openai_client import get_client

logger = logging.getLogger(__name__)

CHUNK_SEPARATOR = b"\0"


def embed_batch(texts: list[str], size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Embed a list of texts, sending them to OpenAI in batches.

    Args:
        texts: The texts to embed.
        size: Number of texts per API request.

    Returns:
        One embedding per text, as rows of a 2-D array.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    output = []

    for start in range(0, len(texts), size):
        response = get_client().embeddings.create(
            model=EMBED_MODEL,
            input=texts[start : start + size],
        )
        output.extend(item.embedding for item in response.data)

    return np.array(output)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query.

    Args:
        text: The question to embed.

    Returns:
        The query embedding as a 1-D array.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    response = get_client().embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(response.data[0].embedding)


def corpus_fingerprint(chunks: list[dict]) -> str:
    """Hash the embedding model and every chunk text into a short cache key.

    Any edit to the corpus, chunking, or embedding model changes the key,
    so stale vectors are never reused for changed text.

    Args:
        chunks: Corpus chunks, each with a ``text`` field.

    Returns:
        The first ``FINGERPRINT_LENGTH`` hex characters of a SHA-256 digest.
    """
    digest = hashlib.sha256(EMBED_MODEL.encode("utf-8"))
    for chunk in chunks:
        digest.update(CHUNK_SEPARATOR)
        digest.update(chunk["text"].encode("utf-8"))
    return digest.hexdigest()[:FINGERPRINT_LENGTH]


def cached_vectors(chunks: list[dict], cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Load chunk embeddings from the cache folder, or create and save them.

    Args:
        chunks: Corpus chunks, each with a ``text`` field.
        cache_dir: Folder holding ``vectors_<fingerprint>.npy`` files.

    Returns:
        One embedding per chunk, in chunk order.

    Raises:
        RuntimeError: If the cache is missing and no OpenAI API key is configured.
    """
    path = cache_dir / f"vectors_{corpus_fingerprint(chunks)}.npy"

    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(chunks):
            logger.info("Using cached vectors: %s", path)
            return vectors
        logger.warning("Cached vector count does not match the corpus. Rebuilding.")

    logger.info("Creating embeddings for %d chunks", len(chunks))
    vectors = embed_batch([chunk["text"] for chunk in chunks])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
