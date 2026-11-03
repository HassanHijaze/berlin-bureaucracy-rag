"""Read statutes and Berlin pages from the corpus folder and turn them into chunks."""

import logging
from pathlib import Path

from ..config import CORPUS_DIR, LEGAL_DIRECTORIES
from .chunking import chunk_berlin, chunk_legal, merge_small

logger = logging.getLogger(__name__)


def load_documents(
    corpus_dir: Path = CORPUS_DIR,
    legal_directories: list[str] = LEGAL_DIRECTORIES,
) -> list[dict]:
    """Load statute .txt files and Berlin .md pages from the corpus folder.

    Args:
        corpus_dir: Folder with the Berlin .md pages and the statute subfolders.
        legal_directories: Names of the subfolders that hold statute .txt files.

    Returns:
        Documents with ``name`` (the filename) and ``text`` fields: statutes
        first, in folder order, then Berlin pages, each group sorted by name.
        A missing statute folder is skipped with a warning.
    """
    documents = []

    for directory_name in legal_directories:
        directory = corpus_dir / directory_name
        if not directory.exists():
            logger.warning("Directory not found: %s", directory)
            continue
        for path in sorted(directory.glob("*.txt")):
            documents.append(
                {"name": path.name, "text": path.read_text(encoding="utf-8")}
            )

    for path in sorted(corpus_dir.glob("*.md")):
        documents.append({"name": path.name, "text": path.read_text(encoding="utf-8")})

    return documents


def build_corpus(corpus_dir: Path = CORPUS_DIR) -> tuple[list[dict], list[dict]]:
    """Load all documents and split them into source-aware chunks.

    Berlin pages (.md) are split at their section headings, statutes (.txt)
    at their paragraph markers. Each chunk starts with its filename in
    brackets so the model can cite it.

    Args:
        corpus_dir: Folder with the Berlin .md pages and the statute subfolders.

    Returns:
        The loaded documents and the chunks, each chunk with ``text``,
        ``source`` and ``index`` fields.
    """
    documents = load_documents(corpus_dir)
    chunks = []

    for document in documents:
        name = document["name"]
        text = document["text"]

        if name.endswith(".md"):
            parts = chunk_berlin(text)
        else:
            parts = chunk_legal(text)

        for index, part in enumerate(parts):
            chunks.append({"text": f"[{name}]\n{part}", "source": name, "index": index})

    return documents, merge_small(chunks)
