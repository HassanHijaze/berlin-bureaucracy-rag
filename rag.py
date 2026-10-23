"""
Berlin Bureaucracy RAG

Hybrid retrieval over German statutes and Berlin administrative sources,
with cited answers and refusal when the sources do not cover a question.

Retrieval:
    - 70% semantic similarity, 30% BM25
    - BM25 combines the original question and a German query expansion
    - Top-K = 12, at most 2 chunks per source

Usage:
    python rag.py ask "Your question here"   # answer one question
    python rag.py                            # run the evaluation set
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from chunking import (
    MAX_CHUNK,
    MIN_CHUNK,
    chunk_berlin,
    chunk_legal,
    merge_small,
    split_hard,
)
from evaluation import run_review
from prompts import EXPANSION_PROMPT, PROMPT


# ================================================================
# CONFIG
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
FOLDER = BASE_DIR / "Documents"
ENV_FILE = BASE_DIR / "API.env"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6"

K = 12
MAX_PER_SOURCE = 2

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
BM25_ORIGINAL_WEIGHT = 0.60
BM25_EXPANDED_WEIGHT = 0.40

MAX_COMPLETION_TOKENS_EXPANSION = 200
MAX_COMPLETION_TOKENS_ANSWER = 800

LEGAL_DIRECTORIES = [
    "bmg_sections",
    "aufenthg_sections",
    "sgb_sections",
    "aufenthv_sections",
    "vab_sections",
]


# ================================================================
# OPENAI CLIENT
# ================================================================

load_dotenv(ENV_FILE)

_client = None


def get_client():
    """Create the OpenAI client on first use, so offline code runs without a key."""
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Put it in your API.env file."
            )
        _client = OpenAI(api_key=api_key)

    return _client


# ================================================================
# CORPUS
# ================================================================

def load_documents():
    """Load statute .txt files and Berlin .md pages from the Documents folder."""
    documents = []

    for directory_name in LEGAL_DIRECTORIES:
        directory = FOLDER / directory_name
        if not directory.exists():
            print(f"WARNING: directory not found: {directory}")
            continue
        for path in sorted(directory.glob("*.txt")):
            documents.append(
                {"name": path.name, "text": path.read_text(encoding="utf-8")}
            )

    for path in sorted(FOLDER.glob("*.md")):
        documents.append(
            {"name": path.name, "text": path.read_text(encoding="utf-8")}
        )

    return documents


def build_corpus():
    """Load all documents and split them into source-aware chunks."""
    documents = load_documents()
    chunks = []

    for document in documents:
        name = document["name"]
        text = document["text"]

        if name.endswith(".md"):
            parts = chunk_berlin(text)
        else:
            parts = chunk_legal(text)

        for index, part in enumerate(parts):
            chunks.append(
                {"text": f"[{name}]\n{part}", "source": name, "index": index}
            )

    return documents, merge_small(chunks)


# ================================================================
# BM25
# ================================================================

def tokenize(text):
    """Lowercase text and split it into word tokens."""
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 keyword scoring, implemented from scratch."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(chunk["text"]) for chunk in chunks]
        self.counts = []
        document_frequency = {}

        for document in self.tokens:
            counts = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            self.counts.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        n = len(self.tokens)
        self.idf = {
            token: np.log(1 + (n - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

        lengths = [len(document) for document in self.tokens]
        self.avg_len = sum(lengths) / n if n else 1

    def score(self, question):
        """Return one BM25 score per chunk for the given question."""
        query_tokens = tokenize(question)
        scores = np.zeros(len(self.tokens))

        for i, counts in enumerate(self.counts):
            if not counts:
                continue

            document_length = len(self.tokens[i])
            score = 0.0

            for token in query_tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue

                idf = self.idf.get(token, 0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.avg_len
                )
                score += idf * frequency * (self.k1 + 1) / denominator

            scores[i] = score

        return scores


# ================================================================
# EMBEDDINGS
# ================================================================

def embed_batch(texts, size=100):
    """Embed a list of texts in batches."""
    output = []

    for start in range(0, len(texts), size):
        response = get_client().embeddings.create(
            model=EMBED_MODEL,
            input=texts[start:start + size],
        )
        output.extend(item.embedding for item in response.data)

    return np.array(output)


def embed_query(text):
    """Embed a single query."""
    response = get_client().embeddings.create(model=EMBED_MODEL, input=[text])
    return np.array(response.data[0].embedding)


def cached_vectors(chunks):
    """Load chunk embeddings from cache, or create and save them."""
    path = FOLDER / f"vectors_{len(chunks)}.npy"

    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(chunks):
            print(f"Using cached vectors: {path}")
            return vectors
        print("Cached vector count does not match current corpus. Rebuilding.")

    print(f"Creating embeddings for {len(chunks)} chunks...")
    vectors = embed_batch([chunk["text"] for chunk in chunks])
    np.save(path, vectors)
    return vectors


# ================================================================
# SYSTEM
# ================================================================

class System:
    """The full RAG system: corpus, embeddings, BM25, search, and answering."""

    def __init__(self, verbose=True):
        self.documents, self.chunks = build_corpus()
        self.vectors = cached_vectors(self.chunks)
        self.bm25 = BM25(self.chunks)

        if verbose:
            self._print_summary()

    def _print_summary(self):
        """Print corpus statistics and retrieval settings."""
        sizes = [len(chunk["text"]) for chunk in self.chunks]

        print(f"{len(self.documents)} documents -> {len(self.chunks)} chunks")
        print(f"embedding index: {self.vectors.shape}")
        print(
            f"chunk size: min={min(sizes)}, "
            f"median={int(np.median(sizes))}, max={max(sizes)}"
        )
        print()
        print("RETRIEVAL SETTINGS")
        print(f"semantic weight: {SEMANTIC_WEIGHT:.0%}")
        print(f"BM25 weight: {BM25_WEIGHT:.0%}")
        print(f"BM25 original weight: {BM25_ORIGINAL_WEIGHT:.0%}")
        print(f"BM25 expanded weight: {BM25_EXPANDED_WEIGHT:.0%}")
        print(f"K: {K}")
        print(f"MAX_PER_SOURCE: {MAX_PER_SOURCE}")
        print()

    # ------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------

    def expand_query_de(self, question):
        """Rewrite a question into German legal keywords for BM25."""
        try:
            response = get_client().chat.completions.create(
                model=CHAT_MODEL,
                max_completion_tokens=MAX_COMPLETION_TOKENS_EXPANSION,
                messages=[
                    {
                        "role": "user",
                        "content": EXPANSION_PROMPT.format(question=question),
                    }
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            print(f"WARNING: query expansion failed: {exc}")
            return ""

    # ------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------

    @staticmethod
    def minmax_normalize(scores):
        """Scale scores to the range 0-1. Returns zeros if all scores are equal."""
        scores = np.asarray(scores, dtype=float)
        minimum, maximum = scores.min(), scores.max()

        if maximum - minimum < 1e-12:
            return np.zeros_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def semantic_scores(self, question):
        """Cosine similarity between the question and every chunk."""
        query_vector = embed_query(question)
        vector_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)

        return self.vectors @ query_vector / (vector_norms * query_norm + 1e-12)

    def bm25_scores(self, question):
        """BM25 on the original question, combined with the German expansion."""
        bm25_original = self.bm25.score(question)

        de_terms = self.expand_query_de(question)
        if de_terms:
            bm25_expanded = self.bm25.score(de_terms)
        else:
            bm25_expanded = np.zeros(len(self.chunks))

        return (
            BM25_ORIGINAL_WEIGHT * self.minmax_normalize(bm25_original)
            + BM25_EXPANDED_WEIGHT * self.minmax_normalize(bm25_expanded)
        )

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------

    def search(self, question, k=K, max_per_source=MAX_PER_SOURCE):
        """
        Rank chunks by weighted semantic + BM25 score.

        Returns a list of (chunk, final_score, semantic_score, bm25_score).
        """
        semantic = self.semantic_scores(question)
        bm25_combined = self.bm25_scores(question)

        final_scores = (
            SEMANTIC_WEIGHT * self.minmax_normalize(semantic)
            + BM25_WEIGHT * bm25_combined
        )

        results = []
        source_counts = {}

        for index in np.argsort(final_scores)[::-1]:
            source = self.chunks[index]["source"]
            count = source_counts.get(source, 0)

            if count >= max_per_source:
                continue

            source_counts[source] = count + 1
            results.append(
                (
                    self.chunks[index],
                    float(final_scores[index]),
                    float(semantic[index]),
                    float(bm25_combined[index]),
                )
            )

            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------

    def ask(self, question, k=K):
        """Retrieve chunks and generate a cited answer. Returns (answer, hits)."""
        hits = self.search(question, k=k)

        context = "\n".join(
            f"\n--- CHUNK {index + 1} [{chunk['source']}] ---\n{chunk['text']}\n"
            for index, (chunk, _, _, _) in enumerate(hits)
        )

        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS_ANSWER,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(context=context, question=question),
                }
            ],
        )

        return response.choices[0].message.content, hits


# ================================================================
# COMMAND LINE
# ================================================================

def main(args):
    """Answer one question (`ask "..."`) or run the full evaluation set."""
    if args and args[0] == "ask":
        question = " ".join(args[1:])
        if not question:
            print("Usage:")
            print('python rag.py ask "Your question here"')
            sys.exit(1)

        system = System(verbose=False)
        answer, _ = system.ask(question)

        print()
        print("ANSWER")
        print("=" * 80)
        print(answer.strip())
        return

    json_path = FOLDER / "100.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    system = System(verbose=False)
    run_review(system, data)


if __name__ == "__main__":
    main(sys.argv[1:])