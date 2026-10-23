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


# ================================================================
# CONFIG
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
FOLDER = BASE_DIR / "Documents"
ENV_FILE = BASE_DIR / "API.env"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6"

MAX_CHUNK = 750
MIN_CHUNK = 150

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

BERLIN_HEADINGS = [
    "Verfahrensablauf",
    "Voraussetzungen",
    "Erforderliche Unterlagen",
    "Formulare",
    "Gebühren",
    "Rechtsgrundlagen",
    "Durchschnittliche Bearbeitungszeit",
    "Weiterführende Informationen",
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
# CHUNKING
# ================================================================

def split_hard(text, max_size=MAX_CHUNK, overlap=100):
    """Split text into fixed-size pieces that overlap by `overlap` characters."""
    if len(text) <= max_size:
        return [text]

    pieces = []
    step = max_size - overlap
    start = 0

    while start < len(text):
        pieces.append(text[start:start + max_size])
        start += step

    return pieces


def chunk_legal(text, max_size=MAX_CHUNK):
    """Split a statute at paragraph markers like (1), (2), keeping the header."""
    if len(text) <= max_size:
        return [text]

    lines = text.split("\n")
    header = lines[0] if lines else ""

    pieces = []
    current = ""

    for part in re.split(r"\n(?=\(\d+\))", text):
        if current and len(current) + len(part) > max_size:
            pieces.append(f"[{header}]\n{current}" if pieces else current)
            current = part
        else:
            current += ("\n" if current else "") + part

    if current:
        pieces.append(f"[{header}]\n{current}" if pieces else current)

    final_pieces = []
    for piece in pieces:
        if len(piece) <= max_size:
            final_pieces.append(piece)
        else:
            final_pieces.extend(split_hard(piece, max_size=max_size))

    return final_pieces


def chunk_berlin(text, max_size=MAX_CHUNK):
    """Split a Berlin administrative page at its standard section headings."""
    pattern = "|".join(re.escape(heading) for heading in BERLIN_HEADINGS)

    parts = [
        part.strip()
        for part in re.split(rf"\n(?={pattern})", text)
        if part.strip()
    ]
    if not parts:
        parts = [text]

    output = []
    for part in parts:
        if len(part) <= max_size:
            output.append(part)
        else:
            output.extend(split_hard(part, max_size=max_size))

    return output


def merge_small(chunks, min_size=MIN_CHUNK):
    """Merge tiny chunks into the previous chunk from the same source."""
    output = []

    for chunk in chunks:
        can_merge = (
            output
            and len(chunk["text"]) < min_size
            and output[-1]["source"] == chunk["source"]
            and len(output[-1]["text"]) + 1 + len(chunk["text"]) <= MAX_CHUNK
        )
        if can_merge:
            output[-1]["text"] += "\n" + chunk["text"]
        else:
            output.append(dict(chunk))

    # Renumber chunk indices per source after merging.
    counts = {}
    for chunk in output:
        index = counts.get(chunk["source"], 0)
        chunk["index"] = index
        counts[chunk["source"]] = index + 1

    return output


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
# PROMPTS
# ================================================================

EXPANSION_PROMPT = (
    "Map the following user question to German administrative and legal "
    "search keywords.\n\n"
    "Include relevant German legal terms, statutory sections, and "
    "administrative terminology when appropriate.\n\n"
    "Examples:\n"
    "- cost/fee -> Gebühren kostenfrei\n"
    "- lost paper/proof -> Meldebescheinigung Meldebestätigung\n"
    "- fine/penalty -> Ordnungswidrigkeit Bußgeld § 54\n"
    "- hotel/hostel -> Beherbergungsstätte § 29\n"
    "- move/change address -> Anmeldung Ummeldung § 17 § 21\n\n"
    "Question: {question}\n\n"
    "Output ONLY space-separated German keywords."
)

PROMPT = """
YOU MUST WRITE YOUR ANSWER IN THE QUESTIONS LANGUAGE.

You are an expert administrative and regulatory intelligence assistant.

Answer the user's question accurately and clearly based ONLY on the
provided SOURCES.

IMPORTANT:

1. SOURCE-BASED ANSWERING

Use only information supported by the provided SOURCES.

Do not introduce outside legal knowledge.

Do not invent missing facts.

Do not guess.


2. METADATA AND STRUCTURED DATA

Treat all information in the context as factual evidence, including:

- document metadata
- headings
- key-value fields
- lists
- tables
- costs
- fees
- conditions
- legal provisions


3. SEMANTIC BRIDGING

Users may use everyday language rather than legal terminology.

Map concepts when the underlying administrative/legal category is clearly
present in the sources.

Examples:

hotel / hostel
-> accommodation / lodging establishment

paperwork
-> required documents

lost proof
-> certificate / confirmation

move / change address
-> Anmeldung / Ummeldung


4. LEGAL MODALITY

Preserve the distinction between:

- must / shall
- may
- can
- cannot
- prohibited
- conditional requirements

Do not turn a discretionary rule into a mandatory rule.

Do not turn a conditional rule into an unconditional rule.


5. REFUSAL

Output EXACTLY:

The sources I have do not cover this.

ONLY when the core policy, legal rule, requirement, or numerical figure
needed to answer the question is absent from the provided SOURCES.

Do not refuse simply because the user's wording differs from the source.


6. WHOSE SITUATION IS THIS?

Before using a rule, identify who or what the rule is written about, and
check that the person asking falls inside that group.

Ask yourself, for each rule you are about to apply:

- Who does this rule govern? A rule about people already registered in
  Germany does not govern someone arriving from abroad. A rule about
  online registration does not govern an in-person appointment. A rule
  about family members insured through a relative does not govern a
  student insured in their own right.

- What question does this text answer? A source explaining WHO may issue
  a document does not answer WHETHER a particular person qualifies. A
  source explaining WHAT a permit is for does not answer WHETHER a change
  of course requires a new one.

If the source is about the right topic but does not address the situation
in the question, say so:

"The sources state [the rule]. They do not address [the specific
situation you asked about]."

That is a complete and useful answer. Do not close the gap yourself.


7. CITATIONS

Every factual claim must cite the exact source filename.

Example:

International students may work up to 140 full working days per year
[aufenthg_en_16b.txt].


8. ANSWER STYLE

Give a direct answer first.

Then provide a short explanation if needed.

Do not add unsupported advice.

SOURCES:

{context}

QUESTION:

{question}

IN THE QUESTIONS LANGUAGE.
"""


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
# EVALUATION
# ================================================================

def parse_question(question, number):
    """Read one evaluation entry, which may be a dict or a plain string."""
    if isinstance(question, dict):
        return {
            "id": question.get("id", f"Q{number}"),
            "text": question.get("question", ""),
            "sources": question.get("source_files", []),
            "coverage": question.get("coverage", "-"),
            "expected": question.get("expected"),
        }

    return {
        "id": f"Q{number}",
        "text": str(question),
        "sources": [],
        "coverage": "-",
        "expected": None,
    }


def print_result(number, total, entry, answer, hits, refused, found):
    """Print the answer and retrieved chunks for one evaluation question."""
    print("=" * 80)
    print(f"[{number}/{total}] {entry['id']}   coverage: {entry['coverage']}")
    print(f"Q: {entry['text']}")
    print("-" * 80)
    print(answer.strip())
    print("-" * 80)

    if entry["expected"]:
        print(f"EXPECTED: {entry['expected']}")

    print("RETRIEVED TOP-12:")
    for rank, (chunk, final, semantic, bm25) in enumerate(hits, 1):
        print(
            f"  {rank:2d}. final={final:.4f} semantic={semantic:.4f} "
            f"bm25={bm25:.4f} | {chunk['source']} (chunk {chunk['index']})"
        )

    if entry["sources"]:
        print(
            f"CORRECT SOURCE FOUND: {'yes' if found else 'NO'} "
            f"(wanted: {', '.join(entry['sources'])})"
        )

    if refused:
        print("FLAGS: REFUSED")

    print()


def print_summary(results):
    """Print retrieval and refusal rates for the evaluation run."""
    should_answer = [r for r in results if r["coverage"] != "none"]
    should_refuse = [r for r in results if r["coverage"] == "none"]

    print("=" * 80)
    print("SUMMARY EVALUATION")
    print("=" * 80)
    print("Retrieval:")

    if should_answer:
        total = len(should_answer)
        retrieved = sum(r["retrieved_correct"] for r in should_answer)
        answered = sum(not r["refused"] for r in should_answer)
        print(f"  right source retrieved: {retrieved}/{total} ({100 * retrieved / total:.1f}%)")
        print(f"  produced an answer: {answered}/{total} ({100 * answered / total:.1f}%)")

    if should_refuse:
        total = len(should_refuse)
        refused = sum(r["refused"] for r in should_refuse)
        print(f"  correctly refused: {refused}/{total} ({100 * refused / total:.1f}%)")

    print()
    print("NOTE:")
    print(
        "This evaluation measures retrieval and refusal behavior. "
        "It does NOT automatically determine whether an answer is legally correct."
    )


def run_review(system, questions, ids=None):
    """Run evaluation questions, print each result, and return a summary list."""
    if isinstance(questions, dict):
        questions = questions.get("questions", [])

    if ids:
        questions = [
            q for q in questions if isinstance(q, dict) and q.get("id") in ids
        ]

    results = []

    for number, question in enumerate(questions, 1):
        entry = parse_question(question, number)
        answer, hits = system.ask(entry["text"])

        refused = "do not cover" in answer.lower()
        retrieved_sources = [chunk["source"] for chunk, _, _, _ in hits]
        found = any(source in entry["sources"] for source in retrieved_sources)

        print_result(number, len(questions), entry, answer, hits, refused, found)

        results.append(
            {
                "id": entry["id"],
                "coverage": entry["coverage"],
                "refused": refused,
                "retrieved_correct": found,
                "answer": answer.strip(),
            }
        )

    print_summary(results)
    return results


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