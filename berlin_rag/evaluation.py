"""Evaluation harness: score retrieval and refusal on a question set.

This module computes results and builds report text. Printing happens in
the command line (``python -m berlin_rag.cli evaluate``).
"""

import random
from collections.abc import Iterator

from .generation.answer import AnswerGenerator
from .generation.prompts import is_refusal

RULE = "=" * 80
THIN_RULE = "-" * 80
NO_COVERAGE = "none"
DISCLAIMER = (
    "This evaluation measures retrieval and refusal behavior. "
    "It does NOT automatically determine whether an answer is legally correct."
)


def parse_question(question: dict | str, number: int) -> dict:
    """Read one evaluation entry, which may be a dict or a plain string.

    Args:
        question: An entry from the question file.
        number: Its 1-based position, used as a fallback id.

    Returns:
        A dict with ``id``, ``text``, ``sources``, ``coverage`` and ``expected``.
    """
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


def select_questions(
    questions: dict | list, ids: list[str] | None = None
) -> list[dict | str]:
    """Get the question list from a question file, optionally filtered by id.

    Args:
        questions: The parsed file: a list, or a dict with a ``questions`` list.
        ids: If given, keep only dict entries with one of these ids.

    Returns:
        The questions to run, in file order.
    """
    if isinstance(questions, dict):
        questions = questions.get("questions", [])

    if ids:
        questions = [q for q in questions if isinstance(q, dict) and q.get("id") in ids]

    return questions


def sample_questions(
    questions: list[dict | str], count: int, seed: int
) -> list[dict | str]:
    """Pick a random subset of questions, reproducible through the seed.

    Args:
        questions: The questions to sample from.
        count: How many to pick. If larger than the list, all are returned.
        seed: Random seed; the same seed always picks the same questions.

    Returns:
        The picked questions, in their original file order.
    """
    count = min(count, len(questions))
    picked = sorted(random.Random(seed).sample(range(len(questions)), count))
    return [questions[index] for index in picked]


def evaluate_question(answer_generator: AnswerGenerator, entry: dict) -> dict:
    """Answer one question and check refusal and retrieval.

    Args:
        answer_generator: Produces the answer and the retrieved hits.
        entry: A parsed question from ``parse_question``.

    Returns:
        A result with ``id``, ``coverage``, ``refused``, ``retrieved_correct``,
        ``answer`` (stripped) and ``hits``.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    answer, hits = answer_generator.ask(entry["text"])

    retrieved_sources = [chunk["source"] for chunk, _, _, _ in hits]
    found = any(source in entry["sources"] for source in retrieved_sources)

    return {
        "id": entry["id"],
        "coverage": entry["coverage"],
        "refused": is_refusal(answer),
        "retrieved_correct": found,
        "answer": answer.strip(),
        "hits": hits,
    }


def run_review(
    answer_generator: AnswerGenerator,
    questions: dict | list,
    ids: list[str] | None = None,
) -> Iterator[tuple[int, int, dict, dict]]:
    """Evaluate questions one by one, so callers can report progress.

    Args:
        answer_generator: Produces the answers.
        questions: The parsed question file.
        ids: If given, run only the questions with these ids.

    Yields:
        ``(number, total, entry, result)`` for each question, in order.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    selected = select_questions(questions, ids)

    for number, question in enumerate(selected, 1):
        entry = parse_question(question, number)
        yield number, len(selected), entry, evaluate_question(answer_generator, entry)


def format_result(number: int, total: int, entry: dict, result: dict) -> str:
    """Build the report block for one evaluation question.

    Args:
        number: 1-based position of the question.
        total: Number of questions in the run.
        entry: The parsed question.
        result: Its result from ``evaluate_question``.

    Returns:
        The block, ending with a blank line.
    """
    lines = [
        RULE,
        f"[{number}/{total}] {entry['id']}   coverage: {entry['coverage']}",
        f"Q: {entry['text']}",
        THIN_RULE,
        result["answer"],
        THIN_RULE,
    ]

    if entry["expected"]:
        lines.append(f"EXPECTED: {entry['expected']}")

    lines.append(f"RETRIEVED TOP-{len(result['hits'])}:")
    for rank, (chunk, final, semantic, bm25) in enumerate(result["hits"], 1):
        lines.append(
            f"  {rank:2d}. final={final:.4f} semantic={semantic:.4f} "
            f"bm25={bm25:.4f} | {chunk['source']} (chunk {chunk['index']})"
        )

    if entry["sources"]:
        lines.append(
            f"CORRECT SOURCE FOUND: {'yes' if result['retrieved_correct'] else 'NO'} "
            f"(wanted: {', '.join(entry['sources'])})"
        )

    if result["refused"]:
        lines.append("FLAGS: REFUSED")

    lines.append("")
    return "\n".join(lines)


def format_rate(label: str, count: int, total: int) -> str:
    """Format one summary line as ``label: count/total (percent%)``.

    Args:
        label: What is being counted.
        count: How many met the condition.
        total: How many were checked.

    Returns:
        The indented summary line.
    """
    return f"  {label}: {count}/{total} ({100 * count / total:.1f}%)"


def format_summary(results: list[dict]) -> str:
    """Build the summary of retrieval and refusal rates for a run.

    Questions with coverage ``none`` should be refused; all others should
    be answered from the right source.

    Args:
        results: Results from ``evaluate_question``.

    Returns:
        The summary report.
    """
    should_answer = [r for r in results if r["coverage"] != NO_COVERAGE]
    should_refuse = [r for r in results if r["coverage"] == NO_COVERAGE]

    lines = [RULE, "SUMMARY EVALUATION", RULE, "Retrieval:"]

    if should_answer:
        total = len(should_answer)
        retrieved = sum(r["retrieved_correct"] for r in should_answer)
        answered = sum(not r["refused"] for r in should_answer)
        lines.append(format_rate("right source retrieved", retrieved, total))
        lines.append(format_rate("produced an answer", answered, total))

    if should_refuse:
        total = len(should_refuse)
        refused = sum(r["refused"] for r in should_refuse)
        lines.append(format_rate("correctly refused", refused, total))

    lines.extend(["", "NOTE:", DISCLAIMER])
    return "\n".join(lines)
