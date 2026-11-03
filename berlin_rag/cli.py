"""Command-line entry point: answer one question or run the evaluation set.

Usage (from the project root):
    python -m berlin_rag.cli ask "Your question here"   # answer one question
    python -m berlin_rag.cli evaluate                   # run the evaluation set
    python -m berlin_rag.cli evaluate --sample 10       # 10 random questions
    python -m berlin_rag.cli evaluate --sample 10 --seed 42   # same 10 again
    python -m berlin_rag.cli                            # same as evaluate

Answers and reports go to stdout; progress and warnings are logged to stderr.
"""

import json
import logging
import random
import sys

from .config import EVAL_FILE, LOG_FORMAT, LOG_LEVEL
from .evaluation import (
    format_result,
    format_summary,
    run_review,
    sample_questions,
    select_questions,
)
from .generation.answer import load_answer_generator

USAGE = (
    "Usage:\n"
    'python -m berlin_rag.cli ask "Your question here"\n'
    "python -m berlin_rag.cli evaluate [--sample N] [--seed S]"
)
EXIT_USAGE_ERROR = 1
MAX_SEED = 10_000


def ask(question: str) -> None:
    """Answer one question and print the answer.

    Args:
        question: The question, in English or German.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    answer, _ = load_answer_generator().ask(question)

    print()
    print("ANSWER")
    print("=" * 80)
    print(answer.strip())


def parse_evaluate_options(options: list[str]) -> tuple[int | None, int | None]:
    """Read the ``--sample N`` and ``--seed S`` options of ``evaluate``.

    Args:
        options: The arguments after ``evaluate``.

    Returns:
        The sample size and the seed; each is None when not given.

    Raises:
        ValueError: If an option is unknown, has no value, or its value is
            not a positive whole number (the seed may also be 0).
    """
    values = {"--sample": None, "--seed": None}

    if len(options) % 2:
        raise ValueError("every option needs a value")

    for name, value in zip(options[::2], options[1::2], strict=True):
        if name not in values or not value.isdigit():
            raise ValueError(f"invalid option: {name} {value}")
        values[name] = int(value)

    if values["--sample"] == 0:
        raise ValueError("--sample must be at least 1")

    return values["--sample"], values["--seed"]


def evaluate(sample: int | None = None, seed: int | None = None) -> None:
    """Run the evaluation set, printing each result as it arrives and a summary.

    Args:
        sample: If given, run only this many randomly picked questions.
        seed: Seed for the random pick; a new one is chosen when not given.

    Raises:
        FileNotFoundError: If the evaluation file is missing.
        RuntimeError: If no OpenAI API key is configured.
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Could not find: {EVAL_FILE}")

    questions = select_questions(json.loads(EVAL_FILE.read_text(encoding="utf-8")))

    if sample is not None:
        if seed is None:
            seed = random.randrange(MAX_SEED)
        total = len(questions)
        questions = sample_questions(questions, sample, seed)
        print(
            f"Random sample: {len(questions)} of {total} questions (seed {seed}). "
            f"Rerun the same questions with: --sample {sample} --seed {seed}"
        )
        print()

    answer_generator = load_answer_generator()

    results = []
    for number, total, entry, result in run_review(answer_generator, questions):
        print(format_result(number, total, entry, result))
        results.append(result)

    print(format_summary(results))


def main(args: list[str]) -> None:
    """Dispatch the command-line arguments.

    Args:
        args: The arguments after the program name.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    if args and args[0] == "ask":
        question = " ".join(args[1:])
        if not question:
            print(USAGE)
            sys.exit(EXIT_USAGE_ERROR)
        ask(question)
        return

    if not args or args[0] == "evaluate":
        try:
            sample, seed = parse_evaluate_options(args[1:])
        except ValueError:
            print(USAGE)
            sys.exit(EXIT_USAGE_ERROR)
        evaluate(sample, seed)
        return

    print(USAGE)
    sys.exit(EXIT_USAGE_ERROR)


if __name__ == "__main__":
    main(sys.argv[1:])
