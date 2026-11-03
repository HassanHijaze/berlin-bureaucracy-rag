import pytest

from berlin_rag.evaluation import (
    evaluate_question,
    format_result,
    format_summary,
    parse_question,
    run_review,
    sample_questions,
    select_questions,
)


class FakeAnswerGenerator:
    def __init__(self, answer):
        self.answer = answer

    def ask(self, question):
        hits = [({"source": "bmg_en_17.txt", "index": 0}, 0.9, 0.8, 0.7)]
        return self.answer, hits


@pytest.fixture
def entry():
    return {
        "id": "R1",
        "text": "How long do I have to register?",
        "sources": ["bmg_en_17.txt"],
        "coverage": "full",
        "expected": "Two weeks.",
    }


@pytest.mark.parametrize(
    "question, expected_result",
    [
        (
            {"id": "A", "question": "Q?", "source_files": ["a.md"], "coverage": "none"},
            {
                "id": "A",
                "text": "Q?",
                "sources": ["a.md"],
                "coverage": "none",
                "expected": None,
            },
        ),
        (
            "Plain question",
            {
                "id": "Q3",
                "text": "Plain question",
                "sources": [],
                "coverage": "-",
                "expected": None,
            },
        ),
    ],
)
def test_parse_question(question, expected_result):
    assert parse_question(question, 3) == expected_result


@pytest.mark.parametrize(
    "questions, ids, expected_result",
    [
        ({"questions": [{"id": "A"}, {"id": "B"}]}, None, [{"id": "A"}, {"id": "B"}]),
        ([{"id": "A"}, {"id": "B"}, "plain"], ["B"], [{"id": "B"}]),
    ],
)
def test_select_questions(questions, ids, expected_result):
    assert select_questions(questions, ids) == expected_result


@pytest.mark.parametrize(
    "answer, expected_refused",
    [
        ("  Two weeks [bmg_en_17.txt].\n", False),
        ("The sources I have do not cover this.", True),
    ],
)
def test_evaluate_question(entry, answer, expected_refused):
    result = evaluate_question(FakeAnswerGenerator(answer), entry)

    assert result["id"] == "R1"
    assert result["refused"] is expected_refused
    assert result["retrieved_correct"] is True
    assert result["answer"] == answer.strip()


def test_run_review_numbers_each_question():
    questions = {"questions": ["First?", "Second?"]}
    progress = [
        (number, total, entry["id"])
        for number, total, entry, _ in run_review(FakeAnswerGenerator("A"), questions)
    ]
    assert progress == [(1, 2, "Q1"), (2, 2, "Q2")]


def test_format_result(entry):
    result = evaluate_question(FakeAnswerGenerator("Two weeks."), entry)
    expected_result = "\n".join(
        [
            "=" * 80,
            "[1/5] R1   coverage: full",
            "Q: How long do I have to register?",
            "-" * 80,
            "Two weeks.",
            "-" * 80,
            "EXPECTED: Two weeks.",
            "RETRIEVED TOP-1:",
            "   1. final=0.9000 semantic=0.8000 bm25=0.7000 | bmg_en_17.txt (chunk 0)",
            "CORRECT SOURCE FOUND: yes (wanted: bmg_en_17.txt)",
            "",
        ]
    )
    assert format_result(1, 5, entry, result) == expected_result


def test_format_summary():
    results = [
        {"coverage": "full", "refused": False, "retrieved_correct": True},
        {"coverage": "full", "refused": True, "retrieved_correct": False},
        {"coverage": "none", "refused": True, "retrieved_correct": False},
    ]
    summary = format_summary(results)

    assert "  right source retrieved: 1/2 (50.0%)" in summary
    assert "  produced an answer: 1/2 (50.0%)" in summary
    assert "  correctly refused: 1/1 (100.0%)" in summary


@pytest.mark.parametrize(
    "count, expected_length",
    [(3, 3), (10, 10), (50, 10)],
)
def test_sample_questions(count, expected_length):
    questions = [f"Q{number}" for number in range(10)]
    sample = sample_questions(questions, count, seed=7)

    assert len(sample) == expected_length
    assert sample == sorted(sample, key=questions.index)
    assert sample == sample_questions(questions, count, seed=7)
