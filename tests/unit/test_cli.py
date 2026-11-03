import json

import pytest

from berlin_rag import cli


class FakeAnswerGenerator:
    def ask(self, question):
        hits = [({"source": "bmg_en_17.txt", "index": 0}, 0.9, 0.8, 0.7)]
        return "  Register within 14 days [bmg_en_17.txt].\n", hits


@pytest.fixture
def fake_generator(monkeypatch):
    monkeypatch.setattr(cli, "load_answer_generator", FakeAnswerGenerator)


@pytest.fixture
def eval_file(tmp_path, monkeypatch):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps({"questions": [{"id": "R1", "question": "How long?"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "EVAL_FILE", path)
    return path


def test_ask(fake_generator, capsys):
    cli.main(["ask", "How", "long?"])
    expected_result = (
        "\nANSWER\n" + "=" * 80 + "\nRegister within 14 days [bmg_en_17.txt].\n"
    )
    assert capsys.readouterr().out == expected_result


@pytest.mark.parametrize("args", [["ask"], ["unknown"]])
def test_invalid_arguments_print_usage(args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args)

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    assert capsys.readouterr().out == cli.USAGE + "\n"


@pytest.mark.parametrize("args", [[], ["evaluate"]])
def test_evaluate(fake_generator, eval_file, capsys, args):
    cli.main(args)
    output = capsys.readouterr().out

    assert "[1/1] R1   coverage: -" in output
    assert "SUMMARY EVALUATION" in output
    assert "produced an answer: 1/1 (100.0%)" in output


def test_evaluate_without_question_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "EVAL_FILE", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        cli.main(["evaluate"])


@pytest.mark.parametrize(
    "options, expected_result",
    [
        ([], (None, None)),
        (["--sample", "10"], (10, None)),
        (["--sample", "10", "--seed", "42"], (10, 42)),
        (["--seed", "0", "--sample", "3"], (3, 0)),
    ],
)
def test_parse_evaluate_options(options, expected_result):
    assert cli.parse_evaluate_options(options) == expected_result


@pytest.mark.parametrize(
    "options",
    [["--sample"], ["--sample", "abc"], ["--sample", "0"], ["--size", "10"]],
)
def test_parse_evaluate_options_rejects_bad_input(options):
    with pytest.raises(ValueError):
        cli.parse_evaluate_options(options)


def test_evaluate_with_bad_sample_prints_usage(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["evaluate", "--sample", "abc"])

    assert exit_info.value.code == cli.EXIT_USAGE_ERROR
    assert capsys.readouterr().out == cli.USAGE + "\n"


def test_evaluate_sample(fake_generator, tmp_path, monkeypatch, capsys):
    path = tmp_path / "questions.json"
    questions = [{"id": f"R{number}", "question": "How?"} for number in range(20)]
    path.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    monkeypatch.setattr(cli, "EVAL_FILE", path)

    cli.main(["evaluate", "--sample", "5", "--seed", "42"])
    output = capsys.readouterr().out

    assert "Random sample: 5 of 20 questions (seed 42)" in output
    assert output.count("coverage: -") == 5
    assert "[5/5]" in output
