import copy

import pytest

from berlin_rag.corpus.chunking import (
    chunk_berlin,
    chunk_legal,
    merge_small,
    split_hard,
)


@pytest.fixture
def chunks():
    return [
        {"text": "x" * 400, "source": "a.md", "index": 0},
        {"text": "Gebühren\nkeine", "source": "a.md", "index": 1},
        {"text": "y" * 400, "source": "a.md", "index": 2},
        {"text": "tiny", "source": "b.md", "index": 0},
    ]


def test_merge_small(chunks):
    expected_result = [
        {"text": "x" * 400 + "\nGebühren\nkeine", "source": "a.md", "index": 0},
        {"text": "y" * 400, "source": "a.md", "index": 1},
        {"text": "tiny", "source": "b.md", "index": 0},
    ]
    assert merge_small(chunks, min_size=150) == expected_result


def test_merge_small_does_not_change_the_input(chunks):
    original = copy.deepcopy(chunks)
    merge_small(chunks, min_size=150)
    assert chunks == original


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("§ 1 Kurz\n(1) Short rule.", ["§ 1 Kurz\n(1) Short rule."]),
        (
            "§ 17 Anmeldung\n(1) " + "x" * 400 + "\n(2) " + "y" * 400,
            ["§ 17 Anmeldung\n(1) " + "x" * 400, "[§ 17 Anmeldung]\n(2) " + "y" * 400],
        ),
    ],
)
def test_chunk_legal(text, expected_result):
    assert chunk_legal(text, max_size=750) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        (
            "Intro text\nVoraussetzungen\nA flat\nGebühren\nkeine",
            ["Intro text", "Voraussetzungen\nA flat", "Gebühren\nkeine"],
        ),
        ("No headings here", ["No headings here"]),
    ],
)
def test_chunk_berlin(text, expected_result):
    assert chunk_berlin(text) == expected_result


@pytest.mark.parametrize(
    "text, max_size, overlap, expected_result",
    [
        ("abc", 750, 100, ["abc"]),
        ("abcdefghij", 4, 1, ["abcd", "defg", "ghij", "j"]),
    ],
)
def test_split_hard(text, max_size, overlap, expected_result):
    assert split_hard(text, max_size=max_size, overlap=overlap) == expected_result
