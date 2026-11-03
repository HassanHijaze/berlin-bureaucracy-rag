import numpy as np
import pytest

from berlin_rag.retrieval.bm25 import BM25Index, tokenize


@pytest.fixture
def chunks():
    return [
        {"text": "Anmeldung der Wohnung beim Bürgeramt", "source": "a.md", "index": 0},
        {"text": "Visum zum Studium", "source": "b.md", "index": 0},
        {"text": "Anmeldung Anmeldung Frist", "source": "c.md", "index": 0},
    ]


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Anmeldung, Wohnung!", ["anmeldung", "wohnung"]),
        ("§ 17 BMG", ["17", "bmg"]),
        ("", []),
    ],
)
def test_tokenize(text, expected_result):
    assert tokenize(text) == expected_result


def test_bm25_index_score(chunks):
    scores = BM25Index(chunks).score("Wo mache ich die Anmeldung?")
    assert scores[1] == 0
    assert scores[2] > scores[0] > 0


@pytest.mark.parametrize("query", ["", "Rundfunkbeitrag"])
def test_bm25_index_score_without_matching_words(chunks, query):
    np.testing.assert_array_equal(BM25Index(chunks).score(query), [0.0, 0.0, 0.0])
