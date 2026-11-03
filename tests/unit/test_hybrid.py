import numpy as np
import pytest

from berlin_rag.retrieval.hybrid import (
    cosine_similarities,
    minmax_normalize,
    select_top_k,
)


@pytest.fixture
def chunks():
    return [
        {"text": "a0", "source": "a.md", "index": 0},
        {"text": "a1", "source": "a.md", "index": 1},
        {"text": "a2", "source": "a.md", "index": 2},
        {"text": "b0", "source": "b.md", "index": 0},
    ]


@pytest.mark.parametrize(
    "scores, expected_result",
    [
        ([2.0, 4.0, 6.0], [0.0, 0.5, 1.0]),
        ([3.0, 3.0], [0.0, 0.0]),
    ],
)
def test_minmax_normalize(scores, expected_result):
    np.testing.assert_allclose(minmax_normalize(scores), expected_result)


def test_cosine_similarities():
    vectors = np.array([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    expected_result = [1.0, 0.0, -1.0]
    np.testing.assert_allclose(
        cosine_similarities(vectors, np.array([5.0, 0.0])), expected_result
    )


@pytest.mark.parametrize(
    "k, max_per_source, expected_result",
    [
        (4, 2, [("a.md", 2), ("a.md", 1), ("b.md", 0)]),
        (2, 2, [("a.md", 2), ("a.md", 1)]),
        (4, 1, [("a.md", 2), ("b.md", 0)]),
    ],
)
def test_select_top_k(chunks, k, max_per_source, expected_result):
    final_scores = np.array([0.1, 0.8, 0.9, 0.5])
    hits = select_top_k(
        chunks, final_scores, final_scores, final_scores, k, max_per_source
    )
    assert [(chunk["source"], chunk["index"]) for chunk, *_ in hits] == expected_result
