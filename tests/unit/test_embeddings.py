import numpy as np
import pytest

from berlin_rag.config import CACHE_DIR
from berlin_rag.corpus.loading import build_corpus
from berlin_rag.retrieval import embeddings
from berlin_rag.retrieval.embeddings import cached_vectors, corpus_fingerprint


@pytest.fixture
def chunks():
    return [
        {"text": "Anmeldung der Wohnung", "source": "a.md", "index": 0},
        {"text": "Visum zum Studium", "source": "b.md", "index": 0},
    ]


@pytest.fixture
def fake_embed_batch(monkeypatch):
    calls = []

    def embed(texts):
        calls.append(texts)
        return np.arange(len(texts) * 3, dtype=float).reshape(len(texts), 3)

    monkeypatch.setattr(embeddings, "embed_batch", embed)
    return calls


def make_chunks(texts):
    return [
        {"text": text, "source": "a.md", "index": i} for i, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    "texts, other_texts, expected_result",
    [
        (["a", "b"], ["a", "b"], True),
        (["Fee: 10 EUR", "b"], ["Fee: 12 EUR", "b"], False),
        (["ab", "c"], ["a", "bc"], False),
    ],
)
def test_corpus_fingerprint(texts, other_texts, expected_result):
    key = corpus_fingerprint(make_chunks(texts))
    other_key = corpus_fingerprint(make_chunks(other_texts))
    assert (key == other_key) is expected_result


def test_cached_vectors_creates_and_saves_the_cache(chunks, tmp_path, fake_embed_batch):
    expected_result = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

    vectors = cached_vectors(chunks, cache_dir=tmp_path)

    np.testing.assert_array_equal(vectors, expected_result)
    assert fake_embed_batch == [["Anmeldung der Wohnung", "Visum zum Studium"]]
    assert tmp_path.joinpath(f"vectors_{corpus_fingerprint(chunks)}.npy").exists()


def test_cached_vectors_reuses_the_cache(chunks, tmp_path, fake_embed_batch):
    expected_result = np.ones((2, 3))
    cache_path = tmp_path / f"vectors_{corpus_fingerprint(chunks)}.npy"
    np.save(cache_path, expected_result)

    vectors = cached_vectors(chunks, cache_dir=tmp_path)

    np.testing.assert_array_equal(vectors, expected_result)
    assert fake_embed_batch == []


def test_cached_vectors_exist_for_the_real_corpus():
    _, chunks = build_corpus()
    cache_path = CACHE_DIR / f"vectors_{corpus_fingerprint(chunks)}.npy"

    assert cache_path.exists()
    assert np.load(cache_path).shape == (len(chunks), 1536)
