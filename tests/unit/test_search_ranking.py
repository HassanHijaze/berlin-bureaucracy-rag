import numpy as np
import pytest

from berlin_rag.corpus.loading import build_corpus
from berlin_rag.retrieval import hybrid
from berlin_rag.retrieval.embeddings import cached_vectors


@pytest.fixture(scope="module")
def retriever():
    _, chunks = build_corpus()
    return hybrid.Retriever(chunks, cached_vectors(chunks))


@pytest.mark.parametrize(
    "question, expansion, query_chunk, expected_result",
    [
        (
            "How long do I have to register my address?",
            "Anmeldung Wohnung Frist zwei Wochen § 17 BMG",
            ("bmg_en_17.txt", 0),
            [
                ("bmg_en_17.txt", 0, 0.8023),
                ("bmg_en_27.txt", 3, 0.7635),
                ("bmg_en_21.txt", 1, 0.6757),
                ("bmg_de_17.txt", 0, 0.6291),
                ("bmg_en_17.txt", 1, 0.6066),
                ("bmg_en_29.txt", 3, 0.5954),
                ("bmg_en_19.txt", 3, 0.5867),
                ("bmg_en_19.txt", 5, 0.5759),
                ("bmg_en_54.txt", 1, 0.5535),
                ("bmg_en_23.txt", 7, 0.5533),
                ("bmg_en_23a.txt", 3, 0.5477),
                ("bmg_en_24.txt", 1, 0.5351),
            ],
        ),
        (
            "Can I work as a student in Germany?",
            "Erwerbstätigkeit Studium 140 Tage § 16b AufenthG",
            ("aufenthg_en_16b.txt", 0),
            [
                ("aufenthg_en_16b.txt", 0, 0.7220),
                ("aufenthg_de_16b.txt", 0, 0.5453),
                ("aufenthg_en_16.txt", 0, 0.5400),
                ("aufenthg_en_16b.txt", 10, 0.5390),
                ("vab_a16b.txt", 58, 0.5304),
                ("aufenthg_en_20.txt", 1, 0.5267),
                ("vab_a16b.txt", 86, 0.5219),
                ("aufenthg_en_16f.txt", 2, 0.4888),
                ("aufenthg_en_16f.txt", 1, 0.4809),
                ("aufenthg_en_20.txt", 2, 0.4356),
                ("aufenthg_de_16b.txt", 8, 0.4268),
                ("vab_a16f.txt", 0, 0.3910),
            ],
        ),
        (
            "Do I need health insurance as a student?",
            "Krankenversicherung Studierende Versicherungspflicht § 5 SGB V",
            ("tk_studenten_ausland.md", 0),
            [
                ("tk_studenten_ausland.md", 0, 0.8200),
                ("tk_studenten_ausland.md", 3, 0.7018),
                ("lea_krankenversicherung.md", 0, 0.6180),
                ("lea_krankenversicherung.md", 3, 0.4723),
                ("sgb_de_199a.txt", 0, 0.4375),
                ("sgb_de_199a.txt", 4, 0.4234),
                ("aufenthg_en_20.txt", 2, 0.4216),
                ("vab_a16b.txt", 13, 0.4117),
                ("gkv_versicherte.md", 0, 0.4087),
                ("gkv_versicherte.md", 15, 0.4018),
                ("vab_a16b.txt", 1, 0.3964),
                ("vab_a20.txt", 44, 0.3834),
            ],
        ),
    ],
)
def test_search_ranking(
    retriever, monkeypatch, question, expansion, query_chunk, expected_result
):
    chunk_positions = {
        (chunk["source"], chunk["index"]): position
        for position, chunk in enumerate(retriever.chunks)
    }
    query_vector = retriever.vectors[chunk_positions[query_chunk]]
    monkeypatch.setattr(hybrid, "embed_query", lambda text: query_vector)
    monkeypatch.setattr(hybrid, "expand_query", lambda text: expansion)

    hits = retriever.search(question)
    ranking = [(chunk["source"], chunk["index"]) for chunk, *_ in hits]
    scores = [final_score for _, final_score, _, _ in hits]

    assert ranking == [(source, index) for source, index, _ in expected_result]
    np.testing.assert_allclose(
        scores, [score for _, _, score in expected_result], atol=1e-4
    )
