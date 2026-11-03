import pytest

from berlin_rag.config import MAX_CHUNK
from berlin_rag.corpus.loading import build_corpus, load_documents


@pytest.fixture
def corpus_dir(tmp_path):
    statutes = tmp_path / "bmg_sections"
    statutes.mkdir()
    statutes.joinpath("bmg_de_17.txt").write_text(
        "§ 17 Anmeldung\n(1) Short.", encoding="utf-8"
    )
    tmp_path.joinpath("lea.md").write_text(
        "Intro text\nGebühren\nkeine", encoding="utf-8"
    )
    return tmp_path


def test_load_documents(corpus_dir):
    expected_result = [
        {"name": "bmg_de_17.txt", "text": "§ 17 Anmeldung\n(1) Short."},
        {"name": "lea.md", "text": "Intro text\nGebühren\nkeine"},
    ]
    assert load_documents(corpus_dir, ["bmg_sections"]) == expected_result


def test_load_documents_skips_missing_directories(corpus_dir):
    documents = load_documents(corpus_dir, ["missing_sections", "bmg_sections"])
    assert [document["name"] for document in documents] == ["bmg_de_17.txt", "lea.md"]


def test_build_corpus(corpus_dir):
    expected_result = [
        {
            "text": "[bmg_de_17.txt]\n§ 17 Anmeldung\n(1) Short.",
            "source": "bmg_de_17.txt",
            "index": 0,
        },
        {
            "text": "[lea.md]\nIntro text\n[lea.md]\nGebühren\nkeine",
            "source": "lea.md",
            "index": 0,
        },
    ]
    _, chunks = build_corpus(corpus_dir)
    assert chunks == expected_result


def test_build_corpus_on_the_real_corpus():
    documents, chunks = build_corpus()
    indices = {}
    for chunk in chunks:
        indices.setdefault(chunk["source"], []).append(chunk["index"])

    assert len(documents) > 50
    assert len(indices) == len(documents)
    assert len(chunks) > 400
    assert all(chunk["source"] and chunk["text"].strip() for chunk in chunks)
    assert all(len(chunk["text"]) <= MAX_CHUNK + 50 for chunk in chunks)
    assert all(sorted(i) == list(range(len(i))) for i in indices.values())
