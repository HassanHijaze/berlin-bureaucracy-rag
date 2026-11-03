from types import SimpleNamespace

import pytest

from berlin_rag.generation import answer
from berlin_rag.generation.answer import AnswerGenerator, build_context


@pytest.fixture
def hits():
    return [
        (
            {
                "text": "[bmg_en_17.txt]\nRegister within two weeks.",
                "source": "bmg_en_17.txt",
            },
            0.9,
            0.8,
            0.7,
        ),
        (
            {"text": "[lea_termine.md]\nBook online.", "source": "lea_termine.md"},
            0.5,
            0.4,
            0.3,
        ),
    ]


@pytest.fixture
def fake_client(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        message = SimpleNamespace(content="Register within 14 days [bmg_en_17.txt].")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(answer, "get_client", lambda: client)
    return requests


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.searches = []

    def search(self, question, k):
        self.searches.append((question, k))
        return self.hits


def test_build_context(hits):
    expected_result = (
        "\n--- CHUNK 1 [bmg_en_17.txt] ---\n"
        "[bmg_en_17.txt]\nRegister within two weeks.\n"
        "\n"
        "\n--- CHUNK 2 [lea_termine.md] ---\n[lea_termine.md]\nBook online.\n"
    )
    assert build_context(hits) == expected_result


def test_answer_generator_ask(hits, fake_client):
    retriever = FakeRetriever(hits)

    text, returned_hits = AnswerGenerator(retriever).ask("How long?", k=2)
    prompt = fake_client[0]["messages"][0]["content"]

    assert text == "Register within 14 days [bmg_en_17.txt]."
    assert returned_hits == hits
    assert retriever.searches == [("How long?", 2)]
    assert build_context(hits) in prompt
    assert "How long?" in prompt
