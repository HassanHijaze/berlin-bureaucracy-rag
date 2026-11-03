from types import SimpleNamespace

import pytest

from berlin_rag.retrieval import expansion
from berlin_rag.retrieval.expansion import expand_query


@pytest.fixture
def fake_client(monkeypatch):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        message = SimpleNamespace(content="  Anmeldung Wohnung Frist § 17 BMG\n")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(expansion, "get_client", lambda: client)
    return requests


def test_expand_query(fake_client):
    expected_result = "Anmeldung Wohnung Frist § 17 BMG"

    assert expand_query("How long do I have to register?") == expected_result
    assert "How long do I have to register?" in fake_client[0]["messages"][0]["content"]


def test_expand_query_falls_back_when_the_call_fails(monkeypatch, caplog):
    def failing_client():
        raise RuntimeError("OPENAI_API_KEY not found")

    monkeypatch.setattr(expansion, "get_client", failing_client)

    assert expand_query("How long do I have to register?") == ""
    assert "Query expansion failed: OPENAI_API_KEY not found" in caplog.text
