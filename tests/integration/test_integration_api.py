import pytest
from fastapi.testclient import TestClient

from berlin_rag.api.main import create_app

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_ask_with_the_real_model(client):
    answered = client.post(
        "/ask", json={"question": "How long do I have to register?"}
    ).json()
    refused = client.post(
        "/ask", json={"question": "What is the Rundfunkbeitrag?"}
    ).json()

    assert set(answered) == {"answer", "refused", "sources", "elapsed_seconds"}
    assert answered["refused"] is False
    assert "[" not in answered["answer"]
    assert refused["refused"] is True
    assert refused["sources"] == []
