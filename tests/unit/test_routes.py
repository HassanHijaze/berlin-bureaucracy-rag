import pytest
from fastapi.testclient import TestClient

from berlin_rag.api.main import create_app
from berlin_rag.api.rate_limit import DailyRequestLimiter
from berlin_rag.config import DAILY_REQUEST_LIMIT


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def limiter(client):
    original_limiter = client.app.state.limiter
    client.app.state.limiter = DailyRequestLimiter()
    yield client.app.state.limiter
    client.app.state.limiter = original_limiter


@pytest.fixture
def model_reply(client, monkeypatch):
    reply = {"text": "Register within 14 days [bmg_en_17.txt]."}
    monkeypatch.setattr(
        client.app.state.answer_generator,
        "ask",
        lambda question: (reply["text"], []),
    )
    return reply


def test_health_and_home(client):
    health = client.get("/health")
    home = client.get("/")
    docs = client.get("/docs")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["documents"] == 68
    assert health.json()["chunks"] > 400
    assert home.status_code == 200
    assert "<h1>" in home.text
    assert docs.status_code == 200


@pytest.mark.parametrize(
    "question, expected_result",
    [("", 422), ("x" * 5000, 422), ("     ", 400)],
)
def test_invalid_questions_are_rejected(client, question, expected_result):
    response = client.post("/ask", json={"question": question})
    assert response.status_code == expected_result


def test_ask(client, model_reply, limiter):
    response = client.post("/ask", json={"question": "How long do I have to register?"})
    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == "Register within 14 days."
    assert body["refused"] is False
    assert body["sources"] == [{"file": "bmg_en_17.txt", "label": "BMG § 17"}]


@pytest.mark.parametrize(
    "reply, expected_result",
    [
        ("The sources I have do not cover this.", True),
        ("   ", True),
        ("Private plans do not cover dental care [tk_studenten_ausland.md].", False),
    ],
)
def test_ask_detects_refusals(client, model_reply, limiter, reply, expected_result):
    model_reply["text"] = reply
    body = client.post("/ask", json={"question": "Is this covered?"}).json()

    assert body["refused"] is expected_result
    assert bool(body["sources"]) is not expected_result


def test_ask_hides_model_errors(client, limiter, monkeypatch, caplog):
    def failing_ask(question):
        raise ConnectionError("upstream timeout with secret details")

    monkeypatch.setattr(client.app.state.answer_generator, "ask", failing_ask)
    response = client.post("/ask", json={"question": "Anything?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "The model call failed: ConnectionError"}
    assert "secret details" in caplog.text


def test_daily_limit_blocks_requests(client, model_reply, limiter):
    for _ in range(DAILY_REQUEST_LIMIT):
        limiter.record()

    blocked = client.post("/ask", json={"question": "One more?"})
    health = client.get("/health").json()

    assert blocked.status_code == 429
    assert health["requests_today"] == DAILY_REQUEST_LIMIT
    assert health["requests_served"] == DAILY_REQUEST_LIMIT
