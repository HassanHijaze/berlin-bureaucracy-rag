import pytest

from berlin_rag import openai_client
from berlin_rag.config import Settings


@pytest.fixture
def settings_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return Settings(_env_file=None)


def test_get_client_without_api_key_raises(monkeypatch, settings_without_key):
    monkeypatch.setattr(openai_client, "_client", None)
    monkeypatch.setattr(openai_client, "get_settings", lambda: settings_without_key)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not found"):
        openai_client.get_client()
