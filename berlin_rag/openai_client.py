"""The shared OpenAI client, created on first use.

Retrieval (embeddings, query expansion) and generation both call OpenAI,
so the client lives here instead of in either layer.
"""

from openai import OpenAI

from .config import get_settings

_client = None


def get_client() -> OpenAI:
    """Create the OpenAI client on first use, so offline code runs without a key.

    Returns:
        The shared OpenAI client.

    Raises:
        RuntimeError: If no OpenAI API key is configured.
    """
    global _client

    if _client is None:
        api_key = get_settings().openai_api_key
        if api_key is None or not api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY not found. Put it in your API.env file.")
        _client = OpenAI(api_key=api_key.get_secret_value())

    return _client
