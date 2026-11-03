"""HTTP API for the Berlin Bureaucracy RAG system.

``app`` is re-exported so the old start command ``uvicorn berlin_rag.api:app``
keeps working next to ``uvicorn berlin_rag.api.main:app``.
"""

from .main import app

__all__ = ["app"]
