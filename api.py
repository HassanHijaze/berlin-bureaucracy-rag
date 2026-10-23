"""HTTP API for the Berlin Bureaucracy RAG system.

Exposes the retrieval-augmented question-answering system as a FastAPI
application. Answers are grounded in German statutes and official Berlin
administrative sources, returned with their cited sources, and declined
when the corpus does not support them.

The RAG system is initialised once at startup and shared across requests.

Endpoints:
    GET  /         Web interface.
    POST /ask      Answer a question with cited sources.
    GET  /health   Service status, corpus size, and uptime.
    GET  /docs     Interactive OpenAPI documentation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from citations import cited_files, readable, strip_citations
from rag import K, System

# ================================================================
# CONFIGURATION
# ================================================================

MAX_QUESTION_CHARS = 500
"""Maximum accepted question length, in characters."""

DAILY_REQUEST_LIMIT = 200
"""Maximum number of answered questions per running instance."""

REFUSAL_TEXT = "The sources I have do not cover this."
"""Fallback answer when the model returns an empty response."""

REFUSAL_MARKER = "do not cover"
"""Phrase that identifies a refusal in the model's answer."""

INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
"""Location of the web interface."""

# ================================================================
# APPLICATION
# ================================================================

app = FastAPI(
    title="Berlin Bureaucracy RAG",
    description=(
        "Answers questions about German registration and residence law "
        "for international students in Berlin, from cited sources. "
        "Declines when the sources do not cover the question. "
        "Not legal advice."
    ),
    version="1.0",
)

system = System()
state: dict[str, float] = {"count": 0, "started": time.time()}

# ================================================================
# SCHEMAS
# ================================================================


class Question(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="The user's question, in English or German.",
        examples=["How long do I have to register my address?"],
    )


class Source(BaseModel):
    """A cited source document."""

    file: str = Field(..., description="Filename of the corpus document.")
    label: str = Field(..., description="Human-readable source name.")


class Answer(BaseModel):
    """Response body for ``POST /ask``."""

    answer: str = Field(..., description="The answer without citation markers.")
    refused: bool = Field(..., description="True if the sources do not cover it.")
    sources: list[Source] = Field(..., description="Sources cited in the answer.")
    elapsed_seconds: float = Field(..., description="Time taken to answer.")


# ================================================================
# HELPERS
# ================================================================


def validate_request(question: str) -> None:
    """Reject empty questions and requests beyond the daily limit.

    Args:
        question: The question after stripping surrounding whitespace.

    Raises:
        HTTPException: 400 if the question is empty, 429 if the request
            limit for this instance has been reached.
    """
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty.")

    if state["count"] >= DAILY_REQUEST_LIMIT:
        raise HTTPException(
            status_code=429, detail="Request limit reached for this instance."
        )


def build_sources(text: str, refused: bool) -> list[Source]:
    """Collect the cited sources of an answer as labelled entries.

    Args:
        text: The generated answer, including citation markers.
        refused: Whether the answer is a refusal.

    Returns:
        The cited sources, sorted by filename. Empty for refusals.
    """
    if refused:
        return []

    return [Source(file=f, label=readable(f)) for f in sorted(cited_files(text))]


# ================================================================
# ENDPOINTS
# ================================================================


@app.post("/ask", response_model=Answer)
def ask(q: Question) -> Answer:
    """Answer one question from the corpus.

    Args:
        q: The request body containing the question.

    Returns:
        The answer, whether it was refused, its sources, and the time taken.

    Raises:
        HTTPException: 400 for an empty question, 429 when the request limit
            is reached, and 502 if the language model call fails.
    """
    question = q.question.strip()
    validate_request(question)
    state["count"] += 1
    started = time.time()

    try:
        text, _ = system.ask(question)
    except Exception as exc:
        # Log full details on the server; expose only the error type.
        print(f"MODEL ERROR: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502, detail=f"The model call failed: {type(exc).__name__}"
        ) from exc

    if not text or not text.strip():
        text = REFUSAL_TEXT

    refused = REFUSAL_MARKER in text.lower()

    return Answer(
        answer=strip_citations(text),
        refused=refused,
        sources=build_sources(text, refused),
        elapsed_seconds=round(time.time() - started, 2),
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Report service status, corpus size, retrieval settings, and uptime.

    Returns:
        A JSON object describing the running instance.
    """
    return {
        "status": "ok",
        "documents": len(system.documents),
        "chunks": len(system.chunks),
        "k": K,
        "requests_served": state["count"],
        "uptime_seconds": round(time.time() - state["started"]),
    }


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the web interface.

    Returns:
        The contents of ``static/index.html``.
    """
    return INDEX_HTML.read_text(encoding="utf-8")