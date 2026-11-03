"""HTTP endpoints: validate input, call the answer generator, shape the response.

Endpoints:
    GET  /         Web interface.
    POST /ask      Answer a question with cited sources.
    GET  /health   Service status, corpus size, and uptime.
"""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from ..citations import cited_files, readable, strip_citations
from ..config import INDEX_HTML, TOP_K
from ..generation.answer import AnswerGenerator
from ..generation.prompts import REFUSAL_TEXT, is_refusal
from .rate_limit import DailyRequestLimiter
from .schemas import Answer, Health, Question, Source

logger = logging.getLogger(__name__)

router = APIRouter()


def get_answer_generator(request: Request) -> AnswerGenerator:
    """Provide the answer generator created at startup, for ``Depends``.

    Args:
        request: The incoming request, which gives access to the app state.

    Returns:
        The shared AnswerGenerator.
    """
    return request.app.state.answer_generator


def get_limiter(request: Request) -> DailyRequestLimiter:
    """Provide the daily request limiter created at startup, for ``Depends``.

    Args:
        request: The incoming request, which gives access to the app state.

    Returns:
        The shared DailyRequestLimiter.
    """
    return request.app.state.limiter


AnswerGeneratorDep = Annotated[AnswerGenerator, Depends(get_answer_generator)]
LimiterDep = Annotated[DailyRequestLimiter, Depends(get_limiter)]


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


@router.post("/ask", response_model=Answer)
def ask(
    body: Question, answer_generator: AnswerGeneratorDep, limiter: LimiterDep
) -> Answer:
    """Answer one question from the corpus.

    Args:
        body: The request body containing the question.
        answer_generator: The shared answer generator.
        limiter: The shared daily request limiter.

    Returns:
        The answer, whether it was refused, its sources, and the time taken.

    Raises:
        HTTPException: 400 for an empty question, 429 when today's request
            limit is reached, and 502 if the language model call fails.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty.")

    if limiter.is_exhausted():
        raise HTTPException(
            status_code=429, detail="Daily request limit reached. Try again tomorrow."
        )

    limiter.record()
    started = time.time()

    try:
        text, _ = answer_generator.ask(question)
    except Exception as exc:
        # Log full details on the server; expose only the error type.
        logger.exception("Model call failed")
        raise HTTPException(
            status_code=502, detail=f"The model call failed: {type(exc).__name__}"
        ) from exc

    if not text or not text.strip():
        text = REFUSAL_TEXT

    refused = is_refusal(text)

    return Answer(
        answer=strip_citations(text),
        refused=refused,
        sources=build_sources(text, refused),
        elapsed_seconds=round(time.time() - started, 2),
    )


@router.get("/health", response_model=Health)
def health(
    request: Request, answer_generator: AnswerGeneratorDep, limiter: LimiterDep
) -> Health:
    """Report service status, corpus size, retrieval settings, and uptime.

    Args:
        request: The incoming request, used to read the startup time.
        answer_generator: The shared answer generator.
        limiter: The shared daily request limiter.

    Returns:
        A description of the running instance.
    """
    chunks = answer_generator.retriever.chunks
    return Health(
        status="ok",
        documents=len({chunk["source"] for chunk in chunks}),
        chunks=len(chunks),
        k=TOP_K,
        requests_served=limiter.requests_total,
        requests_today=limiter.requests_today,
        uptime_seconds=round(time.time() - request.app.state.started_at),
    )


@router.get("/", response_class=HTMLResponse)
def home() -> str:
    """Serve the web interface.

    Returns:
        The contents of ``static/index.html``.
    """
    return INDEX_HTML.read_text(encoding="utf-8")
