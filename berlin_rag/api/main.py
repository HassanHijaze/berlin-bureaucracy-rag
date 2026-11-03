"""Create the FastAPI application and load the RAG system once at startup.

Usage (from the project root):
    uvicorn berlin_rag.api.main:app --host 0.0.0.0 --port 8000

Interactive documentation is served at /docs.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import LOG_FORMAT, LOG_LEVEL
from ..generation.answer import load_answer_generator
from .rate_limit import DailyRequestLimiter
from .routes import router

DESCRIPTION = (
    "Answers questions about German registration and residence law "
    "for international students in Berlin, from cited sources. "
    "Declines when the sources do not cover the question. "
    "Not legal advice."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the corpus and vectors before the first request is served.

    Args:
        app: The application whose state receives the shared objects.

    Yields:
        Control to the running application until it shuts down.
    """
    app.state.answer_generator = load_answer_generator(verbose=True)
    app.state.limiter = DailyRequestLimiter()
    app.state.started_at = time.time()
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with all routes mounted.

    Returns:
        The application. Its RAG system is loaded when the app starts.
    """
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
    app = FastAPI(
        title="Berlin Bureaucracy RAG",
        description=DESCRIPTION,
        version="1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
