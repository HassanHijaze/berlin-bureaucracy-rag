"""Pydantic request and response models for the HTTP API."""

from pydantic import BaseModel, Field

from ..config import MAX_QUESTION_CHARS


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


class Health(BaseModel):
    """Response body for ``GET /health``."""

    status: str = Field(..., description="Always 'ok' while the service runs.")
    documents: int = Field(..., description="Number of source documents.")
    chunks: int = Field(..., description="Number of searchable chunks.")
    k: int = Field(..., description="Chunks given to the model per question.")
    requests_served: int = Field(..., description="Answered since startup.")
    requests_today: int = Field(..., description="Answered today (UTC).")
    uptime_seconds: int = Field(..., description="Seconds since startup.")
