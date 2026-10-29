
"""FastAPI backend for The Librarian."""

import logging
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.generation import OLLAMA_MODEL, generate_answer

logger = logging.getLogger(__name__)

app = FastAPI(
    title="The Librarian API",
    description=(
        "A retrieval-augmented question-answering API "
        "using public South Mediterranean University sources."
    ),
    version="0.2.0",
)


class AskRequest(BaseModel):
    """Request body for asking a question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to answer using the available SMU sources.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject questions containing only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("Please provide a non-empty question.")
        return value


class Source(BaseModel):
    """A source passage used to support an answer."""

    citation: str
    title: str
    source_url: str
    chunk_id: str


class Timings(BaseModel):
    """Timing information returned by the generation pipeline."""

    total: float | None = None
    generation: float | None = None
    prompt_evaluation: float | None = None


class AskResponse(BaseModel):
    """Response returned by the question-answering endpoint."""

    question: str
    answer: str
    sources: list[dict[str, Any]]
    generation_model: str
    timings_ms: dict[str, Any] = Field(default_factory=dict)
    service_latency_ms: float


class HealthResponse(BaseModel):
    """Health status of the API and its generation model."""

    status: str
    generation_model: str
    model_available: bool


@app.get("/")
def root() -> dict[str, str]:
    """Describe the API and link to its documentation."""
    return {
        "name": "The Librarian API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check whether Ollama is reachable and the configured model is installed."""
    from src.generation import OLLAMA_BASE_URL

    model_available = False

    try:
        response = httpx.get(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            timeout=3.0,
        )
        response.raise_for_status()
        models = response.json().get("models", [])
        model_available = any(
            model.get("name") == OLLAMA_MODEL
            or model.get("name", "").split(":")[0]
            == OLLAMA_MODEL.split(":")[0]
            and model.get("name", "").endswith(OLLAMA_MODEL.split(":")[-1])
            for model in models
        )
    except (httpx.HTTPError, ValueError):
        logger.warning("Ollama health check failed.", exc_info=True)

    return HealthResponse(
        status="ok" if model_available else "degraded",
        generation_model=OLLAMA_MODEL,
        model_available=model_available,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Retrieve evidence and generate an answer to a question."""
    start_time = time.perf_counter()

    try:
        result = generate_answer(request.question)

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result.get("sources", []),
            generation_model=result.get(
                "generation_model",
                OLLAMA_MODEL,
            ),
            timings_ms=result.get("timings_ms", {}),
            service_latency_ms=elapsed_ms,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except RuntimeError as error:
        logger.warning("Question processing failed: %s", error)
        raise HTTPException(
            status_code=503,
            detail="The question could not be processed. Check the model and retrieval services.",
        ) from error

    except Exception as error:
        logger.exception("Unexpected error while answering a question.")
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from error
