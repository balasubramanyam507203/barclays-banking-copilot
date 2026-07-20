from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.dependencies import (
    ServicesDependency,
)
from app.api.schemas import (
    HealthResponse,
)
from app.api.settings import (
    get_api_settings,
)


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "/live",
    response_model=HealthResponse,
)
def get_liveness() -> HealthResponse:
    """
    Confirms that the FastAPI process is running.

    This endpoint does not require the RAG pipeline.
    """

    api_settings = get_api_settings()

    return HealthResponse(
        status="ok",
        service=api_settings.service_name,
        version=api_settings.version,
        checked_at=datetime.now(
            timezone.utc
        ),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
)
def get_readiness(
    services: ServicesDependency,
) -> HealthResponse:
    """
    Confirms that the index and RAG services were loaded.
    """

    api_settings = get_api_settings()

    return HealthResponse(
        status="ready",
        service=api_settings.service_name,
        version=api_settings.version,
        checked_at=datetime.now(
            timezone.utc
        ),
        started_at=services.started_at,
        indexed_chunks=(
            services.vector_store.document_count
        ),
        embedding_model=(
            services.embedding_model
        ),
        generation_model=(
            services.generation_model
        ),
        reranker_backend=(
            services.reranker_backend
        ),
    )