import logging
from collections.abc import (
    AsyncIterator,
    Callable,
)
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from app.api.routes.auth import (
    router as auth_router,
)
from app.api.routes.chat import (
    router as chat_router,
)
from app.api.routes.conversations import (
    router as conversations_router,
)
from app.api.routes.feedback import (
    router as feedback_router,
)
from app.api.routes.health import (
    router as health_router,
)
from app.api.routes.sources import (
    router as sources_router,
)
from app.api.services import (
    ApplicationServices,
    build_application_services,
)
from app.api.settings import (
    get_api_settings,
)
from app.observability.logging import (
    configure_logging,
)
from app.observability.middleware import (
    RequestContextMiddleware,
)
from app.observability.telemetry import (
    TelemetryRuntime,
    configure_telemetry,
)


logger = logging.getLogger(__name__)


ServiceFactory = Callable[
    [],
    ApplicationServices,
]


def create_app(
    *,
    service_factory: ServiceFactory = (
        build_application_services
    ),
) -> FastAPI:
    api_settings = get_api_settings()

    configure_logging(
        service_name=api_settings.service_name,
        environment=api_settings.environment,
        log_level=api_settings.log_level,
        log_format=api_settings.log_format,
    )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        application.state.services = None

        try:
            services = service_factory()
            services.database_service.initialize()

            cache_ready = False

            if services.cache_service is not None:
                cache_ready = (
                    services
                    .cache_service
                    .initialize()
                )

            application.state.services = services

            logger.info(
                "Policy Copilot services loaded.",
                extra={
                    "event": "services_loaded",
                    "indexed_chunks": (
                        services
                        .vector_store
                        .document_count
                    ),
                    "embedding_model": (
                        services.embedding_model
                    ),
                    "generation_model": (
                        services.generation_model
                    ),
                    "reranker_backend": (
                        services.reranker_backend
                    ),
                    "auth_mode": (
                        services.auth_mode
                    ),
                    "cache_enabled": bool(
                        services.cache_service
                        and services
                        .cache_service
                        .enabled
                    ),
                    "cache_ready": cache_ready,
                },
            )

        except Exception as error:
            logger.exception(
                "Failed to load Policy Copilot services.",
                extra={
                    "event": (
                        "services_load_failed"
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            )
            raise

        yield

        loaded_services = getattr(
            application.state,
            "services",
            None,
        )

        if loaded_services is not None:
            if (
                loaded_services.cache_service
                is not None
            ):
                loaded_services.cache_service.close()

            loaded_services.database_service.dispose()

        telemetry_runtime: TelemetryRuntime | None = getattr(
            application.state,
            "telemetry_runtime",
            None,
        )

        if telemetry_runtime is not None:
            telemetry_runtime.force_flush()

        application.state.services = None

        logger.info(
            "Policy Copilot services released.",
            extra={
                "event": "services_released",
            },
        )

    application = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        description=(
            "Permission-aware enterprise banking "
            "policy and compliance assistant."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            api_settings.cors_origins
        ),
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
            "OPTIONS",
        ],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
        ],
    )

    application.add_middleware(
        RequestContextMiddleware
    )

    application.include_router(
        health_router,
        prefix=api_settings.prefix,
    )

    application.include_router(
        auth_router,
        prefix=api_settings.prefix,
    )

    application.include_router(
        chat_router,
        prefix=api_settings.prefix,
    )

    application.include_router(
        conversations_router,
        prefix=api_settings.prefix,
    )

    application.include_router(
        feedback_router,
        prefix=api_settings.prefix,
    )

    application.include_router(
        sources_router,
        prefix=api_settings.prefix,
    )

    application.state.telemetry_runtime = (
        configure_telemetry(
            application,
            settings=api_settings,
        )
    )

    return application
