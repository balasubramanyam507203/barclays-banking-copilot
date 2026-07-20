from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.cache.fingerprint import (
    build_vector_index_fingerprint,
)
from app.cache.service import (
    RedisRagResponseCache,
)
from app.cache.settings import (
    get_cache_settings,
)
from app.config import (
    PromptSettings,
    get_embedding_settings,
    get_generation_settings,
    get_hybrid_retrieval_settings,
    get_prompt_settings,
    get_reranker_settings,
    get_vector_store_settings,
)
from app.database.service import (
    DatabaseService,
)
from app.database.settings import (
    get_database_settings,
)
from app.rag.embedding_service import (
    build_embedding_client,
)
from app.rag.faiss_store import (
    LocalFaissStore,
)
from app.rag.generation_service import (
    GroundedAnswerGenerationService,
    build_generation_client,
)
from app.rag.keyword_index import (
    LocalBm25Index,
)
from app.rag.reranker import (
    build_reranker,
)
from app.rag.retrieval_service import (
    PermissionAwareRetrievalService,
)
from app.security.authentication import (
    AuthenticationService,
    get_authentication_settings,
)


@dataclass(frozen=True)
class ApplicationServices:
    """Long-lived services loaded during startup."""

    retrieval_service: (
        PermissionAwareRetrievalService
    )

    generation_service: (
        GroundedAnswerGenerationService
    )

    authentication_service: (
        AuthenticationService
    )

    database_service: DatabaseService

    prompt_settings: PromptSettings
    vector_store: LocalFaissStore

    embedding_model: str
    generation_model: str
    reranker_backend: str

    auth_mode: str
    started_at: datetime

    cache_service: (
        RedisRagResponseCache | None
    ) = None


def validate_index_path(
    index_path: Path,
) -> None:
    if not index_path.exists():
        raise RuntimeError(
            "The FAISS index does not exist at "
            f"'{index_path}'. Run "
            "'python -m app.build_index' first."
        )


def build_application_services(
) -> ApplicationServices:
    embedding_settings = (
        get_embedding_settings()
    )

    vector_store_settings = (
        get_vector_store_settings()
    )

    hybrid_settings = (
        get_hybrid_retrieval_settings()
    )

    reranker_settings = (
        get_reranker_settings()
    )

    prompt_settings = (
        get_prompt_settings()
    )

    generation_settings = (
        get_generation_settings()
    )

    authentication_settings = (
        get_authentication_settings()
    )

    database_settings = (
        get_database_settings()
    )

    cache_settings = get_cache_settings()

    validate_index_path(
        vector_store_settings.index_path
    )

    vector_store = LocalFaissStore.load(
        vector_store_settings.index_path
    )

    keyword_index = LocalBm25Index(
        documents=vector_store.documents
    )

    embedding_client = (
        build_embedding_client(
            embedding_settings
        )
    )

    reranker = build_reranker(
        reranker_settings
    )

    retrieval_service = (
        PermissionAwareRetrievalService(
            vector_store=vector_store,
            keyword_index=keyword_index,
            embedding_client=embedding_client,
            embedding_settings=(
                embedding_settings
            ),
            hybrid_settings=hybrid_settings,
            reranker=reranker,
            reranker_settings=(
                reranker_settings
            ),
            default_top_k=(
                reranker_settings.top_n
            ),
            default_vector_minimum_score=(
                vector_store_settings
                .retrieval_min_score
            ),
        )
    )

    generation_client = (
        build_generation_client(
            generation_settings
        )
    )

    generation_service = (
        GroundedAnswerGenerationService(
            chat_model=generation_client,
            settings=generation_settings,
        )
    )

    authentication_service = (
        AuthenticationService(
            authentication_settings
        )
    )

    database_service = DatabaseService(
        database_settings
    )

    index_fingerprint = (
        build_vector_index_fingerprint(
            vector_store
        )
    )

    cache_service = RedisRagResponseCache(
        settings=cache_settings,
        index_fingerprint=index_fingerprint,
        embedding_model=(
            embedding_settings.model
        ),
        generation_model=(
            generation_settings.model
        ),
        reranker_backend=(
            reranker_settings.backend
        ),
    )

    return ApplicationServices(
        retrieval_service=retrieval_service,
        generation_service=generation_service,
        authentication_service=(
            authentication_service
        ),
        database_service=database_service,
        prompt_settings=prompt_settings,
        vector_store=vector_store,
        embedding_model=(
            embedding_settings.model
        ),
        generation_model=(
            generation_settings.model
        ),
        reranker_backend=(
            reranker_settings.backend
        ),
        auth_mode=(
            authentication_settings.mode
        ),
        started_at=datetime.now(
            timezone.utc
        ),
        cache_service=cache_service,
    )
