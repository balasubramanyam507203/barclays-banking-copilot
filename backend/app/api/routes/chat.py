import logging
from time import perf_counter
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from opentelemetry import trace

from app.api.dependencies import (
    AccessContextDependency,
    CurrentPrincipalDependency,
    DatabaseSessionDependency,
    ServicesDependency,
)
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    GuardrailStatusResponse,
    SourceReferenceResponse,
    TokenUsageResponse,
)
from app.cache.models import CachedRagResponse
from app.database.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
)
from app.observability.context import (
    get_or_create_request_id,
    reset_conversation_id,
    reset_user_id,
    set_conversation_id,
    set_user_id,
)
from app.observability.metrics import (
    record_rag_error,
    record_rag_success,
)
from app.rag.context_builder import (
    PromptPackage,
    build_prompt_package,
)
from app.rag.generation_service import (
    GroundedAnswerResult,
)


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


router = APIRouter(
    tags=["Chat"],
)


conversation_store = ConversationStore()


def identity_value(value: Any) -> str:
    """Returns a stable string for enums or plain values."""

    enum_value = getattr(
        value,
        "value",
        None,
    )

    if enum_value is not None:
        return str(enum_value)

    return str(value)


def build_approved_sources(
    *,
    prompt_package: PromptPackage,
    answer_result: GroundedAnswerResult,
) -> list[SourceReferenceResponse]:
    """Returns sources used by a fully approved answer."""

    if answer_result.abstained:
        return []

    if not (
        answer_result.citation_validation_passed
        and answer_result
        .post_generation_guardrails_passed
    ):
        return []

    used_labels = set(
        answer_result.citations_used
    )

    return [
        SourceReferenceResponse(
            label=citation.label,
            document_id=citation.document_id,
            title=citation.title,
            version=citation.version,
            chunk_id=citation.chunk_id,
            source=citation.source,
            citation=citation.citation,
        )
        for citation in prompt_package.citations
        if citation.label in used_labels
    ]


def build_guardrail_response(
    answer_result: GroundedAnswerResult,
) -> GuardrailStatusResponse:
    return GuardrailStatusResponse(
        citation_validation_passed=(
            answer_result
            .citation_validation_passed
        ),
        post_generation_guardrails_passed=(
            answer_result
            .post_generation_guardrails_passed
        ),
        claims_checked=(
            answer_result.claims_checked
        ),
        supported_claims=(
            answer_result.supported_claims
        ),
    )


def build_usage_response(
    answer_result: GroundedAnswerResult,
) -> TokenUsageResponse:
    return TokenUsageResponse(
        input_tokens=(
            answer_result.input_tokens
        ),
        output_tokens=(
            answer_result.output_tokens
        ),
        total_tokens=(
            answer_result.total_tokens
        ),
    )


def build_cached_usage_response() -> TokenUsageResponse:
    """A cache hit consumes no new model tokens."""

    return TokenUsageResponse(
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
    )


def build_assistant_details(
    *,
    abstained: bool,
    model_called: bool,
    citations_used: list[str],
    sources: list[SourceReferenceResponse],
    evidence_count: int,
    guardrails: GuardrailStatusResponse,
    usage: TokenUsageResponse,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    cache_hit: bool,
) -> dict[str, Any]:
    return {
        "abstained": abstained,
        "model_called": model_called,
        "citations_used": citations_used,
        "sources": [
            source.model_dump(mode="json")
            for source in sources
        ],
        "evidence_count": evidence_count,
        "guardrails": guardrails.model_dump(
            mode="json"
        ),
        "usage": usage.model_dump(
            mode="json"
        ),
        "observability": {
            "retrieval_latency_ms": round(
                retrieval_latency_ms,
                3,
            ),
            "generation_latency_ms": round(
                generation_latency_ms,
                3,
            ),
        },
        "cache": {
            "hit": cache_hit,
        },
    }


def build_cached_response_payload(
    *,
    services: ServicesDependency,
    status_value: str,
    answer_result: GroundedAnswerResult,
    sources: list[SourceReferenceResponse],
    evidence_count: int,
    guardrails: GuardrailStatusResponse,
) -> CachedRagResponse | None:
    cache_service = services.cache_service

    if cache_service is None:
        return None

    guardrail_passed = (
        answer_result
        .citation_validation_passed
        and answer_result
        .post_generation_guardrails_passed
    )

    if not guardrail_passed:
        return None

    return CachedRagResponse(
        schema_version=(
            cache_service
            .settings
            .schema_version
        ),
        status=status_value,
        answer=answer_result.answer,
        abstained=answer_result.abstained,
        citations_used=list(
            answer_result.citations_used
        ),
        sources=[
            source.model_dump(mode="json")
            for source in sources
        ],
        evidence_count=evidence_count,
        guardrails=guardrails.model_dump(
            mode="json"
        ),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat(
    payload: ChatRequest,
    services: ServicesDependency,
    principal: CurrentPrincipalDependency,
    access_context: AccessContextDependency,
    database_session: DatabaseSessionDependency,
) -> ChatResponse:
    """
    Executes the permission-aware RAG pipeline.

    A Redis cache hit skips retrieval and generation while
    still persisting a fresh user and assistant message in
    the conversation database.
    """

    request_id = get_or_create_request_id()
    total_started_at = perf_counter()

    retrieval_latency_ms = 0.0
    generation_latency_ms = 0.0
    current_stage = "conversation"

    role = identity_value(
        principal.role
    )
    region = identity_value(
        principal.region
    )

    user_context_token = set_user_id(
        principal.subject
    )

    conversation_context_token = None

    requested_conversation_id = (
        str(payload.conversation_id)
        if payload.conversation_id is not None
        else None
    )

    try:
        with tracer.start_as_current_span(
            "rag.chat"
        ) as chat_span:
            chat_span.set_attribute(
                "app.request_id",
                request_id,
            )
            chat_span.set_attribute(
                "enduser.role",
                role,
            )
            chat_span.set_attribute(
                "enduser.region",
                region,
            )
            chat_span.set_attribute(
                "enduser.clearance_rank",
                int(principal.clearance_rank),
            )

            conversation = (
                conversation_store
                .create_or_get_conversation(
                    database_session,
                    principal=principal,
                    conversation_id=(
                        requested_conversation_id
                    ),
                    initial_question=(
                        payload.question
                    ),
                )
            )

            conversation_context_token = (
                set_conversation_id(
                    conversation.id
                )
            )

            chat_span.set_attribute(
                "app.conversation_id",
                conversation.id,
            )

            user_message = (
                conversation_store.add_message(
                    database_session,
                    conversation=conversation,
                    role="user",
                    content=payload.question,
                    request_id=request_id,
                )
            )

            cache_service = services.cache_service
            cache_key: str | None = None
            cached_response: (
                CachedRagResponse | None
            ) = None

            if cache_service is not None:
                current_stage = "cache_lookup"

                cache_key = cache_service.build_key(
                    question=payload.question,
                    role=role,
                    region=region,
                    clearance_rank=int(
                        principal.clearance_rank
                    ),
                )

                with tracer.start_as_current_span(
                    "rag.cache_lookup"
                ) as cache_span:
                    cached_response = (
                        cache_service.get(
                            cache_key
                        )
                    )

                    cache_span.set_attribute(
                        "cache.hit",
                        cached_response is not None,
                    )
                    cache_span.set_attribute(
                        "cache.enabled",
                        cache_service.enabled,
                    )
                    cache_span.set_attribute(
                        "cache.available",
                        cache_service.available,
                    )

            if cached_response is not None:
                current_stage = "cached_response_persistence"

                cached_sources = [
                    SourceReferenceResponse
                    .model_validate(source)
                    for source
                    in cached_response.sources
                ]

                cached_guardrails = (
                    GuardrailStatusResponse
                    .model_validate(
                        cached_response.guardrails
                    )
                )

                cached_usage = (
                    build_cached_usage_response()
                )

                assistant_details = (
                    build_assistant_details(
                        abstained=(
                            cached_response.abstained
                        ),
                        model_called=False,
                        citations_used=list(
                            cached_response
                            .citations_used
                        ),
                        sources=cached_sources,
                        evidence_count=(
                            cached_response
                            .evidence_count
                        ),
                        guardrails=(
                            cached_guardrails
                        ),
                        usage=cached_usage,
                        retrieval_latency_ms=0.0,
                        generation_latency_ms=0.0,
                        cache_hit=True,
                    )
                )

                assistant_message = (
                    conversation_store.add_message(
                        database_session,
                        conversation=conversation,
                        role="assistant",
                        content=(
                            cached_response.answer
                        ),
                        request_id=request_id,
                        status=(
                            cached_response.status
                        ),
                        details_json=(
                            assistant_details
                        ),
                    )
                )

                total_latency_ms = (
                    perf_counter()
                    - total_started_at
                ) * 1_000

                chat_span.set_attribute(
                    "cache.hit",
                    True,
                )
                chat_span.set_attribute(
                    "rag.status",
                    cached_response.status,
                )
                chat_span.set_attribute(
                    "rag.total_latency_ms",
                    total_latency_ms,
                )

                record_rag_success(
                    role=role,
                    status=(
                        cached_response.status
                    ),
                    retrieval_latency_ms=0.0,
                    generation_latency_ms=0.0,
                    total_latency_ms=(
                        total_latency_ms
                    ),
                    abstained=(
                        cached_response.abstained
                    ),
                    guardrail_passed=True,
                    cache_hit=True,
                )

                logger.info(
                    "Guarded RAG response returned from "
                    "Redis cache.",
                    extra={
                        "event": "rag_cache_hit",
                        "cache_hit": True,
                        "total_latency_ms": round(
                            total_latency_ms,
                            3,
                        ),
                        "evidence_count": (
                            cached_response
                            .evidence_count
                        ),
                        "citations_count": len(
                            cached_sources
                        ),
                        "abstained": (
                            cached_response
                            .abstained
                        ),
                        "model_called": False,
                        "guardrail_passed": True,
                        "role": role,
                        "region": region,
                        "clearance_rank": int(
                            principal.clearance_rank
                        ),
                    },
                )

                return ChatResponse(
                    request_id=request_id,
                    conversation_id=(
                        conversation.id
                    ),
                    user_message_id=(
                        user_message.id
                    ),
                    assistant_message_id=(
                        assistant_message.id
                    ),
                    status=(
                        cached_response.status
                    ),
                    answer=cached_response.answer,
                    abstained=(
                        cached_response.abstained
                    ),
                    model_called=False,
                    citations_used=list(
                        cached_response
                        .citations_used
                    ),
                    sources=cached_sources,
                    evidence_count=(
                        cached_response
                        .evidence_count
                    ),
                    guardrails=(
                        cached_guardrails
                    ),
                    usage=cached_usage,
                )

            chat_span.set_attribute(
                "cache.hit",
                False,
            )

            current_stage = "retrieval"
            retrieval_started_at = perf_counter()

            with tracer.start_as_current_span(
                "rag.retrieve"
            ) as retrieval_span:
                retrieval_response = (
                    services
                    .retrieval_service
                    .retrieve(
                        payload.question,
                        access_context=(
                            access_context
                        ),
                    )
                )

                retrieval_latency_ms = (
                    perf_counter()
                    - retrieval_started_at
                ) * 1_000

                retrieval_span.set_attribute(
                    "rag.retrieved_chunk_count",
                    len(
                        retrieval_response.results
                    ),
                )
                retrieval_span.set_attribute(
                    "rag.retrieval_latency_ms",
                    retrieval_latency_ms,
                )

            current_stage = "context_assembly"

            with tracer.start_as_current_span(
                "rag.context_assembly"
            ) as context_span:
                prompt_package = (
                    build_prompt_package(
                        retrieval_response,
                        settings=(
                            services.prompt_settings
                        ),
                    )
                )

                context_span.set_attribute(
                    "rag.evidence_count",
                    prompt_package.evidence_count,
                )
                context_span.set_attribute(
                    "rag.context_token_count",
                    prompt_package.context_token_count,
                )
                context_span.set_attribute(
                    "rag.should_abstain",
                    prompt_package.should_abstain,
                )

            current_stage = "generation"
            generation_started_at = perf_counter()

            with tracer.start_as_current_span(
                "rag.generate"
            ) as generation_span:
                answer_result = (
                    services
                    .generation_service
                    .generate(
                        prompt_package
                    )
                )

                generation_latency_ms = (
                    perf_counter()
                    - generation_started_at
                ) * 1_000

                guardrail_passed = (
                    answer_result
                    .citation_validation_passed
                    and answer_result
                    .post_generation_guardrails_passed
                )

                generation_span.set_attribute(
                    "gen_ai.model",
                    answer_result.model_name,
                )
                generation_span.set_attribute(
                    "rag.abstained",
                    answer_result.abstained,
                )
                generation_span.set_attribute(
                    "rag.model_called",
                    answer_result.model_called,
                )
                generation_span.set_attribute(
                    "rag.guardrail_passed",
                    guardrail_passed,
                )
                generation_span.set_attribute(
                    "rag.claims_checked",
                    answer_result.claims_checked,
                )
                generation_span.set_attribute(
                    "rag.supported_claims",
                    answer_result.supported_claims,
                )
                generation_span.set_attribute(
                    "rag.generation_latency_ms",
                    generation_latency_ms,
                )

                if answer_result.total_tokens is not None:
                    generation_span.set_attribute(
                        "gen_ai.usage.total_tokens",
                        answer_result.total_tokens,
                    )

            current_stage = "response_persistence"

            approved_sources = build_approved_sources(
                prompt_package=prompt_package,
                answer_result=answer_result,
            )

            response_status = (
                "abstained"
                if answer_result.abstained
                else "answered"
            )

            guardrail_response = (
                build_guardrail_response(
                    answer_result
                )
            )

            usage_response = (
                build_usage_response(
                    answer_result
                )
            )

            assistant_details = (
                build_assistant_details(
                    abstained=(
                        answer_result.abstained
                    ),
                    model_called=(
                        answer_result.model_called
                    ),
                    citations_used=list(
                        answer_result.citations_used
                    ),
                    sources=approved_sources,
                    evidence_count=(
                        prompt_package.evidence_count
                    ),
                    guardrails=(
                        guardrail_response
                    ),
                    usage=usage_response,
                    retrieval_latency_ms=(
                        retrieval_latency_ms
                    ),
                    generation_latency_ms=(
                        generation_latency_ms
                    ),
                    cache_hit=False,
                )
            )

            assistant_message = (
                conversation_store.add_message(
                    database_session,
                    conversation=conversation,
                    role="assistant",
                    content=answer_result.answer,
                    request_id=request_id,
                    status=response_status,
                    details_json=assistant_details,
                )
            )

            if (
                cache_service is not None
                and cache_key is not None
            ):
                current_stage = "cache_write"

                cache_payload = (
                    build_cached_response_payload(
                        services=services,
                        status_value=(
                            response_status
                        ),
                        answer_result=(
                            answer_result
                        ),
                        sources=approved_sources,
                        evidence_count=(
                            prompt_package
                            .evidence_count
                        ),
                        guardrails=(
                            guardrail_response
                        ),
                    )
                )

                if cache_payload is not None:
                    with tracer.start_as_current_span(
                        "rag.cache_write"
                    ) as cache_write_span:
                        cache_written = (
                            cache_service.set(
                                cache_key,
                                cache_payload,
                            )
                        )

                        cache_write_span.set_attribute(
                            "cache.write_success",
                            cache_written,
                        )

            total_latency_ms = (
                perf_counter()
                - total_started_at
            ) * 1_000

            guardrail_passed = (
                answer_result
                .citation_validation_passed
                and answer_result
                .post_generation_guardrails_passed
            )

            chat_span.set_attribute(
                "rag.status",
                response_status,
            )
            chat_span.set_attribute(
                "rag.total_latency_ms",
                total_latency_ms,
            )

            record_rag_success(
                role=role,
                status=response_status,
                retrieval_latency_ms=(
                    retrieval_latency_ms
                ),
                generation_latency_ms=(
                    generation_latency_ms
                ),
                total_latency_ms=(
                    total_latency_ms
                ),
                abstained=(
                    answer_result.abstained
                ),
                guardrail_passed=(
                    guardrail_passed
                ),
                cache_hit=False,
            )

            logger.info(
                "Guarded RAG request completed.",
                extra={
                    "event": (
                        "rag_request_completed"
                    ),
                    "cache_hit": False,
                    "retrieval_latency_ms": round(
                        retrieval_latency_ms,
                        3,
                    ),
                    "generation_latency_ms": round(
                        generation_latency_ms,
                        3,
                    ),
                    "total_latency_ms": round(
                        total_latency_ms,
                        3,
                    ),
                    "evidence_count": (
                        prompt_package
                        .evidence_count
                    ),
                    "citations_count": len(
                        approved_sources
                    ),
                    "claims_checked": (
                        answer_result
                        .claims_checked
                    ),
                    "supported_claims": (
                        answer_result
                        .supported_claims
                    ),
                    "abstained": (
                        answer_result.abstained
                    ),
                    "model_called": (
                        answer_result.model_called
                    ),
                    "guardrail_passed": (
                        guardrail_passed
                    ),
                    "role": role,
                    "region": region,
                    "clearance_rank": int(
                        principal.clearance_rank
                    ),
                    "generation_model": (
                        services.generation_model
                    ),
                    "reranker_backend": (
                        services.reranker_backend
                    ),
                },
            )

            return ChatResponse(
                request_id=request_id,
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                assistant_message_id=(
                    assistant_message.id
                ),
                status=response_status,
                answer=answer_result.answer,
                abstained=(
                    answer_result.abstained
                ),
                model_called=(
                    answer_result.model_called
                ),
                citations_used=list(
                    answer_result.citations_used
                ),
                sources=approved_sources,
                evidence_count=(
                    prompt_package.evidence_count
                ),
                guardrails=guardrail_response,
                usage=usage_response,
            )

    except ConversationNotFoundError as error:
        record_rag_error(
            role=role,
            stage=current_stage,
            error_type=type(error).__name__,
        )

        logger.warning(
            "Conversation was not found.",
            extra={
                "event": "rag_request_rejected",
                "error_stage": current_stage,
                "error_type": type(error).__name__,
                "role": role,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except ValueError as error:
        record_rag_error(
            role=role,
            stage=current_stage,
            error_type=type(error).__name__,
        )

        logger.warning(
            "RAG request validation failed.",
            extra={
                "event": "rag_request_rejected",
                "error_stage": current_stage,
                "error_type": type(error).__name__,
                "role": role,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        record_rag_error(
            role=role,
            stage=current_stage,
            error_type=type(error).__name__,
        )

        logger.exception(
            "Guarded RAG request failed.",
            extra={
                "event": "rag_request_failed",
                "error_stage": current_stage,
                "error_type": type(error).__name__,
                "role": role,
            },
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The policy assistant is temporarily "
                "unable to process this request."
            ),
        ) from error

    finally:
        if conversation_context_token is not None:
            reset_conversation_id(
                conversation_context_token
            )

        reset_user_id(
            user_context_token
        )
