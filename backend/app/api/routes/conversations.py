from fastapi import (
    APIRouter,
    HTTPException,
    Path,
    Query,
    status,
)

from app.api.dependencies import (
    CurrentPrincipalDependency,
    DatabaseSessionDependency,
)
from app.api.schemas import (
    ConversationDetailResponse,
    ConversationMessageResponse,
    ConversationSummaryResponse,
    FeedbackResponse,
    GuardrailStatusResponse,
    SourceReferenceResponse,
    TokenUsageResponse,
)
from app.database.conversation_store import (
    ConversationNotFoundError,
    ConversationStore,
)
from app.database.models import (
    ConversationMessage,
)


router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


conversation_store = ConversationStore()


def build_feedback_response(
    message: ConversationMessage,
) -> FeedbackResponse | None:
    feedback = message.feedback

    if feedback is None:
        return None

    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


def build_message_response(
    message: ConversationMessage,
) -> ConversationMessageResponse:
    """
    Converts a stored ORM message into a safe API
    response.
    """

    details = message.details_json or {}

    raw_sources = details.get(
        "sources",
        [],
    )

    sources = [
        SourceReferenceResponse.model_validate(
            source
        )
        for source in raw_sources
        if isinstance(source, dict)
    ]

    raw_guardrails = details.get(
        "guardrails"
    )

    guardrails = (
        GuardrailStatusResponse
        .model_validate(raw_guardrails)
        if isinstance(
            raw_guardrails,
            dict,
        )
        else None
    )

    raw_usage = details.get("usage")

    usage = (
        TokenUsageResponse.model_validate(
            raw_usage
        )
        if isinstance(raw_usage, dict)
        else None
    )

    return ConversationMessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        request_id=message.request_id,
        status=message.status,
        abstained=bool(
            details.get(
                "abstained",
                False,
            )
        ),
        model_called=bool(
            details.get(
                "model_called",
                False,
            )
        ),
        citations_used=list(
            details.get(
                "citations_used",
                [],
            )
        ),
        sources=sources,
        evidence_count=int(
            details.get(
                "evidence_count",
                0,
            )
        ),
        guardrails=guardrails,
        usage=usage,
        feedback=build_feedback_response(
            message
        ),
    )


@router.get(
    "",
    response_model=list[
        ConversationSummaryResponse
    ],
)
def list_conversations(
    principal: CurrentPrincipalDependency,
    database_session: (
        DatabaseSessionDependency
    ),
    limit: int = Query(
        default=30,
        ge=1,
        le=100,
    ),
) -> list[ConversationSummaryResponse]:
    conversations = (
        conversation_store
        .list_conversations(
            database_session,
            principal=principal,
            limit=limit,
        )
    )

    return [
        ConversationSummaryResponse(
            id=conversation.id,
            title=conversation.title,
            created_at=(
                conversation.created_at
            ),
            updated_at=(
                conversation.updated_at
            ),
            message_count=len(
                conversation.messages
            ),
        )
        for conversation in conversations
    ]


@router.get(
    "/{conversation_id}",
    response_model=(
        ConversationDetailResponse
    ),
)
def get_conversation(
    principal: CurrentPrincipalDependency,
    database_session: (
        DatabaseSessionDependency
    ),
    conversation_id: str = Path(
        min_length=36,
        max_length=36,
    ),
) -> ConversationDetailResponse:
    try:
        conversation = (
            conversation_store
            .get_conversation(
                database_session,
                principal=principal,
                conversation_id=(
                    conversation_id
                ),
            )
        )

    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            build_message_response(
                message
            )
            for message
            in conversation.messages
        ],
    )