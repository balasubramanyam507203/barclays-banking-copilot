from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.api.dependencies import (
    CurrentPrincipalDependency,
    DatabaseSessionDependency,
)
from app.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
)
from app.database.conversation_store import (
    ConversationStore,
    MessageNotFoundError,
)


router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
)


conversation_store = ConversationStore()


@router.post(
    "",
    response_model=FeedbackResponse,
)
def submit_feedback(
    payload: FeedbackRequest,
    principal: CurrentPrincipalDependency,
    database_session: (
        DatabaseSessionDependency
    ),
) -> FeedbackResponse:
    """
    Saves or updates feedback for one authorized
    assistant message.
    """

    try:
        feedback = (
            conversation_store.save_feedback(
                database_session,
                principal=principal,
                message_id=str(
                    payload.message_id
                ),
                rating=payload.rating,
                comment=payload.comment,
            )
        )

    except MessageNotFoundError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=str(error),
        ) from error

    return FeedbackResponse(
        id=feedback.id,
        message_id=feedback.message_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )