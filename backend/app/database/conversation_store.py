from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    Select,
    select,
)
from sqlalchemy.orm import (
    Session,
    selectinload,
)

from app.database.models import (
    Conversation,
    ConversationMessage,
    MessageFeedback,
)
from app.security.authentication import (
    AuthenticatedPrincipal,
)


class ConversationNotFoundError(Exception):
    """
    Raised for nonexistent or unauthorized
    conversations.
    """


class MessageNotFoundError(Exception):
    """
    Raised for nonexistent or unauthorized messages.
    """


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def build_conversation_title(
    question: str,
) -> str:
    """
    Builds a short title from the first question.
    """

    normalized_question = " ".join(
        question.split()
    )

    maximum_length = 80

    if len(normalized_question) <= maximum_length:
        return normalized_question

    return (
        normalized_question[
            :maximum_length - 1
        ].rstrip()
        + "…"
    )


def conversation_query(
) -> Select[tuple[Conversation]]:
    """
    Base query that eagerly loads messages and
    feedback.
    """

    return select(
        Conversation
    ).options(
        selectinload(
            Conversation.messages
        ).selectinload(
            ConversationMessage.feedback
        )
    )


class ConversationStore:
    """
    Conversation, message, and feedback persistence.
    """

    def create_or_get_conversation(
        self,
        session: Session,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: str | None,
        initial_question: str,
    ) -> Conversation:
        """
        Creates a conversation or returns an existing
        conversation owned by the authenticated user.
        """

        if conversation_id is not None:
            conversation = (
                session.execute(
                    conversation_query().where(
                        Conversation.id
                        == conversation_id,
                        Conversation.user_subject
                        == principal.subject,
                    )
                )
                .scalars()
                .first()
            )

            if conversation is None:
                raise ConversationNotFoundError(
                    "Conversation not found."
                )

            return conversation

        conversation = Conversation(
            id=str(uuid4()),
            user_subject=principal.subject,
            username=principal.username,
            title=build_conversation_title(
                initial_question
            ),
        )

        session.add(conversation)
        session.commit()
        session.refresh(conversation)

        return conversation

    def add_message(
        self,
        session: Session,
        *,
        conversation: Conversation,
        role: str,
        content: str,
        request_id: str | None = None,
        status: str | None = None,
        details_json: dict | None = None,
    ) -> ConversationMessage:
        """
        Persists one conversation message.
        """

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "Message role must be user or "
                "assistant."
            )

        message = ConversationMessage(
            id=str(uuid4()),
            conversation_id=(
                conversation.id
            ),
            role=role,
            content=content,
            request_id=request_id,
            status=status,
            details_json=(
                details_json or {}
            ),
        )

        conversation.updated_at = utc_now()

        session.add(message)
        session.add(conversation)
        session.commit()
        session.refresh(message)

        return message

    def list_conversations(
        self,
        session: Session,
        *,
        principal: AuthenticatedPrincipal,
        limit: int,
    ) -> list[Conversation]:
        """
        Lists only conversations owned by the user.
        """

        statement = (
            conversation_query()
            .where(
                Conversation.user_subject
                == principal.subject
            )
            .order_by(
                Conversation.updated_at.desc()
            )
            .limit(limit)
        )

        return list(
            session.execute(
                statement
            )
            .scalars()
            .unique()
            .all()
        )

    def get_conversation(
        self,
        session: Session,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: str,
    ) -> Conversation:
        """
        Returns one authorized conversation.
        """

        conversation = (
            session.execute(
                conversation_query().where(
                    Conversation.id
                    == conversation_id,
                    Conversation.user_subject
                    == principal.subject,
                )
            )
            .scalars()
            .unique()
            .first()
        )

        if conversation is None:
            raise ConversationNotFoundError(
                "Conversation not found."
            )

        return conversation

    def save_feedback(
        self,
        session: Session,
        *,
        principal: AuthenticatedPrincipal,
        message_id: str,
        rating: str,
        comment: str | None,
    ) -> MessageFeedback:
        """
        Creates or updates feedback for an assistant
        message owned by the authenticated user.
        """

        statement = (
            select(
                ConversationMessage
            )
            .join(
                Conversation,
                Conversation.id
                == ConversationMessage
                .conversation_id,
            )
            .options(
                selectinload(
                    ConversationMessage.feedback
                )
            )
            .where(
                ConversationMessage.id
                == message_id,
                ConversationMessage.role
                == "assistant",
                Conversation.user_subject
                == principal.subject,
            )
        )

        message = (
            session.execute(statement)
            .scalars()
            .first()
        )

        if message is None:
            raise MessageNotFoundError(
                "Assistant message not found."
            )

        normalized_comment = (
            comment.strip()
            if comment is not None
            else None
        )

        if normalized_comment == "":
            normalized_comment = None

        feedback = message.feedback

        if feedback is None:
            feedback = MessageFeedback(
                id=str(uuid4()),
                message_id=message.id,
                conversation_id=(
                    message.conversation_id
                ),
                user_subject=(
                    principal.subject
                ),
                rating=rating,
                comment=normalized_comment,
            )

            session.add(feedback)

        else:
            feedback.rating = rating
            feedback.comment = normalized_comment
            feedback.updated_at = utc_now()

            session.add(feedback)

        session.commit()
        session.refresh(feedback)

        return feedback