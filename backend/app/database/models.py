from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def utc_now() -> datetime:
    """
    Returns the current UTC time.
    """

    return datetime.now(
        timezone.utc
    )


class Base(DeclarativeBase):
    """
    Base class for SQLAlchemy ORM models.

    Alembic imports Base.metadata from this module to
    compare the declared models with the database
    schema.
    """


class Conversation(Base):
    """
    One authenticated employee conversation.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    user_subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    username: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        index=True,
    )

    messages: Mapped[
        list["ConversationMessage"]
    ] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by=(
            "ConversationMessage.created_at"
        ),
    )

    __table_args__ = (
        Index(
            "ix_conversations_user_updated",
            "user_subject",
            "updated_at",
        ),
    )


class ConversationMessage(Base):
    """
    One user or assistant message.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    request_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    status: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    details_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    conversation: Mapped[
        "Conversation"
    ] = relationship(
        back_populates="messages"
    )

    feedback: Mapped[
        "MessageFeedback | None"
    ] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (
        Index(
            "ix_messages_conversation_created",
            "conversation_id",
            "created_at",
        ),
    )


class MessageFeedback(Base):
    """
    Employee feedback for an assistant message.
    """

    __tablename__ = "message_feedback"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "conversation_messages.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "conversations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    user_subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    rating: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    comment: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    message: Mapped[
        "ConversationMessage"
    ] = relationship(
        back_populates="feedback"
    )

    __table_args__ = (
        UniqueConstraint(
            "message_id",
            name=(
                "uq_message_feedback_message_id"
            ),
        ),
    )
