"""Create conversation, message, and feedback tables.

Revision ID: 20260720_0001
Revises:
Create Date: 2026-07-20
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260720_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Creates the initial conversation persistence
    schema.
    """

    op.create_table(
        "conversations",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "user_subject",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "username",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "title",
            sa.String(length=200),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "id"
        ),
    )

    op.create_index(
        "ix_conversations_updated_at",
        "conversations",
        [
            "updated_at",
        ],
        unique=False,
    )

    op.create_index(
        "ix_conversations_user_subject",
        "conversations",
        [
            "user_subject",
        ],
        unique=False,
    )

    op.create_index(
        "ix_conversations_user_updated",
        "conversations",
        [
            "user_subject",
            "updated_at",
        ],
        unique=False,
    )

    op.create_table(
        "conversation_messages",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            sa.String(length=36),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=30),
            nullable=True,
        ),
        sa.Column(
            "details_json",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            [
                "conversation_id",
            ],
            [
                "conversations.id",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id"
        ),
    )

    op.create_index(
        "ix_conversation_messages_conversation_id",
        "conversation_messages",
        [
            "conversation_id",
        ],
        unique=False,
    )

    op.create_index(
        "ix_conversation_messages_created_at",
        "conversation_messages",
        [
            "created_at",
        ],
        unique=False,
    )

    op.create_index(
        "ix_conversation_messages_request_id",
        "conversation_messages",
        [
            "request_id",
        ],
        unique=False,
    )

    op.create_index(
        "ix_messages_conversation_created",
        "conversation_messages",
        [
            "conversation_id",
            "created_at",
        ],
        unique=False,
    )

    op.create_table(
        "message_feedback",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "user_subject",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "rating",
            sa.String(length=30),
            nullable=False,
        ),
        sa.Column(
            "comment",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            [
                "conversation_id",
            ],
            [
                "conversations.id",
            ],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            [
                "message_id",
            ],
            [
                "conversation_messages.id",
            ],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id"
        ),
        sa.UniqueConstraint(
            "message_id",
            name=(
                "uq_message_feedback_message_id"
            ),
        ),
    )

    op.create_index(
        "ix_message_feedback_conversation_id",
        "message_feedback",
        [
            "conversation_id",
        ],
        unique=False,
    )

    op.create_index(
        "ix_message_feedback_user_subject",
        "message_feedback",
        [
            "user_subject",
        ],
        unique=False,
    )


def downgrade() -> None:
    """
    Removes the initial conversation persistence
    schema.
    """

    op.drop_index(
        "ix_message_feedback_user_subject",
        table_name="message_feedback",
    )

    op.drop_index(
        "ix_message_feedback_conversation_id",
        table_name="message_feedback",
    )

    op.drop_table(
        "message_feedback"
    )

    op.drop_index(
        "ix_messages_conversation_created",
        table_name="conversation_messages",
    )

    op.drop_index(
        "ix_conversation_messages_request_id",
        table_name="conversation_messages",
    )

    op.drop_index(
        "ix_conversation_messages_created_at",
        table_name="conversation_messages",
    )

    op.drop_index(
        "ix_conversation_messages_conversation_id",
        table_name="conversation_messages",
    )

    op.drop_table(
        "conversation_messages"
    )

    op.drop_index(
        "ix_conversations_user_updated",
        table_name="conversations",
    )

    op.drop_index(
        "ix_conversations_user_subject",
        table_name="conversations",
    )

    op.drop_index(
        "ix_conversations_updated_at",
        table_name="conversations",
    )

    op.drop_table(
        "conversations"
    )
