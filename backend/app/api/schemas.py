from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class DevelopmentLoginRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    profile: Literal[
        "compliance_analyst",
        "customer_support",
        "security_investigator",
    ]

    password: str = Field(
        min_length=1,
        max_length=200,
    )


class CurrentUserResponse(BaseModel):
    subject: str
    username: str

    role: str
    region: str
    clearance_rank: int

    groups: list[str]


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int

    user: CurrentUserResponse


class ChatRequest(BaseModel):
    """
    A conversation ID is optional.

    No conversation ID:
        Start a new conversation.

    Existing conversation ID:
        Continue that authorized conversation.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    question: str = Field(
        min_length=1,
        max_length=2_000,
    )

    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(
        cls,
        value: str,
    ) -> str:
        normalized_question = " ".join(
            value.split()
        )

        if not normalized_question:
            raise ValueError(
                "Question cannot be empty."
            )

        return normalized_question


class SourceReferenceResponse(BaseModel):
    label: str

    document_id: str
    title: str
    version: str

    chunk_id: str
    source: str
    citation: str


class TokenUsageResponse(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GuardrailStatusResponse(BaseModel):
    citation_validation_passed: bool
    post_generation_guardrails_passed: bool

    claims_checked: int
    supported_claims: int


class ChatResponse(BaseModel):
    request_id: str

    conversation_id: str
    user_message_id: str
    assistant_message_id: str

    status: Literal[
        "answered",
        "abstained",
    ]

    answer: str
    abstained: bool
    model_called: bool

    citations_used: list[str]
    sources: list[SourceReferenceResponse]

    evidence_count: int

    guardrails: GuardrailStatusResponse
    usage: TokenUsageResponse


class HealthResponse(BaseModel):
    status: Literal[
        "ok",
        "ready",
    ]

    service: str
    version: str

    checked_at: datetime
    started_at: datetime | None = None

    indexed_chunks: int | None = None

    embedding_model: str | None = None
    generation_model: str | None = None
    reranker_backend: str | None = None


class SourceChunkResponse(BaseModel):
    chunk_id: str
    chunk_number: int
    total_chunks: int

    content: str


class SourceDocumentResponse(BaseModel):
    document_id: str
    title: str
    version: str
    source: str

    chunks: list[SourceChunkResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    message_id: UUID

    rating: Literal[
        "helpful",
        "not_helpful",
    ]

    comment: str | None = Field(
        default=None,
        max_length=2_000,
    )


class FeedbackResponse(BaseModel):
    id: str
    message_id: str
    rating: Literal[
        "helpful",
        "not_helpful",
    ]

    comment: str | None

    created_at: datetime
    updated_at: datetime


class ConversationSummaryResponse(BaseModel):
    id: str
    title: str

    created_at: datetime
    updated_at: datetime

    message_count: int


class ConversationMessageResponse(BaseModel):
    id: str
    role: Literal[
        "user",
        "assistant",
    ]

    content: str
    created_at: datetime

    request_id: str | None = None
    status: Literal[
        "answered",
        "abstained",
    ] | None = None

    abstained: bool = False
    model_called: bool = False

    citations_used: list[str] = []
    sources: list[
        SourceReferenceResponse
    ] = []

    evidence_count: int = 0

    guardrails: (
        GuardrailStatusResponse | None
    ) = None

    usage: TokenUsageResponse | None = None

    feedback: FeedbackResponse | None = None


class ConversationDetailResponse(BaseModel):
    id: str
    title: str

    created_at: datetime
    updated_at: datetime

    messages: list[
        ConversationMessageResponse
    ]