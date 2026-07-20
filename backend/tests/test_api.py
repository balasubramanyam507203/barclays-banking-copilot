from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.app_factory import create_app
from app.config import PromptSettings
from app.database.service import DatabaseService
from app.database.settings import DatabaseSettings
from app.rag.faiss_store import SearchAccessContext
from app.rag.generation_service import (
    GroundedAnswerResult,
)
from app.rag.retrieval_service import (
    RetrievalResponse,
    RetrievedChunk,
)
from app.security.authentication import (
    AuthenticatedPrincipal,
    AuthenticationError,
)


class FakeVectorStore:
    """
    Small vector-store replacement for API tests.
    """

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents

    @property
    def document_count(self) -> int:
        return len(self.documents)


class FakeAuthenticationService:
    """
    Validates deterministic bearer tokens during API
    tests without issuing or verifying real JWTs.
    """

    def verify_access_token(
        self,
        token: str,
    ) -> AuthenticatedPrincipal:
        if token == "valid-compliance-token":
            return AuthenticatedPrincipal(
                subject="test-compliance-user",
                username="compliance.analyst",
                role="compliance_analyst",
                region="US",
                clearance_rank=2,
                groups=(
                    "ComplianceAnalysts",
                ),
            )

        if token == "valid-support-token":
            return AuthenticatedPrincipal(
                subject="test-support-user",
                username="customer.support",
                role="customer_support",
                region="US",
                clearance_rank=1,
                groups=(
                    "CustomerSupport",
                ),
            )

        raise AuthenticationError(
            "The access token is invalid."
        )


class FakeRetrievalService:
    """
    Returns one deterministic policy chunk.
    """

    def __init__(
        self,
        document: Document,
    ) -> None:
        self.document = document

    def retrieve(
        self,
        query: str,
        *,
        access_context: SearchAccessContext,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        result = RetrievedChunk(
            document=self.document,
            rank=1,
            rerank_score=0.95,
            reranker_backend=(
                "fake-reranker"
            ),
            reranker_model=(
                "fake-reranker-model"
            ),
            hybrid_rank=1,
            hybrid_score=0.03,
            vector_rank=1,
            vector_score=0.90,
            keyword_rank=1,
            keyword_score=2.50,
            matched_by=(
                "semantic",
                "keyword",
            ),
            citation=(
                "Payment Review Policy "
                "(PAY-POL-1042, version 1.0, "
                "chunk 1/1)"
            ),
        )

        return RetrievalResponse(
            query=query,
            access_context=access_context,
            results=[result],
        )


class FakeGenerationService:
    """
    Returns one deterministic guardrail-approved answer.
    """

    def generate(
        self,
        prompt_package,
    ) -> GroundedAnswerResult:
        answer = (
            "High-risk international payments require "
            "enhanced verification before approval "
            "[SOURCE 1].\n\n"
            "Sources:\n"
            "- [SOURCE 1]"
        )

        return GroundedAnswerResult(
            answer=answer,
            raw_answer=answer,
            model_called=True,
            model_name=(
                "fake-generation-model"
            ),
            abstained=False,
            citation_validation_passed=True,
            post_generation_guardrails_passed=True,
            citations_used=(
                "SOURCE 1",
            ),
            claims_checked=1,
            supported_claims=1,
            validation_errors=(),
            guardrail_errors=(),
            input_tokens=200,
            output_tokens=40,
            total_tokens=240,
            finish_reason="stop",
        )


def create_policy_document() -> Document:
    """
    Creates one indexed policy chunk that is available
    only to compliance analysts.
    """

    return Document(
        page_content=(
            "High-risk international payments require "
            "enhanced verification before approval."
        ),
        metadata={
            "chunk_id": "payment-chunk",
            "chunk_number": 1,
            "total_chunks": 1,
            "document_id": "PAY-POL-1042",
            "title": (
                "Payment Review Policy"
            ),
            "version": "1.0",
            "source": (
                "local://PAY-POL-1042.txt"
            ),
            "retrieval_enabled": True,
            "allowed_roles": [
                "compliance_analyst",
            ],
            "allowed_regions": [
                "US",
            ],
            "classification_rank": 2,
        },
    )


def build_fake_services(
    database_path: Path,
):
    """
    Creates application services without external API
    calls.

    A temporary SQLite database is used because the
    current chat endpoint persists conversations and
    messages.

    Redis caching is disabled during these API tests by
    setting cache_service to None.
    """

    document = create_policy_document()

    database_settings = DatabaseSettings(
        url=f"sqlite:///{database_path}",
        echo=False,
        auto_create_schema=True,
    )

    database_service = DatabaseService(
        database_settings
    )

    return SimpleNamespace(
        retrieval_service=(
            FakeRetrievalService(
                document
            )
        ),
        generation_service=(
            FakeGenerationService()
        ),
        authentication_service=(
            FakeAuthenticationService()
        ),
        database_service=(
            database_service
        ),
        cache_service=None,
        prompt_settings=PromptSettings(
            max_context_tokens=1_000,
            max_context_chunks=5,
            answer_max_tokens=500,
            token_encoding="cl100k_base",
            minimum_evidence_chunks=1,
        ),
        vector_store=FakeVectorStore(
            [document]
        ),
        embedding_model=(
            "fake-embedding-model"
        ),
        generation_model=(
            "fake-generation-model"
        ),
        reranker_backend=(
            "fake-reranker"
        ),
        auth_mode="local_jwt",
        started_at=datetime.now(
            timezone.utc
        ),
    )


@pytest.fixture
def client(
    tmp_path: Path,
) -> Iterator[TestClient]:
    """
    Starts the FastAPI lifespan with isolated services
    and a separate temporary database for each test.
    """

    database_path = (
        tmp_path / "test_policy_copilot.db"
    )

    application = create_app(
        service_factory=lambda: (
            build_fake_services(
                database_path
            )
        )
    )

    with TestClient(
        application
    ) as test_client:
        yield test_client


@pytest.fixture
def compliance_headers() -> dict[str, str]:
    return {
        "Authorization": (
            "Bearer valid-compliance-token"
        ),
    }


@pytest.fixture
def support_headers() -> dict[str, str]:
    return {
        "Authorization": (
            "Bearer valid-support-token"
        ),
    }


def test_liveness(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/health/live"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert (
        body["service"]
        == "enterprise-banking-policy-copilot"
    )


def test_readiness(
    client: TestClient,
) -> None:
    response = client.get(
        "/api/v1/health/ready"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["indexed_chunks"] == 1

    assert (
        body["embedding_model"]
        == "fake-embedding-model"
    )

    assert (
        body["generation_model"]
        == "fake-generation-model"
    )

    assert (
        body["reranker_backend"]
        == "fake-reranker"
    )


def test_chat_returns_guarded_answer(
    client: TestClient,
    compliance_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=compliance_headers,
        json={
            "question": (
                "What verification is required for "
                "high-risk international payments?"
            ),
            "conversation_id": None,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "answered"
    assert body["abstained"] is False
    assert body["model_called"] is True

    assert isinstance(
        body["conversation_id"],
        str,
    )

    assert len(
        body["conversation_id"]
    ) == 36

    assert isinstance(
        body["user_message_id"],
        str,
    )

    assert len(
        body["user_message_id"]
    ) == 36

    assert isinstance(
        body["assistant_message_id"],
        str,
    )

    assert len(
        body["assistant_message_id"]
    ) == 36

    assert (
        body["guardrails"]
        ["citation_validation_passed"]
        is True
    )

    assert (
        body["guardrails"]
        ["post_generation_guardrails_passed"]
        is True
    )

    assert body["citations_used"] == [
        "SOURCE 1"
    ]

    assert len(body["sources"]) == 1

    assert (
        body["sources"][0]["document_id"]
        == "PAY-POL-1042"
    )

    assert (
        body["usage"]["total_tokens"]
        == 240
    )


def test_chat_requires_bearer_token(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "question": (
                "What is the payment policy?"
            ),
            "conversation_id": None,
        },
    )

    assert response.status_code == 401

    assert (
        response.headers[
            "www-authenticate"
        ]
        == "Bearer"
    )


def test_chat_rejects_invalid_token(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers={
            "Authorization": (
                "Bearer invalid-token"
            ),
        },
        json={
            "question": (
                "What is the payment policy?"
            ),
            "conversation_id": None,
        },
    )

    assert response.status_code == 401

    assert (
        response.json()["detail"]
        == "The access token is invalid."
    )


def test_chat_rejects_empty_question(
    client: TestClient,
    compliance_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=compliance_headers,
        json={
            "question": "   ",
            "conversation_id": None,
        },
    )

    assert response.status_code == 422


def test_authorized_source_is_returned(
    client: TestClient,
    compliance_headers: dict[str, str],
) -> None:
    response = client.get(
        "/api/v1/sources/PAY-POL-1042",
        headers=compliance_headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["document_id"]
        == "PAY-POL-1042"
    )

    assert (
        body["title"]
        == "Payment Review Policy"
    )

    assert len(body["chunks"]) == 1

    assert (
        "enhanced verification"
        in body["chunks"][0]["content"]
    )


def test_unauthorized_source_returns_404(
    client: TestClient,
    support_headers: dict[str, str],
) -> None:
    response = client.get(
        "/api/v1/sources/PAY-POL-1042",
        headers=support_headers,
    )

    assert response.status_code == 404

    assert (
        response.json()["detail"]
        == "Source document not found."
    )