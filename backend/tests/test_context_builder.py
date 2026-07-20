import pytest
from langchain_core.documents import Document

from app.config import PromptSettings
from app.rag.context_builder import (
    build_prompt_package,
    truncate_text_to_token_limit,
)
from app.rag.faiss_store import (
    SearchAccessContext,
)
from app.rag.retrieval_service import (
    RetrievalResponse,
    RetrievedChunk,
)


def create_access_context(
) -> SearchAccessContext:
    return SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )


def create_retrieved_chunk(
    *,
    chunk_id: str,
    title: str,
    document_id: str,
    content: str,
    rank: int,
    allowed_roles: list[str] | None = None,
) -> RetrievedChunk:
    document = Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_number": rank,
            "total_chunks": 3,
            "document_id": document_id,
            "title": title,
            "version": "1.0",
            "source": (
                f"local://{document_id}.txt"
            ),
            "retrieval_enabled": True,
            "allowed_roles": (
                allowed_roles
                if allowed_roles is not None
                else ["compliance_analyst"]
            ),
            "allowed_regions": ["US"],
            "classification_rank": 2,
        },
    )

    return RetrievedChunk(
        document=document,
        rank=rank,
        rerank_score=0.9,
        reranker_backend=(
            "local_feature"
        ),
        reranker_model=(
            "local-feature-reranker-v1"
        ),
        hybrid_rank=rank,
        hybrid_score=0.03,
        vector_rank=rank,
        vector_score=0.8,
        keyword_rank=rank,
        keyword_score=1.5,
        matched_by=(
            "semantic",
            "keyword",
        ),
        citation=(
            f"{title} "
            f"({document_id}, version 1.0, "
            f"chunk {rank}/3)"
        ),
    )


def create_settings(
    *,
    max_context_tokens: int = 1_000,
    max_context_chunks: int = 5,
) -> PromptSettings:
    return PromptSettings(
        max_context_tokens=max_context_tokens,
        max_context_chunks=max_context_chunks,
        answer_max_tokens=500,
        token_encoding="cl100k_base",
        minimum_evidence_chunks=1,
    )


def test_prompt_package_contains_citations() -> None:
    access_context = create_access_context()

    first_result = create_retrieved_chunk(
        chunk_id="payment-chunk",
        title="Payment Review Policy",
        document_id="PAY-POL-1042",
        content=(
            "Enhanced verification is required for "
            "high-risk international payments."
        ),
        rank=1,
    )

    response = RetrievalResponse(
        query=(
            "What verification is required for "
            "high-risk international payments?"
        ),
        access_context=access_context,
        results=[first_result],
    )

    prompt_package = build_prompt_package(
        response,
        settings=create_settings(),
    )

    assert prompt_package.evidence_count == 1

    assert (
        prompt_package.should_abstain
        is False
    )

    assert (
        "[SOURCE 1]"
        in prompt_package.context_text
    )

    assert (
        "PAY-POL-1042"
        in prompt_package.context_text
    )

    assert (
        "[SOURCE 1]"
        in prompt_package.user_prompt
        or "source labels"
        in prompt_package.user_prompt.lower()
    )


def test_empty_evidence_creates_abstention_prompt(
) -> None:
    response = RetrievalResponse(
        query="Unknown policy question",
        access_context=(
            create_access_context()
        ),
        results=[],
    )

    prompt_package = build_prompt_package(
        response,
        settings=create_settings(),
    )

    assert prompt_package.evidence_count == 0

    assert (
        prompt_package.should_abstain
        is True
    )

    assert (
        "No authorized policy evidence"
        in prompt_package.user_prompt
    )


def test_maximum_context_chunks_is_enforced(
) -> None:
    results = [
        create_retrieved_chunk(
            chunk_id=f"chunk-{index}",
            title=f"Policy {index}",
            document_id=f"POL-{index}",
            content=(
                "Example banking policy evidence."
            ),
            rank=index + 1,
        )
        for index in range(5)
    ]

    response = RetrievalResponse(
        query="Banking policy",
        access_context=(
            create_access_context()
        ),
        results=results,
    )

    prompt_package = build_prompt_package(
        response,
        settings=create_settings(
            max_context_chunks=2
        ),
    )

    assert prompt_package.evidence_count == 2

    assert len(
        prompt_package.citations
    ) == 2


def test_context_token_budget_is_enforced() -> None:
    long_content = " ".join(
        [
            "Enhanced verification is required."
            for _ in range(500)
        ]
    )

    result = create_retrieved_chunk(
        chunk_id="large-chunk",
        title="Large Payment Policy",
        document_id="PAY-POL-LARGE",
        content=long_content,
        rank=1,
    )

    response = RetrievalResponse(
        query="Payment verification",
        access_context=(
            create_access_context()
        ),
        results=[result],
    )

    settings = create_settings(
        max_context_tokens=150
    )

    prompt_package = build_prompt_package(
        response,
        settings=settings,
    )

    assert (
        prompt_package.context_token_count
        <= settings.max_context_tokens
    )


def test_unauthorized_evidence_fails_closed(
) -> None:
    unauthorized_result = create_retrieved_chunk(
        chunk_id="security-chunk",
        title="Security Investigation Policy",
        document_id="SEC-POL-100",
        content=(
            "Restricted investigation procedure."
        ),
        rank=1,
        allowed_roles=[
            "security_investigator"
        ],
    )

    response = RetrievalResponse(
        query="Security investigation procedure",
        access_context=(
            create_access_context()
        ),
        results=[unauthorized_result],
    )

    with pytest.raises(
        ValueError,
        match="Unauthorized evidence",
    ):
        build_prompt_package(
            response,
            settings=create_settings(),
        )


def test_text_truncation() -> None:
    text = " ".join(
        [
            "international payment review"
            for _ in range(100)
        ]
    )

    truncated_text = (
        truncate_text_to_token_limit(
            text,
            maximum_tokens=20,
            encoding_name="cl100k_base",
        )
    )

    assert len(truncated_text) < len(text)

    assert (
        "Content truncated"
        in truncated_text
    )