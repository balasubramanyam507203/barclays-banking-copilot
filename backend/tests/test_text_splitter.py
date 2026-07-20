import pytest
from langchain_core.documents import Document

from app.rag.text_splitter import (
    ChunkingConfig,
    count_tokens,
    split_parent_document,
    split_retrievable_documents,
)


def create_parent_document(
    *,
    document_id: str = "PAY-POL-1042",
    retrieval_enabled: bool = True,
    content: str,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "parent_document_key": (
                f"{document_id}:7.2"
            ),
            "document_id": document_id,
            "title": "Payment Review Policy",
            "version": "7.2",
            "status": "ACTIVE",
            "effective_date": "2026-01-01",
            "department": "Payments Compliance",
            "region": "US",
            "allowed_regions": ["US"],
            "classification": (
                "Internal Confidential"
            ),
            "classification_rank": 2,
            "allowed_roles": [
                "compliance_analyst",
                "payments_analyst",
            ],
            "entitlement_key": (
                "department=payments-compliance"
                "|regions=US"
                "|classification=2"
                "|roles=compliance_analyst,"
                "payments_analyst"
            ),
            "retrieval_enabled": (
                retrieval_enabled
            ),
            "source": (
                "local://sample_documents/"
                "payment_policy.txt"
            ),
            "content_hash": "a" * 64,
            "record_type": "parent_document",
        },
    )


def create_long_policy_text() -> str:
    paragraphs = []

    for section_number in range(1, 15):
        paragraph = (
            f"Section {section_number}: "
            "Enhanced Payment Review\n"
            "International payments involving "
            "high-risk destinations require enhanced "
            "review. The analyst must verify customer "
            "identity, payment purpose, source of funds, "
            "beneficiary information, and supporting "
            "documentation before approving the transfer."
        )

        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def test_count_tokens_returns_positive_number() -> None:
    token_count = count_tokens(
        "Enhanced review is required.",
        encoding_name="cl100k_base",
    )

    assert token_count > 0


def test_split_parent_document_creates_chunks() -> None:
    document = create_parent_document(
        content=create_long_policy_text()
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_parent_document(
        document,
        config=config,
    )

    assert len(chunks) > 1

    assert all(
        chunk.metadata["record_type"] == "chunk"
        for chunk in chunks
    )


def test_chunks_remain_within_token_limit() -> None:
    document = create_parent_document(
        content=create_long_policy_text()
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_parent_document(
        document,
        config=config,
    )

    for chunk in chunks:
        assert (
            chunk.metadata["chunk_token_count"]
            <= config.chunk_size_tokens
        )


def test_chunk_security_metadata_is_preserved() -> None:
    document = create_parent_document(
        content=create_long_policy_text()
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_parent_document(
        document,
        config=config,
    )

    first_chunk = chunks[0]

    assert first_chunk.metadata[
        "allowed_roles"
    ] == [
        "compliance_analyst",
        "payments_analyst",
    ]

    assert first_chunk.metadata[
        "allowed_regions"
    ] == ["US"]

    assert first_chunk.metadata[
        "classification_rank"
    ] == 2

    assert first_chunk.metadata[
        "retrieval_enabled"
    ] is True


def test_chunk_ids_are_unique() -> None:
    document = create_parent_document(
        content=create_long_policy_text()
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_parent_document(
        document,
        config=config,
    )

    chunk_ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )


def test_chunk_numbers_are_correct() -> None:
    document = create_parent_document(
        content=create_long_policy_text()
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_parent_document(
        document,
        config=config,
    )

    total_chunks = len(chunks)

    for index, chunk in enumerate(chunks):
        assert (
            chunk.metadata["chunk_index"]
            == index
        )

        assert (
            chunk.metadata["chunk_number"]
            == index + 1
        )

        assert (
            chunk.metadata["total_chunks"]
            == total_chunks
        )


def test_non_retrievable_document_is_rejected() -> None:
    document = create_parent_document(
        retrieval_enabled=False,
        content=create_long_policy_text(),
    )

    with pytest.raises(
        ValueError,
        match="retrieval-enabled",
    ):
        split_parent_document(document)


def test_split_multiple_documents() -> None:
    first_document = create_parent_document(
        document_id="PAY-POL-1042",
        content=create_long_policy_text(),
    )

    second_document = create_parent_document(
        document_id="CMP-POL-3015",
        content=create_long_policy_text(),
    )

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = split_retrievable_documents(
        [
            first_document,
            second_document,
        ],
        config=config,
    )

    document_ids = {
        chunk.metadata["document_id"]
        for chunk in chunks
    }

    assert document_ids == {
        "PAY-POL-1042",
        "CMP-POL-3015",
    }


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="smaller",
    ):
        ChunkingConfig(
            chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )