import pytest
from langchain_core.documents import Document

from app.config import EmbeddingSettings
from app.rag.embedding_service import (
    create_batches,
    embed_chunk_documents,
    embed_query_text,
)


class FakeEmbeddingClient:
    """
    Fake embedding model used by unit tests.

    It avoids network requests, API cost, and dependence
    on an OpenAI API key.
    """

    def __init__(
        self,
        *,
        dimensions: int,
    ) -> None:
        self.dimensions = dimensions
        self.document_batch_sizes: list[int] = []

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.document_batch_sizes.append(
            len(texts)
        )

        vectors = []

        for text_index, _ in enumerate(texts):
            vector_value = float(
                text_index + 1
            )

            vectors.append(
                [
                    vector_value
                    for _ in range(
                        self.dimensions
                    )
                ]
            )

        return vectors

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return [
            float(len(text))
            for _ in range(
                self.dimensions
            )
        ]


def create_settings(
    *,
    dimensions: int = 4,
    batch_size: int = 2,
) -> EmbeddingSettings:
    return EmbeddingSettings(
        api_key="test-api-key",
        model="fake-embedding-model",
        dimensions=dimensions,
        batch_size=batch_size,
        base_url=None,
    )


def create_chunk_document(
    *,
    chunk_index: int,
    pii_detected: bool = False,
    pii_masked: bool = False,
    embedding_status: str = "PENDING",
) -> Document:
    return Document(
        page_content=(
            f"Banking policy chunk "
            f"{chunk_index}."
        ),
        metadata={
            "chunk_id": (
                f"PAY-POL-1042:7.2:"
                f"chunk-{chunk_index:04d}"
            ),
            "chunk_index": chunk_index,
            "chunk_number": chunk_index + 1,
            "total_chunks": 5,
            "document_id": "PAY-POL-1042",
            "title": "Payment Policy",
            "version": "7.2",
            "allowed_roles": [
                "compliance_analyst",
                "payments_analyst",
            ],
            "allowed_regions": ["US"],
            "classification_rank": 2,
            "entitlement_key": (
                "department=payments-compliance"
                "|regions=US"
                "|classification=2"
                "|roles=compliance_analyst,"
                "payments_analyst"
            ),
            "retrieval_enabled": True,
            "pii_detected": pii_detected,
            "pii_masked": pii_masked,
            "record_type": "chunk",
            "embedding_status": (
                embedding_status
            ),
        },
    )


def test_create_batches() -> None:
    documents = [
        create_chunk_document(
            chunk_index=index
        )
        for index in range(5)
    ]

    batches = create_batches(
        documents,
        batch_size=2,
    )

    assert len(batches) == 3

    assert [
        len(batch)
        for batch in batches
    ] == [2, 2, 1]


def test_embed_chunk_documents() -> None:
    documents = [
        create_chunk_document(
            chunk_index=index
        )
        for index in range(3)
    ]

    settings = create_settings(
        dimensions=4,
        batch_size=2,
    )

    fake_client = FakeEmbeddingClient(
        dimensions=4
    )

    embedded_chunks = (
        embed_chunk_documents(
            documents,
            embedding_client=fake_client,
            settings=settings,
        )
    )

    assert len(embedded_chunks) == 3

    assert (
        fake_client.document_batch_sizes
        == [2, 1]
    )

    for embedded_chunk in embedded_chunks:
        assert len(
            embedded_chunk.vector
        ) == 4

        assert (
            embedded_chunk.document.metadata[
                "embedding_status"
            ]
            == "COMPLETED"
        )

        assert (
            embedded_chunk.document.metadata[
                "embedding_model"
            ]
            == "fake-embedding-model"
        )

        assert (
            embedded_chunk.document.metadata[
                "embedding_dimensions"
            ]
            == 4
        )

        # The large vector is kept outside metadata.
        assert (
            "embedding_vector"
            not in embedded_chunk.document.metadata
        )


def test_unmasked_pii_is_rejected() -> None:
    document = create_chunk_document(
        chunk_index=0,
        pii_detected=True,
        pii_masked=False,
    )

    settings = create_settings()

    fake_client = FakeEmbeddingClient(
        dimensions=4
    )

    with pytest.raises(
        ValueError,
        match="has not been masked",
    ):
        embed_chunk_documents(
            [document],
            embedding_client=fake_client,
            settings=settings,
        )


def test_completed_chunk_is_not_embedded_again() -> None:
    document = create_chunk_document(
        chunk_index=0,
        embedding_status="COMPLETED",
    )

    settings = create_settings()

    fake_client = FakeEmbeddingClient(
        dimensions=4
    )

    with pytest.raises(
        ValueError,
        match="status PENDING",
    ):
        embed_chunk_documents(
            [document],
            embedding_client=fake_client,
            settings=settings,
        )


def test_dimension_mismatch_is_rejected() -> None:
    document = create_chunk_document(
        chunk_index=0
    )

    settings = create_settings(
        dimensions=4
    )

    fake_client = FakeEmbeddingClient(
        dimensions=3
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        embed_chunk_documents(
            [document],
            embedding_client=fake_client,
            settings=settings,
        )


def test_embed_query_text() -> None:
    fake_client = FakeEmbeddingClient(
        dimensions=4
    )

    query_vector = embed_query_text(
        "What are the payment review rules?",
        embedding_client=fake_client,
        expected_dimensions=4,
    )

    assert len(query_vector) == 4


def test_empty_query_is_rejected() -> None:
    fake_client = FakeEmbeddingClient(
        dimensions=4
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        embed_query_text(
            "   ",
            embedding_client=fake_client,
            expected_dimensions=4,
        )