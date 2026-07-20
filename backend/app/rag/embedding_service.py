from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import EmbeddingSettings


REQUIRED_EMBEDDING_METADATA = (
    "chunk_id",
    "document_id",
    "version",
    "allowed_roles",
    "allowed_regions",
    "classification_rank",
    "entitlement_key",
    "retrieval_enabled",
    "record_type",
    "embedding_status",
)


class EmbeddingClient(Protocol):
    """
    Interface required by our embedding service.

    OpenAIEmbeddings follows this interface. Using a protocol
    also lets our unit tests provide a fake embedding model
    without making real API calls.
    """

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        ...

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """
    Connects one LangChain chunk Document to its numerical
    embedding vector.

    The vector is intentionally not inserted into the
    LangChain metadata dictionary.
    """

    document: Document
    vector: list[float]


def build_embedding_client(
    settings: EmbeddingSettings,
) -> OpenAIEmbeddings:
    """
    Creates the LangChain OpenAI embedding client.

    Locally, the request goes to the approved OpenAI endpoint.

    In production, base_url can point to the organization's
    model gateway.
    """

    client_arguments = {
        "api_key": settings.api_key,
        "model": settings.model,
        "dimensions": settings.dimensions,
    }

    if settings.base_url is not None:
        client_arguments["base_url"] = (
            settings.base_url
        )

    return OpenAIEmbeddings(
        **client_arguments
    )


def validate_chunk_for_embedding(
    document: Document,
) -> None:
    """
    Verifies that one chunk is safe and ready to be sent
    to the embedding model.
    """

    if not document.page_content.strip():
        raise ValueError(
            "Chunk content cannot be empty."
        )

    missing_fields = [
        field_name
        for field_name
        in REQUIRED_EMBEDDING_METADATA
        if field_name not in document.metadata
    ]

    if missing_fields:
        missing_fields_text = ", ".join(
            missing_fields
        )

        raise ValueError(
            "Chunk is missing required embedding "
            f"metadata: {missing_fields_text}"
        )

    chunk_id = str(
        document.metadata["chunk_id"]
    )

    if (
        document.metadata.get("record_type")
        != "chunk"
    ):
        raise ValueError(
            f"Document '{chunk_id}' is not a chunk."
        )

    if (
        document.metadata.get(
            "retrieval_enabled"
        )
        is not True
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' is not enabled "
            "for retrieval."
        )

    if not document.metadata.get(
        "allowed_roles"
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' has no allowed roles."
        )

    if not document.metadata.get(
        "allowed_regions"
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' has no allowed regions."
        )

    if (
        document.metadata.get(
            "classification_rank"
        )
        is None
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' has no "
            "classification rank."
        )

    if not document.metadata.get(
        "entitlement_key"
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' has no "
            "entitlement key."
        )

    pii_detected = (
        document.metadata.get(
            "pii_detected"
        )
        is True
    )

    pii_masked = (
        document.metadata.get(
            "pii_masked"
        )
        is True
    )

    if pii_detected and not pii_masked:
        raise ValueError(
            f"Chunk '{chunk_id}' contains detected "
            "PII that has not been masked."
        )

    if (
        document.metadata.get(
            "embedding_status"
        )
        != "PENDING"
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' does not have "
            "embedding status PENDING."
        )


def create_batches(
    items: Sequence[Document],
    *,
    batch_size: int,
) -> list[list[Document]]:
    """
    Divides chunk documents into smaller embedding batches.

    Example:

    Five chunks with batch size two become:

    Batch 1: chunks 1 and 2
    Batch 2: chunks 3 and 4
    Batch 3: chunk 5
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    return [
        list(
            items[
                start_index:
                start_index + batch_size
            ]
        )
        for start_index in range(
            0,
            len(items),
            batch_size,
        )
    ]


def validate_embedding_vector(
    *,
    vector: list[float],
    expected_dimensions: int,
    chunk_id: str,
) -> None:
    """
    Validates the shape and contents of one embedding
    vector.
    """

    if not vector:
        raise ValueError(
            f"Embedding model returned an empty "
            f"vector for chunk '{chunk_id}'."
        )

    if len(vector) != expected_dimensions:
        raise ValueError(
            f"Embedding dimension mismatch for chunk "
            f"'{chunk_id}'. Expected "
            f"{expected_dimensions}, received "
            f"{len(vector)}."
        )

    if not all(
        isinstance(value, int | float)
        for value in vector
    ):
        raise ValueError(
            f"Embedding vector for chunk "
            f"'{chunk_id}' contains a non-numeric "
            "value."
        )


def mark_chunk_as_embedded(
    *,
    document: Document,
    settings: EmbeddingSettings,
    vector_dimensions: int,
    embedded_at: str,
) -> Document:
    """
    Returns a new chunk Document with embedding-processing
    metadata.

    The vector itself is not placed inside metadata.
    """

    updated_metadata = {
        **document.metadata,
        "embedding_status": "COMPLETED",
        "embedding_provider": "openai",
        "embedding_model": settings.model,
        "embedding_dimensions": (
            vector_dimensions
        ),
        "embedding_pipeline_version": "1.0",
        "embedded_at": embedded_at,
    }

    return Document(
        page_content=document.page_content,
        metadata=updated_metadata,
    )


def embed_chunk_documents(
    documents: list[Document],
    *,
    embedding_client: EmbeddingClient,
    settings: EmbeddingSettings,
) -> list[EmbeddedChunk]:
    """
    Generates embeddings for chunk Documents in batches.

    The position of every returned vector must correspond
    to the position of its input chunk.
    """

    if not documents:
        return []

    for document in documents:
        validate_chunk_for_embedding(
            document
        )

    batches = create_batches(
        documents,
        batch_size=settings.batch_size,
    )

    embedded_chunks: list[
        EmbeddedChunk
    ] = []

    for batch in batches:
        batch_texts = [
            document.page_content
            for document in batch
        ]

        batch_vectors = (
            embedding_client.embed_documents(
                batch_texts
            )
        )

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "Embedding model returned a different "
                "number of vectors than the number of "
                "submitted chunks."
            )

        embedded_at = datetime.now(
            timezone.utc
        ).isoformat()

        for document, vector in zip(
            batch,
            batch_vectors,
            strict=True,
        ):
            chunk_id = str(
                document.metadata["chunk_id"]
            )

            validate_embedding_vector(
                vector=vector,
                expected_dimensions=(
                    settings.dimensions
                ),
                chunk_id=chunk_id,
            )

            embedded_document = (
                mark_chunk_as_embedded(
                    document=document,
                    settings=settings,
                    vector_dimensions=len(vector),
                    embedded_at=embedded_at,
                )
            )

            embedded_chunks.append(
                EmbeddedChunk(
                    document=embedded_document,
                    vector=vector,
                )
            )

    return embedded_chunks


def embed_query_text(
    query: str,
    *,
    embedding_client: EmbeddingClient,
    expected_dimensions: int,
) -> list[float]:
    """
    Converts a user query into an embedding vector.

    The query vector will later be compared against stored
    document vectors during similarity search.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_vector = (
        embedding_client.embed_query(
            cleaned_query
        )
    )

    validate_embedding_vector(
        vector=query_vector,
        expected_dimensions=(
            expected_dimensions
        ),
        chunk_id="QUERY",
    )

    return query_vector