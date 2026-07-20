from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.rag.embedding_service import (
    EmbeddedChunk,
)
from app.rag.faiss_store import (
    LocalFaissStore,
    SearchAccessContext,
)


def create_embedded_chunk(
    *,
    chunk_id: str,
    vector: list[float],
    allowed_roles: list[str],
    allowed_regions: list[str] | None = None,
    classification_rank: int = 2,
    content: str = "Example banking policy.",
) -> EmbeddedChunk:
    document = Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "TEST-POL-100",
            "title": "Test Banking Policy",
            "version": "1.0",
            "record_type": "chunk",
            "embedding_status": "COMPLETED",
            "embedding_model": (
                "fake-embedding-model"
            ),
            "embedding_dimensions": len(
                vector
            ),
            "retrieval_enabled": True,
            "allowed_roles": allowed_roles,
            "allowed_regions": (
                allowed_regions
                if allowed_regions is not None
                else ["US"]
            ),
            "classification_rank": (
                classification_rank
            ),
            "entitlement_key": (
                "department=compliance"
                "|regions=US"
                "|classification=2"
            ),
            "pii_detected": False,
            "pii_masked": False,
            "source": "local://test-policy.txt",
        },
    )

    return EmbeddedChunk(
        document=document,
        vector=vector,
    )


def create_test_store() -> LocalFaissStore:
    compliance_chunk = create_embedded_chunk(
        chunk_id="chunk-compliance",
        vector=[1.0, 0.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
        content=(
            "Enhanced review is required for "
            "high-risk payments."
        ),
    )

    support_chunk = create_embedded_chunk(
        chunk_id="chunk-support",
        vector=[0.0, 1.0, 0.0],
        allowed_roles=[
            "customer_support"
        ],
        content=(
            "Customer complaints must be "
            "acknowledged promptly."
        ),
    )

    restricted_chunk = create_embedded_chunk(
        chunk_id="chunk-restricted",
        vector=[0.9, 0.1, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
        classification_rank=3,
        content=(
            "Restricted investigation "
            "procedure."
        ),
    )

    return LocalFaissStore.from_embedded_chunks(
        [
            compliance_chunk,
            support_chunk,
            restricted_chunk,
        ]
    )


def test_build_faiss_store() -> None:
    store = create_test_store()

    assert store.document_count == 3
    assert store.dimensions == 3
    assert store.index.ntotal == 3


def test_similarity_search_returns_authorized_result() -> None:
    store = create_test_store()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = store.search_by_vector(
        [1.0, 0.0, 0.0],
        access_context=access_context,
        k=5,
        minimum_score=0.0,
    )

    assert len(results) == 1

    assert (
        results[0].document.metadata[
            "chunk_id"
        ]
        == "chunk-compliance"
    )

    assert results[0].score == pytest.approx(
        1.0
    )


def test_role_filter_removes_unauthorized_result() -> None:
    store = create_test_store()

    access_context = SearchAccessContext(
        role="customer_support",
        region="US",
        clearance_rank=2,
    )

    results = store.search_by_vector(
        [1.0, 0.0, 0.0],
        access_context=access_context,
        k=5,
        minimum_score=0.0,
    )

    returned_chunk_ids = {
        result.document.metadata[
            "chunk_id"
        ]
        for result in results
    }

    assert "chunk-compliance" not in (
        returned_chunk_ids
    )

    assert "chunk-restricted" not in (
        returned_chunk_ids
    )


def test_clearance_filter_removes_restricted_result() -> None:
    store = create_test_store()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = store.search_by_vector(
        [0.9, 0.1, 0.0],
        access_context=access_context,
        k=5,
        minimum_score=0.0,
    )

    returned_chunk_ids = {
        result.document.metadata[
            "chunk_id"
        ]
        for result in results
    }

    assert "chunk-restricted" not in (
        returned_chunk_ids
    )


def test_region_filter_removes_document() -> None:
    store = create_test_store()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="UK",
        clearance_rank=3,
    )

    results = store.search_by_vector(
        [1.0, 0.0, 0.0],
        access_context=access_context,
        k=5,
        minimum_score=0.0,
    )

    assert results == []


def test_save_and_load_store(
    tmp_path: Path,
) -> None:
    store = create_test_store()

    index_directory = (
        tmp_path / "test_faiss_index"
    )

    store.save(index_directory)

    loaded_store = LocalFaissStore.load(
        index_directory
    )

    assert loaded_store.document_count == 3
    assert loaded_store.dimensions == 3

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = loaded_store.search_by_vector(
        [1.0, 0.0, 0.0],
        access_context=access_context,
        k=1,
    )

    assert len(results) == 1

    assert (
        results[0].document.metadata[
            "chunk_id"
        ]
        == "chunk-compliance"
    )


def test_vector_dimension_mismatch_is_rejected() -> None:
    first_chunk = create_embedded_chunk(
        chunk_id="chunk-one",
        vector=[1.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
    )

    second_chunk = create_embedded_chunk(
        chunk_id="chunk-two",
        vector=[1.0, 0.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
    )

    with pytest.raises(
        ValueError,
        match="dimensions",
    ):
        LocalFaissStore.from_embedded_chunks(
            [
                first_chunk,
                second_chunk,
            ]
        )


def test_duplicate_chunk_id_is_rejected() -> None:
    first_chunk = create_embedded_chunk(
        chunk_id="same-chunk",
        vector=[1.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
    )

    second_chunk = create_embedded_chunk(
        chunk_id="same-chunk",
        vector=[0.0, 1.0],
        allowed_roles=[
            "compliance_analyst"
        ],
    )

    with pytest.raises(
        ValueError,
        match="Duplicate chunk ID",
    ):
        LocalFaissStore.from_embedded_chunks(
            [
                first_chunk,
                second_chunk,
            ]
        )


def test_query_dimension_mismatch_is_rejected() -> None:
    store = create_test_store()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        store.search_by_vector(
            [1.0, 0.0],
            access_context=access_context,
        )