import pytest
from langchain_core.documents import Document

from app.config import (
    EmbeddingSettings,
    HybridRetrievalSettings,
    RerankerSettings,
)
from app.rag.embedding_service import (
    EmbeddedChunk,
)
from app.rag.faiss_store import (
    LocalFaissStore,
    SearchAccessContext,
)
from app.rag.keyword_index import (
    LocalBm25Index,
)
from app.rag.reranker import (
    LocalFeatureReranker,
)
from app.rag.retrieval_service import (
    PermissionAwareRetrievalService,
    build_chunk_citation,
)


class FakeEmbeddingClient:
    """
    Fake query embedding client.
    """

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self.embed_query(text)
            for text in texts
        ]

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        normalized_text = text.lower()

        if (
            "payment" in normalized_text
            or "transfer" in normalized_text
        ):
            return [1.0, 0.0, 0.0]

        if "complaint" in normalized_text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def create_embedded_chunk(
    *,
    chunk_id: str,
    document_id: str,
    title: str,
    vector: list[float],
    allowed_roles: list[str],
    content: str,
) -> EmbeddedChunk:
    document = Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "chunk_number": 1,
            "total_chunks": 1,
            "document_id": document_id,
            "title": title,
            "version": "1.0",
            "department": "Compliance",
            "document_type": "policy",
            "record_type": "chunk",
            "embedding_status": "COMPLETED",
            "embedding_model": (
                "fake-embedding-model"
            ),
            "embedding_dimensions": 3,
            "retrieval_enabled": True,
            "allowed_roles": allowed_roles,
            "allowed_regions": ["US"],
            "classification_rank": 2,
            "entitlement_key": (
                "department=compliance"
                "|regions=US"
                "|classification=2"
            ),
            "pii_detected": False,
            "pii_masked": False,
            "source": (
                f"local://{document_id}.txt"
            ),
        },
    )

    return EmbeddedChunk(
        document=document,
        vector=vector,
    )


def create_retrieval_service(
) -> PermissionAwareRetrievalService:
    payment_chunk = create_embedded_chunk(
        chunk_id="payment-chunk",
        document_id="PAY-POL-1042",
        title="International Payment Review Policy",
        vector=[1.0, 0.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
        content=(
            "Enhanced verification is required for "
            "high-risk international payments."
        ),
    )

    complaint_chunk = create_embedded_chunk(
        chunk_id="complaint-chunk",
        document_id="CMP-POL-3015",
        title="Complaints Handling Policy",
        vector=[0.0, 1.0, 0.0],
        allowed_roles=[
            "customer_support"
        ],
        content=(
            "Customer complaints must be "
            "acknowledged promptly."
        ),
    )

    vector_store = (
        LocalFaissStore.from_embedded_chunks(
            [
                payment_chunk,
                complaint_chunk,
            ]
        )
    )

    keyword_index = LocalBm25Index(
        documents=vector_store.documents
    )

    embedding_settings = EmbeddingSettings(
        api_key="test-api-key",
        model="fake-embedding-model",
        dimensions=3,
        batch_size=2,
        base_url=None,
    )

    hybrid_settings = HybridRetrievalSettings(
        vector_candidate_k=10,
        keyword_candidate_k=10,
        keyword_minimum_score=0.0,
        reciprocal_rank_constant=60,
        vector_weight=1.0,
        keyword_weight=1.0,
    )

    reranker_settings = RerankerSettings(
        backend="local_feature",
        candidate_k=10,
        top_n=5,
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L6-v2"
        ),
        device="cpu",
    )

    return PermissionAwareRetrievalService(
        vector_store=vector_store,
        keyword_index=keyword_index,
        embedding_client=(
            FakeEmbeddingClient()
        ),
        embedding_settings=(
            embedding_settings
        ),
        hybrid_settings=hybrid_settings,
        reranker=LocalFeatureReranker(),
        reranker_settings=(
            reranker_settings
        ),
        default_top_k=5,
        default_vector_minimum_score=0.1,
    )


def test_retrieval_returns_reranked_payment_chunk(
) -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    response = retrieval_service.retrieve(
        (
            "What verification is required for "
            "a high-risk international payment?"
        ),
        access_context=access_context,
    )

    assert response.result_count == 1

    result = response.results[0]

    assert (
        result.document.metadata["chunk_id"]
        == "payment-chunk"
    )

    assert result.rank == 1
    assert result.rerank_score > 0

    assert (
        result.reranker_backend
        == "local_feature"
    )


def test_exact_policy_id_is_retrieved() -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    response = retrieval_service.retrieve(
        "Show policy PAY-POL-1042",
        access_context=access_context,
    )

    assert response.result_count == 1

    assert (
        response.results[0]
        .document
        .metadata["document_id"]
        == "PAY-POL-1042"
    )


def test_unauthorized_role_receives_no_chunk(
) -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="customer_support",
        region="US",
        clearance_rank=2,
    )

    response = retrieval_service.retrieve(
        "PAY-POL-1042 risky payment",
        access_context=access_context,
    )

    assert response.result_count == 0
    assert response.documents == []


def test_wrong_region_receives_no_chunk() -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="UK",
        clearance_rank=2,
    )

    response = retrieval_service.retrieve(
        "International payment review",
        access_context=access_context,
    )

    assert response.result_count == 0


def test_retrieve_documents_returns_documents(
) -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    documents = (
        retrieval_service.retrieve_documents(
            "International payment verification",
            access_context=access_context,
        )
    )

    assert len(documents) == 1

    assert isinstance(
        documents[0],
        Document,
    )


def test_empty_query_is_rejected() -> None:
    retrieval_service = (
        create_retrieval_service()
    )

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        retrieval_service.retrieve(
            "   ",
            access_context=access_context,
        )


def test_build_chunk_citation() -> None:
    chunk = create_embedded_chunk(
        chunk_id="payment-chunk",
        document_id="PAY-POL-1042",
        title="Payment Review Policy",
        vector=[1.0, 0.0, 0.0],
        allowed_roles=[
            "compliance_analyst"
        ],
        content="Payment policy.",
    )

    citation = build_chunk_citation(
        chunk.document
    )

    assert "Payment Review Policy" in citation
    assert "PAY-POL-1042" in citation
    assert "version 1.0" in citation
    assert "chunk 1/1" in citation