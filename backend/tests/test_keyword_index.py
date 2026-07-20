from langchain_core.documents import Document

from app.rag.faiss_store import (
    SearchAccessContext,
)
from app.rag.keyword_index import (
    LocalBm25Index,
    tokenize_for_keyword_search,
)


def create_chunk(
    *,
    chunk_id: str,
    document_id: str,
    title: str,
    content: str,
    allowed_roles: list[str],
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "title": title,
            "version": "1.0",
            "department": "Compliance",
            "document_type": "policy",
            "retrieval_enabled": True,
            "allowed_roles": allowed_roles,
            "allowed_regions": ["US"],
            "classification_rank": 2,
        },
    )


def create_keyword_index() -> LocalBm25Index:
    payment_document = create_chunk(
        chunk_id="payment-chunk",
        document_id="PAY-POL-1042",
        title="International Payment Review Policy",
        content=(
            "Enhanced review is required for "
            "high-risk international payments."
        ),
        allowed_roles=[
            "compliance_analyst"
        ],
    )

    complaint_document = create_chunk(
        chunk_id="complaint-chunk",
        document_id="CMP-POL-3015",
        title="Complaints Handling Policy",
        content=(
            "Customer complaints must be "
            "acknowledged promptly."
        ),
        allowed_roles=[
            "customer_support"
        ],
    )

    return LocalBm25Index(
        documents=[
            payment_document,
            complaint_document,
        ]
    )


def test_tokenizer_preserves_banking_terms() -> None:
    tokens = tokenize_for_keyword_search(
        "PAY-POL-1042 Section 4.3 AML KYC"
    )

    assert "pay-pol-1042" in tokens
    assert "4.3" in tokens
    assert "aml" in tokens
    assert "kyc" in tokens


def test_exact_policy_id_search() -> None:
    keyword_index = create_keyword_index()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = keyword_index.search(
        "PAY-POL-1042",
        access_context=access_context,
        k=5,
    )

    assert len(results) == 1

    assert (
        results[0].document.metadata[
            "document_id"
        ]
        == "PAY-POL-1042"
    )


def test_keyword_content_search() -> None:
    keyword_index = create_keyword_index()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = keyword_index.search(
        "high-risk international payments",
        access_context=access_context,
        k=5,
    )

    assert len(results) == 1

    assert (
        results[0].document.metadata[
            "chunk_id"
        ]
        == "payment-chunk"
    )


def test_unauthorized_keyword_result_is_removed() -> None:
    keyword_index = create_keyword_index()

    access_context = SearchAccessContext(
        role="customer_support",
        region="US",
        clearance_rank=2,
    )

    results = keyword_index.search(
        "PAY-POL-1042 international payments",
        access_context=access_context,
        k=5,
    )

    assert results == []


def test_query_with_no_keyword_evidence_returns_none() -> None:
    keyword_index = create_keyword_index()

    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    results = keyword_index.search(
        "completely unrelated astronomy",
        access_context=access_context,
        k=5,
    )

    assert results == []