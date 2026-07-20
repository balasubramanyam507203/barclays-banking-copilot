import pytest
from langchain_core.documents import Document

from app.rag.faiss_store import (
    SearchAccessContext,
)
from app.rag.reranker import (
    LocalFeatureReranker,
    RerankCandidate,
)


def create_candidate(
    *,
    chunk_id: str,
    document_id: str,
    title: str,
    content: str,
    hybrid_rank: int,
    allowed_roles: list[str] | None = None,
) -> RerankCandidate:
    document = Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "title": title,
            "version": "1.0",
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

    return RerankCandidate(
        document=document,
        hybrid_rank=hybrid_rank,
        hybrid_score=(
            1.0 / (60 + hybrid_rank)
        ),
        vector_rank=hybrid_rank,
        vector_score=0.8,
        keyword_rank=hybrid_rank,
        keyword_score=1.0,
        matched_by=(
            "semantic",
            "keyword",
        ),
    )


def create_access_context(
) -> SearchAccessContext:
    return SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )


def test_exact_policy_id_is_ranked_first() -> None:
    reranker = LocalFeatureReranker()

    payment_candidate = create_candidate(
        chunk_id="payment-chunk",
        document_id="PAY-POL-1042",
        title="International Payment Policy",
        content=(
            "Enhanced review is required for "
            "international transfers."
        ),
        hybrid_rank=2,
    )

    complaint_candidate = create_candidate(
        chunk_id="complaint-chunk",
        document_id="CMP-POL-3015",
        title="Complaints Policy",
        content=(
            "Customer complaints must be "
            "acknowledged promptly."
        ),
        hybrid_rank=1,
    )

    results = reranker.rerank(
        "Show policy PAY-POL-1042",
        candidates=[
            complaint_candidate,
            payment_candidate,
        ],
        access_context=(
            create_access_context()
        ),
        top_n=2,
    )

    assert len(results) == 2

    assert (
        results[0]
        .document
        .metadata["document_id"]
        == "PAY-POL-1042"
    )


def test_relevant_policy_content_is_ranked_first() -> None:
    reranker = LocalFeatureReranker()

    payment_candidate = create_candidate(
        chunk_id="payment-chunk",
        document_id="PAY-POL-1042",
        title="Payment Review Policy",
        content=(
            "High-risk international payments "
            "require enhanced verification."
        ),
        hybrid_rank=2,
    )

    complaint_candidate = create_candidate(
        chunk_id="complaint-chunk",
        document_id="CMP-POL-3015",
        title="Complaints Policy",
        content=(
            "Complaints must be acknowledged "
            "within the required timeframe."
        ),
        hybrid_rank=1,
    )

    results = reranker.rerank(
        (
            "What verification is required for "
            "high-risk international payments?"
        ),
        candidates=[
            complaint_candidate,
            payment_candidate,
        ],
        access_context=(
            create_access_context()
        ),
        top_n=2,
    )

    assert (
        results[0]
        .document
        .metadata["chunk_id"]
        == "payment-chunk"
    )


def test_top_n_limits_results() -> None:
    reranker = LocalFeatureReranker()

    candidates = [
        create_candidate(
            chunk_id=f"chunk-{index}",
            document_id=f"POL-{index}",
            title=f"Policy {index}",
            content="Banking policy content.",
            hybrid_rank=index + 1,
        )
        for index in range(5)
    ]

    results = reranker.rerank(
        "banking policy",
        candidates=candidates,
        access_context=(
            create_access_context()
        ),
        top_n=2,
    )

    assert len(results) == 2


def test_unauthorized_candidate_is_removed() -> None:
    reranker = LocalFeatureReranker()

    unauthorized_candidate = create_candidate(
        chunk_id="restricted-role-chunk",
        document_id="SEC-POL-100",
        title="Security Policy",
        content="Restricted security procedure.",
        hybrid_rank=1,
        allowed_roles=[
            "security_investigator"
        ],
    )

    results = reranker.rerank(
        "security procedure",
        candidates=[
            unauthorized_candidate
        ],
        access_context=(
            create_access_context()
        ),
        top_n=5,
    )

    assert results == []


def test_duplicate_candidate_is_rejected() -> None:
    reranker = LocalFeatureReranker()

    candidate = create_candidate(
        chunk_id="same-chunk",
        document_id="PAY-POL-1042",
        title="Payment Policy",
        content="Payment policy content.",
        hybrid_rank=1,
    )

    with pytest.raises(
        ValueError,
        match="Duplicate rerank candidate",
    ):
        reranker.rerank(
            "payment policy",
            candidates=[
                candidate,
                candidate,
            ],
            access_context=(
                create_access_context()
            ),
            top_n=5,
        )