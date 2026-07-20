from datetime import date

import pytest
from langchain_core.documents import Document

from app.models.banking_document import (
    BankingDocument,
)
from app.preprocessing.deduplication import (
    attach_content_hash,
)
from app.preprocessing.metadata_enricher import (
    enrich_document_metadata,
)
from app.rag.langchain_converter import (
    build_langchain_metadata,
    convert_to_langchain_document,
    convert_to_langchain_documents,
    select_retrievable_documents,
)


def create_enriched_test_document(
    *,
    status: str = "ACTIVE",
    effective_date: str = "2026-01-01",
) -> BankingDocument:
    document = BankingDocument(
        file_name="payment_policy.txt",
        document_id="PAY-POL-1042",
        title=(
            "International Payment Review Policy"
        ),
        version="7.2",
        status=status,
        effective_date=effective_date,
        department="Payments Compliance",
        region="US",
        classification=(
            "Internal Confidential"
        ),
        allowed_roles=[
            "payments_analyst",
            "compliance_analyst",
        ],
        content=(
            "Enhanced review is required for "
            "high-risk international payments."
        ),
        source_system="local_files",
        source_uri=(
            "local://sample_documents/"
            "payment_policy.txt"
        ),
        owner="payments-compliance",
        allowed_regions=["US"],
        pii_detected=False,
        pii_masked=False,
        pii_types=[],
        pii_count=0,
    )

    hashed_document = attach_content_hash(
        document
    )

    return enrich_document_metadata(
        hashed_document,
        as_of_date=date(2026, 7, 17),
    )


def test_build_langchain_metadata() -> None:
    document = create_enriched_test_document()

    metadata = build_langchain_metadata(
        document
    )

    assert (
        metadata["document_id"]
        == "PAY-POL-1042"
    )

    assert metadata["version"] == "7.2"

    assert (
        metadata["classification"]
        == "Internal Confidential"
    )

    assert (
        metadata["classification_rank"]
        == 2
    )

    assert metadata["allowed_regions"] == [
        "US"
    ]

    assert metadata["allowed_roles"] == [
        "compliance_analyst",
        "payments_analyst",
    ]

    assert (
        metadata["retrieval_enabled"]
        is True
    )

    assert (
        metadata["record_type"]
        == "parent_document"
    )


def test_convert_to_langchain_document() -> None:
    banking_document = (
        create_enriched_test_document()
    )

    langchain_document = (
        convert_to_langchain_document(
            banking_document
        )
    )

    assert isinstance(
        langchain_document,
        Document,
    )

    assert (
        langchain_document.page_content
        == banking_document.content
    )

    assert (
        langchain_document.metadata[
            "document_id"
        ]
        == banking_document.document_id
    )


def test_convert_multiple_documents() -> None:
    first_document = (
        create_enriched_test_document()
    )

    second_document = (
        first_document.model_copy(
            update={
                "file_name": (
                    "complaints_policy.txt"
                ),
                "document_id": (
                    "CMP-POL-3015"
                ),
                "title": (
                    "Complaints Handling Policy"
                ),
                "content_hash": "b" * 64,
            }
        )
    )

    langchain_documents = (
        convert_to_langchain_documents(
            [
                first_document,
                second_document,
            ]
        )
    )

    assert len(langchain_documents) == 2

    assert all(
        isinstance(document, Document)
        for document in langchain_documents
    )


def test_select_retrievable_documents() -> None:
    active_document = (
        create_enriched_test_document(
            status="ACTIVE",
            effective_date="2026-01-01",
        )
    )

    future_document = (
        create_enriched_test_document(
            status="ACTIVE",
            effective_date="2027-01-01",
        )
    )

    # Re-evaluate the future document using the fixed
    # test date.
    future_document = enrich_document_metadata(
        future_document,
        as_of_date=date(2026, 7, 17),
    )

    langchain_documents = (
        convert_to_langchain_documents(
            [
                active_document,
                future_document,
            ]
        )
    )

    retrievable_documents = (
        select_retrievable_documents(
            langchain_documents
        )
    )

    assert len(retrievable_documents) == 1

    assert (
        retrievable_documents[0].metadata[
            "retrieval_enabled"
        ]
        is True
    )


def test_conversion_fails_without_enrichment() -> None:
    document = BankingDocument(
        file_name="policy.txt",
        document_id="TEST-POL-100",
        title="Test Policy",
        version="1.0",
        status="ACTIVE",
        effective_date="2026-01-01",
        department="Compliance",
        region="US",
        classification="Internal",
        allowed_roles=[
            "compliance_analyst"
        ],
        content="Example policy content.",
    )

    with pytest.raises(
        ValueError,
        match="metadata-enriched",
    ):
        convert_to_langchain_document(
            document
        )