from typing import Any

from langchain_core.documents import Document

from app.models.banking_document import BankingDocument


def build_langchain_metadata(
    document: BankingDocument,
) -> dict[str, Any]:
    """
    Creates serializable metadata for a LangChain Document.

    Dates, datetimes, and enums are converted into standard
    string values so they can later be stored in vector
    databases and used for retrieval filtering.
    """

    if not document.metadata_enriched:
        raise ValueError(
            f"Document '{document.file_name}' must be "
            "metadata-enriched before LangChain conversion"
        )

    if document.content_hash is None:
        raise ValueError(
            f"Document '{document.file_name}' does not "
            "have a content hash"
        )

    if document.classification_rank is None:
        raise ValueError(
            f"Document '{document.file_name}' does not "
            "have a classification rank"
        )

    if document.entitlement_key is None:
        raise ValueError(
            f"Document '{document.file_name}' does not "
            "have an entitlement key"
        )

    source = (
        document.source_uri
        if document.source_uri is not None
        else document.file_name
    )

    owner = (
        document.owner
        if document.owner is not None
        else ""
    )

    parent_document_key = (
        f"{document.document_id}:{document.version}"
    )

    return {
        # Unique parent-document identity
        "parent_document_key": parent_document_key,
        "document_id": document.document_id,
        "title": document.title,
        "version": document.version,

        # Governance and lifecycle
        "status": document.status.value,
        "effective_date": (
            document.effective_date.isoformat()
        ),
        "document_type": document.document_type,
        "schema_version": document.schema_version,
        "language": document.language,

        # Business ownership
        "department": document.department,
        "owner": owner,

        # Authorization metadata
        "region": document.region,
        "allowed_regions": list(
            document.allowed_regions
        ),
        "classification": (
            document.classification.value
        ),
        "classification_rank": (
            document.classification_rank
        ),
        "allowed_roles": list(
            document.allowed_roles
        ),
        "entitlement_key": (
            document.entitlement_key
        ),
        "retrieval_enabled": (
            document.retrieval_enabled
        ),

        # Source lineage
        "file_name": document.file_name,
        "source": source,
        "source_uri": source,
        "source_system": document.source_system,
        "ingested_at": (
            document.ingested_at.isoformat()
        ),

        # Processing lineage
        "content_hash": document.content_hash,
        "metadata_enriched": (
            document.metadata_enriched
        ),

        # PII-processing status
        "pii_detected": document.pii_detected,
        "pii_masked": document.pii_masked,
        "pii_types": list(document.pii_types),
        "pii_count": document.pii_count,

        # Current object level
        "record_type": "parent_document",
    }


def convert_to_langchain_document(
    document: BankingDocument,
) -> Document:
    """
    Converts one enriched BankingDocument into a LangChain
    Document.

    page_content contains the cleaned and PII-masked text.

    metadata contains identity, authorization, lifecycle,
    lineage, and processing information.
    """

    metadata = build_langchain_metadata(document)

    return Document(
        page_content=document.content,
        metadata=metadata,
    )


def convert_to_langchain_documents(
    documents: list[BankingDocument],
) -> list[Document]:
    """
    Converts multiple BankingDocument objects into LangChain
    Document objects.
    """

    return [
        convert_to_langchain_document(document)
        for document in documents
    ]


def select_retrievable_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Selects only documents that are eligible for normal RAG
    retrieval and indexing.

    Superseded, archived, expired, or future-effective
    documents remain available for audit but are not included
    in the normal searchable index.
    """

    return [
        document
        for document in documents
        if document.metadata.get(
            "retrieval_enabled"
        ) is True
    ]