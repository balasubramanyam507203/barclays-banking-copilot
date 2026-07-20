import re
from datetime import date

from app.models.banking_document import (
    BankingDocument,
    DocumentStatus,
    SecurityClassification,
)


CLASSIFICATION_RANKS = {
    SecurityClassification.INTERNAL: 1,
    SecurityClassification.INTERNAL_CONFIDENTIAL: 2,
    SecurityClassification.RESTRICTED: 3,
}


NON_ALPHANUMERIC_PATTERN = re.compile(
    r"[^a-z0-9]+"
)


def create_slug(value: str) -> str:
    """
    Converts a string into a normalized slug.

    Example:
    'Payments Compliance'

    becomes:
    'payments-compliance'
    """

    lowercase_value = value.strip().lower()

    slug = NON_ALPHANUMERIC_PATTERN.sub(
        "-",
        lowercase_value,
    )

    return slug.strip("-")


def create_entitlement_key(
    document: BankingDocument,
    classification_rank: int,
) -> str:
    """
    Creates a deterministic permission-group key.
    """

    department_slug = create_slug(
        document.department
    )

    regions = ",".join(
        sorted(document.allowed_regions)
    )

    roles = ",".join(
        sorted(document.allowed_roles)
    )

    return (
        f"department={department_slug}"
        f"|regions={regions}"
        f"|classification={classification_rank}"
        f"|roles={roles}"
    )


def should_enable_retrieval(
    document: BankingDocument,
    *,
    as_of_date: date,
) -> bool:
    """
    Determines whether a document can be included
    in the normal RAG retrieval index.
    """

    return (
        document.status == DocumentStatus.ACTIVE
        and document.effective_date <= as_of_date
        and bool(document.allowed_roles)
        and bool(document.allowed_regions)
        and document.content_hash is not None
    )


def enrich_document_metadata(
    document: BankingDocument,
    *,
    as_of_date: date | None = None,
) -> BankingDocument:
    """
    Adds standardized permission and retrieval metadata
    to one BankingDocument.
    """

    evaluation_date = as_of_date or date.today()

    classification_rank = CLASSIFICATION_RANKS[
        document.classification
    ]

    normalized_regions = sorted(
        {
            document.region.strip().upper(),
            *[
                region.strip().upper()
                for region in document.allowed_regions
                if region.strip()
            ],
        }
    )

    document_with_regions = document.model_copy(
        update={
            "region": document.region.upper(),
            "allowed_regions": normalized_regions,
        }
    )

    entitlement_key = create_entitlement_key(
        document_with_regions,
        classification_rank,
    )

    retrieval_enabled = should_enable_retrieval(
        document_with_regions,
        as_of_date=evaluation_date,
    )

    return document_with_regions.model_copy(
        update={
            "classification_rank": (
                classification_rank
            ),
            "entitlement_key": entitlement_key,
            "retrieval_enabled": retrieval_enabled,
            "metadata_enriched": True,
        }
    )


def enrich_documents_metadata(
    documents: list[BankingDocument],
    *,
    as_of_date: date | None = None,
) -> list[BankingDocument]:
    """
    Enriches multiple BankingDocument objects.
    """

    return [
        enrich_document_metadata(
            document,
            as_of_date=as_of_date,
        )
        for document in documents
    ]