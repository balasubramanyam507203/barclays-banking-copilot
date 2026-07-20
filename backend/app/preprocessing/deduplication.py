import hashlib
from collections import defaultdict
from typing import Any

from app.models.banking_document import (
    BankingDocument,
    DocumentStatus,
)


def canonicalize_content_for_hash(content: str) -> str:
    """
    Creates a stable text representation before hashing.

    Cleaning should already have happened before this function.
    This function only removes trailing line spaces and normalizes
    the final representation.
    """

    canonical_lines = [
        line.rstrip()
        for line in content.strip().splitlines()
    ]

    return "\n".join(canonical_lines)


def create_content_hash(content: str) -> str:
    """
    Creates a SHA-256 fingerprint from cleaned document content.
    """

    canonical_content = canonicalize_content_for_hash(content)

    return hashlib.sha256(
        canonical_content.encode("utf-8")
    ).hexdigest()


def attach_content_hash(
    document: BankingDocument,
) -> BankingDocument:
    """
    Returns a new BankingDocument containing its content hash.
    """

    content_hash = create_content_hash(document.content)

    return document.model_copy(
        update={
            "content_hash": content_hash,
        }
    )


def detect_duplicate_and_version_issues(
    documents: list[BankingDocument],
) -> tuple[
    list[BankingDocument],
    list[dict[str, Any]],
]:
    """
    Adds content hashes and detects:

    1. Duplicate copies of the same document version.
    2. Conflicting content for the same document ID and version.
    3. Multiple ACTIVE versions of one document.

    Returns accepted documents and rejected-document details.
    """

    hashed_documents = [
        attach_content_hash(document)
        for document in documents
    ]

    rejection_reasons: dict[str, list[str]] = defaultdict(list)

    documents_by_identity: dict[
        tuple[str, str],
        list[BankingDocument],
    ] = defaultdict(list)

    for document in hashed_documents:
        identity = (
            document.document_id,
            document.version,
        )

        documents_by_identity[identity].append(document)

    # Detect duplicates or content conflicts for the same
    # document ID and version.
    for (
        document_id,
        version,
    ), identity_documents in documents_by_identity.items():

        unique_hashes = {
            document.content_hash
            for document in identity_documents
        }

        if len(unique_hashes) > 1:
            reason = (
                "VERSION_CONTENT_CONFLICT: Multiple files use "
                f"document ID '{document_id}' and version "
                f"'{version}', but their content is different."
            )

            for document in identity_documents:
                rejection_reasons[document.file_name].append(
                    reason
                )

        elif len(identity_documents) > 1:
            canonical_document = min(
                identity_documents,
                key=lambda document: document.file_name,
            )

            for document in identity_documents:
                if (
                    document.file_name
                    != canonical_document.file_name
                ):
                    reason = (
                        "EXACT_DUPLICATE: Same document ID, "
                        f"version and content as "
                        f"'{canonical_document.file_name}'."
                    )

                    rejection_reasons[
                        document.file_name
                    ].append(reason)

    active_documents_by_id: dict[
        str,
        list[BankingDocument],
    ] = defaultdict(list)

    for document in hashed_documents:
        if rejection_reasons.get(document.file_name):
            continue

        if document.status == DocumentStatus.ACTIVE:
            active_documents_by_id[
                document.document_id
            ].append(document)

    # Detect more than one active version for one document ID.
    for (
        document_id,
        active_documents,
    ) in active_documents_by_id.items():

        active_versions = {
            document.version
            for document in active_documents
        }

        if len(active_versions) > 1:
            versions_text = ", ".join(
                sorted(active_versions)
            )

            reason = (
                "MULTIPLE_ACTIVE_VERSIONS: Document ID "
                f"'{document_id}' has multiple active versions: "
                f"{versions_text}."
            )

            for document in active_documents:
                rejection_reasons[
                    document.file_name
                ].append(reason)

    accepted_documents = [
        document
        for document in hashed_documents
        if not rejection_reasons.get(document.file_name)
    ]

    rejected_documents = []

    for document in hashed_documents:
        reasons = rejection_reasons.get(document.file_name)

        if not reasons:
            continue

        rejected_documents.append(
            {
                "file_name": document.file_name,
                "document_id": document.document_id,
                "version": document.version,
                "content_hash": document.content_hash,
                "reasons": reasons,
            }
        )

    return accepted_documents, rejected_documents