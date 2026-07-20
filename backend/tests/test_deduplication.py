from app.models.banking_document import BankingDocument
from app.preprocessing.deduplication import (
    attach_content_hash,
    create_content_hash,
    detect_duplicate_and_version_issues,
)


def create_test_document(
    *,
    file_name: str,
    version: str = "1.0",
    status: str = "ACTIVE",
    content: str = "Example banking policy.",
) -> BankingDocument:
    return BankingDocument(
        file_name=file_name,
        document_id="TEST-POL-100",
        title="Test Policy",
        version=version,
        status=status,
        effective_date="2026-01-01",
        department="Compliance",
        region="US",
        classification="Internal",
        allowed_roles=["compliance_analyst"],
        content=content,
    )


def test_content_hash_is_deterministic() -> None:
    content = "Enhanced review is required."

    first_hash = create_content_hash(content)
    second_hash = create_content_hash(content)

    assert first_hash == second_hash


def test_content_change_produces_different_hash() -> None:
    first_hash = create_content_hash(
        "Enhanced review is required."
    )

    second_hash = create_content_hash(
        "Enhanced review may be required."
    )

    assert first_hash != second_hash


def test_attach_content_hash() -> None:
    document = create_test_document(
        file_name="policy.txt"
    )

    hashed_document = attach_content_hash(document)

    assert hashed_document.content_hash is not None
    assert len(hashed_document.content_hash) == 64


def test_exact_duplicate_is_rejected() -> None:
    first_document = create_test_document(
        file_name="policy_a.txt"
    )

    second_document = create_test_document(
        file_name="policy_b.txt"
    )

    accepted, rejected = (
        detect_duplicate_and_version_issues(
            [first_document, second_document]
        )
    )

    assert len(accepted) == 1
    assert len(rejected) == 1
    assert "EXACT_DUPLICATE" in rejected[0]["reasons"][0]


def test_multiple_active_versions_are_rejected() -> None:
    version_one = create_test_document(
        file_name="policy_v1.txt",
        version="1.0",
        status="ACTIVE",
        content="Version one content.",
    )

    version_two = create_test_document(
        file_name="policy_v2.txt",
        version="2.0",
        status="ACTIVE",
        content="Version two content.",
    )

    accepted, rejected = (
        detect_duplicate_and_version_issues(
            [version_one, version_two]
        )
    )

    assert len(accepted) == 0
    assert len(rejected) == 2