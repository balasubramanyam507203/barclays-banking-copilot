from typing import Any

from app.models.banking_document import BankingDocument


def get_required_metadata(
    metadata: dict[str, str],
    key: str,
    file_name: str,
) -> str:
    """
    Returns a required metadata value.

    Raises an error when the field is missing or empty.
    """

    value = metadata.get(key)

    if value is None or not value.strip():
        raise ValueError(
            f"Missing required metadata '{key}' "
            f"in document '{file_name}'"
        )

    return value.strip()


def get_optional_metadata(
    metadata: dict[str, str],
    key: str,
) -> str | None:
    """
    Returns an optional metadata value.

    Returns None when the field is missing or empty.
    """

    value = metadata.get(key)

    if value is None:
        return None

    cleaned_value = value.strip()

    return cleaned_value or None


def parse_allowed_roles(
    value: str,
) -> list[str]:
    """
    Converts comma-separated roles into a Python list.

    Example:

    'payments_analyst, compliance_analyst'

    becomes:

    ['payments_analyst', 'compliance_analyst']
    """

    roles = [
        role.strip()
        for role in value.split(",")
        if role.strip()
    ]

    if not roles:
        raise ValueError(
            "Allowed Roles must contain at least one role"
        )

    return roles


def map_raw_document_to_banking_document(
    raw_document: dict[str, Any],
) -> BankingDocument:
    """
    Converts a raw loader dictionary into a validated
    BankingDocument object.
    """

    file_name = raw_document["file_name"]
    metadata = raw_document["metadata"]
    content = raw_document["content"]

    region = get_required_metadata(
        metadata,
        "Region",
        file_name,
    )

    return BankingDocument(
        file_name=file_name,
        document_id=get_required_metadata(
            metadata,
            "Document ID",
            file_name,
        ),
        title=get_required_metadata(
            metadata,
            "Title",
            file_name,
        ),
        version=get_required_metadata(
            metadata,
            "Version",
            file_name,
        ),
        status=get_required_metadata(
            metadata,
            "Status",
            file_name,
        ),
        effective_date=get_required_metadata(
            metadata,
            "Effective Date",
            file_name,
        ),
        department=get_required_metadata(
            metadata,
            "Department",
            file_name,
        ),
        region=region,
        classification=get_required_metadata(
            metadata,
            "Classification",
            file_name,
        ),
        allowed_roles=parse_allowed_roles(
            get_required_metadata(
                metadata,
                "Allowed Roles",
                file_name,
            )
        ),
        content=content,
        source_system="local_files",
        source_uri=(
            "local://sample_documents/"
            f"{file_name}"
        ),
        owner=get_optional_metadata(
            metadata,
            "Owner",
        ),
        allowed_regions=[region],
    )