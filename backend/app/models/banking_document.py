from datetime import date, datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    ARCHIVED = "ARCHIVED"


class SecurityClassification(StrEnum):
    INTERNAL = "Internal"
    INTERNAL_CONFIDENTIAL = "Internal Confidential"
    RESTRICTED = "Restricted"


class BankingDocument(BaseModel):
    """
    Standard internal representation of a banking document
    before chunking and embedding.
    """

    # Core document identity
    file_name: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: DocumentStatus
    effective_date: date

    # Business metadata
    department: str = Field(min_length=1)
    region: str = Field(min_length=2)
    classification: SecurityClassification
    allowed_roles: list[str] = Field(min_length=1)

    # Actual searchable policy content
    content: str = Field(min_length=1)

    # Source lineage
    source_system: str = "local_files"
    source_uri: str | None = None
    owner: str | None = None
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Document processing metadata
    content_hash: str | None = None
    schema_version: str = "1.0"
    document_type: str = "policy"
    language: str = "en"

    # PII processing metadata
    pii_detected: bool = False
    pii_masked: bool = False
    pii_types: list[str] = Field(default_factory=list)
    pii_count: int = 0

    # Permission and retrieval metadata
    allowed_regions: list[str] = Field(default_factory=list)
    classification_rank: int | None = None
    entitlement_key: str | None = None
    retrieval_enabled: bool = False
    metadata_enriched: bool = False

    @field_validator(
        "file_name",
        "document_id",
        "title",
        "version",
        "department",
        "region",
        "content",
        "source_system",
        "schema_version",
        "document_type",
        "language",
    )
    @classmethod
    def remove_surrounding_spaces(
        cls,
        value: str,
    ) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Value must not be empty")

        return cleaned_value

    @field_validator("owner")
    @classmethod
    def clean_optional_owner(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        cleaned_value = value.strip()

        return cleaned_value or None

    @field_validator("allowed_roles")
    @classmethod
    def normalize_allowed_roles(
        cls,
        roles: list[str],
    ) -> list[str]:
        cleaned_roles = {
            role.strip().lower()
            for role in roles
            if role.strip()
        }

        if not cleaned_roles:
            raise ValueError(
                "At least one allowed role is required"
            )

        return sorted(cleaned_roles)

    @field_validator("allowed_regions")
    @classmethod
    def normalize_allowed_regions(
        cls,
        regions: list[str],
    ) -> list[str]:
        cleaned_regions = {
            region.strip().upper()
            for region in regions
            if region.strip()
        }

        return sorted(cleaned_regions)