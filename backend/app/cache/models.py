from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class CachedRagResponse(BaseModel):
    """
    Guardrail-approved response data stored in Redis.

    The cache intentionally excludes raw model drafts,
    prompts, authentication tokens, database objects,
    and request-specific identifiers.
    """

    schema_version: str = Field(
        min_length=1,
        max_length=50,
    )

    status: Literal[
        "answered",
        "abstained",
    ]

    answer: str = Field(
        min_length=1,
    )

    abstained: bool
    citations_used: list[str]
    sources: list[dict[str, Any]]
    evidence_count: int = Field(
        ge=0,
    )
    guardrails: dict[str, Any]

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
