from __future__ import annotations

import hashlib
import json
from typing import Any


def normalize_metadata_value(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


def build_vector_index_fingerprint(
    vector_store: Any,
) -> str:
    """
    Builds a stable fingerprint for the currently loaded
    vector index.

    The fingerprint changes when document identity,
    version, chunk identity, content hash, or chunk text
    changes. It is calculated once during service startup.
    """

    fingerprint_rows: list[dict[str, str]] = []

    for document in vector_store.documents:
        metadata = document.metadata

        content_hash = normalize_metadata_value(
            metadata.get("content_hash")
        )

        if not content_hash:
            content_hash = hashlib.sha256(
                document.page_content.encode(
                    "utf-8"
                )
            ).hexdigest()

        fingerprint_rows.append(
            {
                "document_id": normalize_metadata_value(
                    metadata.get("document_id")
                ),
                "version": normalize_metadata_value(
                    metadata.get("version")
                ),
                "chunk_id": normalize_metadata_value(
                    metadata.get("chunk_id")
                ),
                "content_hash": content_hash,
            }
        )

    fingerprint_rows.sort(
        key=lambda row: (
            row["document_id"],
            row["version"],
            row["chunk_id"],
            row["content_hash"],
        )
    )

    canonical_payload = json.dumps(
        fingerprint_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    return hashlib.sha256(
        canonical_payload.encode("utf-8")
    ).hexdigest()
