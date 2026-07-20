import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.documents import Document
from numpy.typing import NDArray

from app.rag.embedding_service import (
    EmbeddedChunk,
)


INDEX_FILE_NAME = "index.faiss"
DOCUMENTS_FILE_NAME = "documents.json"
MANIFEST_FILE_NAME = "manifest.json"
INDEX_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class SearchAccessContext:
    """
    Security context belonging to the authenticated user.

    Later, these values will come from validated JWT claims.
    """

    role: str
    region: str
    clearance_rank: int

    def __post_init__(self) -> None:
        normalized_role = self.role.strip().lower()
        normalized_region = self.region.strip().upper()

        if not normalized_role:
            raise ValueError(
                "User role cannot be empty."
            )

        if not normalized_region:
            raise ValueError(
                "User region cannot be empty."
            )

        if self.clearance_rank <= 0:
            raise ValueError(
                "User clearance rank must be "
                "greater than zero."
            )

        object.__setattr__(
            self,
            "role",
            normalized_role,
        )

        object.__setattr__(
            self,
            "region",
            normalized_region,
        )


@dataclass(frozen=True)
class SearchResult:
    """
    One authorized similarity-search result.
    """

    document: Document
    score: float
    rank: int


class LocalFaissStore:
    """
    Local FAISS vector index with a separate JSON document
    store.

    FAISS stores the numerical vectors.

    The JSON file stores:
    - chunk text
    - document metadata
    - security metadata
    - citation metadata

    Documents and vectors must remain in the same order.
    """

    def __init__(
        self,
        *,
        index: Any,
        documents: list[Document],
        dimensions: int,
    ) -> None:
        if dimensions <= 0:
            raise ValueError(
                "Vector dimensions must be greater "
                "than zero."
            )

        if index.d != dimensions:
            raise ValueError(
                "FAISS index dimension does not match "
                "the configured dimension."
            )

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS vector count does not match "
                "the document count."
            )

        self.index = index
        self.documents = documents
        self.dimensions = dimensions

    @property
    def document_count(self) -> int:
        """
        Returns the number of indexed chunks.
        """

        return len(self.documents)

    @classmethod
    def from_embedded_chunks(
        cls,
        embedded_chunks: list[EmbeddedChunk],
    ) -> "LocalFaissStore":
        """
        Builds an exact cosine-similarity FAISS index from
        precomputed chunk vectors.

        The embedding API is not called again.
        """

        if not embedded_chunks:
            raise ValueError(
                "At least one embedded chunk is required "
                "to build the FAISS index."
            )

        first_vector = embedded_chunks[0].vector
        dimensions = len(first_vector)

        if dimensions <= 0:
            raise ValueError(
                "Embedding vectors cannot be empty."
            )

        documents: list[Document] = []
        vectors: list[list[float]] = []
        chunk_ids: set[str] = set()

        for embedded_chunk in embedded_chunks:
            document = embedded_chunk.document
            vector = embedded_chunk.vector

            validate_indexable_document(
                document
            )

            chunk_id = str(
                document.metadata["chunk_id"]
            )

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID detected: "
                    f"'{chunk_id}'."
                )

            chunk_ids.add(chunk_id)

            if len(vector) != dimensions:
                raise ValueError(
                    f"Chunk '{chunk_id}' has "
                    f"{len(vector)} vector dimensions, "
                    f"but expected {dimensions}."
                )

            if not all(
                isinstance(value, int | float)
                for value in vector
            ):
                raise ValueError(
                    f"Chunk '{chunk_id}' contains a "
                    "non-numeric vector value."
                )

            documents.append(document)
            vectors.append(vector)

        vector_matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        normalize_vector_matrix(
            vector_matrix
        )

        # Inner product on normalized vectors is equivalent
        # to cosine similarity.
        index = faiss.IndexFlatIP(
            dimensions
        )

        index.add(vector_matrix)

        return cls(
            index=index,
            documents=documents,
            dimensions=dimensions,
        )

    def save(
        self,
        directory: Path,
    ) -> None:
        """
        Persists the FAISS index, document store, and
        manifest.

        We use JSON for documents instead of pickle.
        """

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            directory / INDEX_FILE_NAME
        )

        documents_path = (
            directory / DOCUMENTS_FILE_NAME
        )

        manifest_path = (
            directory / MANIFEST_FILE_NAME
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        serialized_documents = [
            {
                "page_content": (
                    document.page_content
                ),
                "metadata": document.metadata,
            }
            for document in self.documents
        ]

        documents_path.write_text(
            json.dumps(
                serialized_documents,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest = {
            "format_version": (
                INDEX_FORMAT_VERSION
            ),
            "vector_dimensions": (
                self.dimensions
            ),
            "document_count": (
                self.document_count
            ),
            "distance_metric": (
                "cosine_similarity"
            ),
            "faiss_index_type": (
                "IndexFlatIP"
            ),
        }

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> "LocalFaissStore":
        """
        Loads a previously saved local FAISS index.
        """

        index_path = (
            directory / INDEX_FILE_NAME
        )

        documents_path = (
            directory / DOCUMENTS_FILE_NAME
        )

        manifest_path = (
            directory / MANIFEST_FILE_NAME
        )

        required_files = [
            index_path,
            documents_path,
            manifest_path,
        ]

        missing_files = [
            path.name
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "FAISS index directory is missing: "
                + ", ".join(missing_files)
            )

        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            manifest.get("format_version")
            != INDEX_FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported FAISS index format "
                "version."
            )

        dimensions = int(
            manifest["vector_dimensions"]
        )

        expected_document_count = int(
            manifest["document_count"]
        )

        serialized_documents = json.loads(
            documents_path.read_text(
                encoding="utf-8"
            )
        )

        documents = [
            Document(
                page_content=item[
                    "page_content"
                ],
                metadata=item["metadata"],
            )
            for item in serialized_documents
        ]

        index = faiss.read_index(
            str(index_path)
        )

        if (
            len(documents)
            != expected_document_count
        ):
            raise ValueError(
                "Saved document count does not match "
                "the manifest."
            )

        return cls(
            index=index,
            documents=documents,
            dimensions=dimensions,
        )

    def search_by_vector(
        self,
        query_vector: list[float],
        *,
        access_context: SearchAccessContext,
        k: int = 5,
        minimum_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Finds semantically similar chunks and removes
        unauthorized results before returning them.

        Higher scores indicate greater similarity.
        """

        if k <= 0:
            raise ValueError(
                "Search result count 'k' must be "
                "greater than zero."
            )

        if not -1.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score must be between "
                "-1.0 and 1.0."
            )

        if len(query_vector) != self.dimensions:
            raise ValueError(
                "Query vector dimension does not "
                "match the FAISS index dimension."
            )

        if not all(
            isinstance(value, int | float)
            for value in query_vector
        ):
            raise ValueError(
                "Query vector contains a "
                "non-numeric value."
            )

        query_matrix = np.asarray(
            [query_vector],
            dtype=np.float32,
        )

        normalize_vector_matrix(
            query_matrix
        )

        # Local FAISS does not provide our banking metadata
        # filter. We retrieve all local candidates and apply
        # authorization before returning any document.
        candidate_count = self.index.ntotal

        if candidate_count == 0:
            return []

        scores, indexes = self.index.search(
            query_matrix,
            candidate_count,
        )

        authorized_results: list[
            SearchResult
        ] = []

        for score, document_index in zip(
            scores[0],
            indexes[0],
            strict=True,
        ):
            if document_index < 0:
                continue

            document = self.documents[
                int(document_index)
            ]

            numeric_score = float(score)

            if numeric_score < minimum_score:
                continue

            if not is_document_authorized(
                document,
                access_context=access_context,
            ):
                continue

            authorized_results.append(
                SearchResult(
                    document=document,
                    score=numeric_score,
                    rank=(
                        len(
                            authorized_results
                        )
                        + 1
                    ),
                )
            )

            if len(authorized_results) >= k:
                break

        return authorized_results


def normalize_vector_matrix(
    vector_matrix: NDArray[np.float32],
) -> None:
    """
    L2-normalizes vectors in place.

    FAISS IndexFlatIP then returns cosine-similarity
    scores.
    """

    if vector_matrix.ndim != 2:
        raise ValueError(
            "Vector matrix must be two-dimensional."
        )

    if vector_matrix.shape[0] == 0:
        raise ValueError(
            "Vector matrix cannot be empty."
        )

    norms = np.linalg.norm(
        vector_matrix,
        axis=1,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Zero-length vectors cannot be indexed "
            "or searched."
        )

    faiss.normalize_L2(
        vector_matrix
    )


def validate_indexable_document(
    document: Document,
) -> None:
    """
    Verifies that a chunk is ready for vector indexing.
    """

    required_metadata = (
        "chunk_id",
        "document_id",
        "version",
        "record_type",
        "embedding_status",
        "embedding_model",
        "embedding_dimensions",
        "retrieval_enabled",
        "allowed_roles",
        "allowed_regions",
        "classification_rank",
        "entitlement_key",
    )

    missing_fields = [
        field_name
        for field_name in required_metadata
        if field_name not in document.metadata
    ]

    if missing_fields:
        raise ValueError(
            "Chunk is missing required index metadata: "
            + ", ".join(missing_fields)
        )

    chunk_id = str(
        document.metadata["chunk_id"]
    )

    if not document.page_content.strip():
        raise ValueError(
            f"Chunk '{chunk_id}' has empty content."
        )

    if (
        document.metadata["record_type"]
        != "chunk"
    ):
        raise ValueError(
            f"Document '{chunk_id}' is not a chunk."
        )

    if (
        document.metadata["embedding_status"]
        != "COMPLETED"
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' has not completed "
            "embedding generation."
        )

    if (
        document.metadata["retrieval_enabled"]
        is not True
    ):
        raise ValueError(
            f"Chunk '{chunk_id}' is not enabled "
            "for retrieval."
        )

    pii_detected = (
        document.metadata.get(
            "pii_detected"
        )
        is True
    )

    pii_masked = (
        document.metadata.get(
            "pii_masked"
        )
        is True
    )

    if pii_detected and not pii_masked:
        raise ValueError(
            f"Chunk '{chunk_id}' contains "
            "unmasked PII."
        )


def is_document_authorized(
    document: Document,
    *,
    access_context: SearchAccessContext,
) -> bool:
    """
    Checks role, region, clearance, lifecycle, and
    retrieval eligibility.

    This check must happen before the chunk is sent to
    an LLM.
    """

    metadata = document.metadata

    if (
        metadata.get("retrieval_enabled")
        is not True
    ):
        return False

    allowed_roles = {
        str(role).strip().lower()
        for role in metadata.get(
            "allowed_roles",
            [],
        )
    }

    allowed_regions = {
        str(region).strip().upper()
        for region in metadata.get(
            "allowed_regions",
            [],
        )
    }

    classification_rank = metadata.get(
        "classification_rank"
    )

    if access_context.role not in allowed_roles:
        return False

    if access_context.region not in allowed_regions:
        return False

    if classification_rank is None:
        return False

    if (
        int(classification_rank)
        > access_context.clearance_rank
    ):
        return False

    return True