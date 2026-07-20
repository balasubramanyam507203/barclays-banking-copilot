from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from app.config import RerankerSettings
from app.rag.faiss_store import (
    SearchAccessContext,
    is_document_authorized,
)
from app.rag.keyword_index import (
    build_searchable_text,
    tokenize_for_keyword_search,
)


QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class RerankCandidate:
    """
    One candidate produced by hybrid retrieval.
    """

    document: Document

    hybrid_rank: int
    hybrid_score: float

    vector_rank: int | None
    vector_score: float | None

    keyword_rank: int | None
    keyword_score: float | None

    matched_by: tuple[str, ...]


@dataclass(frozen=True)
class RerankedResult:
    """
    One final result after second-stage reranking.
    """

    document: Document

    rank: int
    rerank_score: float

    reranker_backend: str
    reranker_model: str

    hybrid_rank: int
    hybrid_score: float

    vector_rank: int | None
    vector_score: float | None

    keyword_rank: int | None
    keyword_score: float | None

    matched_by: tuple[str, ...]


class Reranker(Protocol):
    """
    Interface used by the retrieval service.

    A local feature reranker, cross-encoder, or managed
    endpoint can implement the same interface.
    """

    def rerank(
        self,
        query: str,
        *,
        candidates: list[RerankCandidate],
        access_context: SearchAccessContext,
        top_n: int,
    ) -> list[RerankedResult]:
        ...


def normalize_text(value: str) -> str:
    """
    Normalizes text for deterministic local matching.
    """

    return " ".join(
        value.lower().strip().split()
    )


def get_meaningful_query_tokens(
    query: str,
) -> set[str]:
    """
    Removes common question words and preserves useful
    policy terms and identifiers.
    """

    all_tokens = tokenize_for_keyword_search(
        query
    )

    meaningful_tokens = {
        token
        for token in all_tokens
        if token not in QUERY_STOP_WORDS
    }

    if meaningful_tokens:
        return meaningful_tokens

    return set(all_tokens)


def calculate_token_coverage(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    """
    Calculates the percentage of meaningful query tokens
    present in the candidate text.
    """

    if not query_tokens:
        return 0.0

    matched_tokens = (
        query_tokens & document_tokens
    )

    return (
        len(matched_tokens)
        / len(query_tokens)
    )


def validate_rerank_candidates(
    candidates: list[RerankCandidate],
    *,
    access_context: SearchAccessContext,
) -> list[RerankCandidate]:
    """
    Applies defense-in-depth authorization validation
    before reranking.

    Hybrid retrieval already applies authorization, but
    the reranker verifies it again.
    """

    validated_candidates = []
    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        document = candidate.document

        chunk_id = str(
            document.metadata.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            raise ValueError(
                "Rerank candidate is missing chunk_id."
            )

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                f"Duplicate rerank candidate detected: "
                f"'{chunk_id}'."
            )

        seen_chunk_ids.add(chunk_id)

        if not is_document_authorized(
            document,
            access_context=access_context,
        ):
            continue

        validated_candidates.append(
            candidate
        )

    return validated_candidates


class LocalFeatureReranker:
    """
    Dependency-light local reranker.

    It combines:

    - Existing hybrid rank
    - Query-token coverage
    - Title-token coverage
    - Exact policy-ID matching
    - Exact phrase matching

    This is useful for local development and testing.

    Production can replace it with the cross-encoder
    adapter without changing the retrieval service.
    """

    backend_name = "local_feature"
    model_name = "local-feature-reranker-v1"

    def rerank(
        self,
        query: str,
        *,
        candidates: list[RerankCandidate],
        access_context: SearchAccessContext,
        top_n: int,
    ) -> list[RerankedResult]:
        """
        Reranks hybrid candidates using deterministic
        relevance features.
        """

        cleaned_query = normalize_text(query)

        if not cleaned_query:
            raise ValueError(
                "Reranker query cannot be empty."
            )

        if top_n <= 0:
            raise ValueError(
                "Reranker top_n must be greater "
                "than zero."
            )

        authorized_candidates = (
            validate_rerank_candidates(
                candidates,
                access_context=access_context,
            )
        )

        query_tokens = (
            get_meaningful_query_tokens(
                cleaned_query
            )
        )

        scored_candidates: list[
            tuple[RerankCandidate, float]
        ] = []

        for candidate in authorized_candidates:
            document = candidate.document
            metadata = document.metadata

            searchable_text = (
                build_searchable_text(
                    document
                )
            )

            normalized_searchable_text = (
                normalize_text(
                    searchable_text
                )
            )

            document_tokens = set(
                tokenize_for_keyword_search(
                    searchable_text
                )
            )

            title = str(
                metadata.get(
                    "title",
                    "",
                )
            )

            title_tokens = set(
                tokenize_for_keyword_search(
                    title
                )
            )

            document_id = normalize_text(
                str(
                    metadata.get(
                        "document_id",
                        "",
                    )
                )
            )

            content_coverage = (
                calculate_token_coverage(
                    query_tokens,
                    document_tokens,
                )
            )

            title_coverage = (
                calculate_token_coverage(
                    query_tokens,
                    title_tokens,
                )
            )

            exact_identifier_match = (
                1.0
                if (
                    document_id
                    and document_id
                    in cleaned_query
                )
                else 0.0
            )

            exact_phrase_match = (
                1.0
                if cleaned_query
                in normalized_searchable_text
                else 0.0
            )

            hybrid_rank_signal = (
                1.0
                / candidate.hybrid_rank
            )

            rerank_score = (
                0.25 * hybrid_rank_signal
                + 0.35 * content_coverage
                + 0.20 * title_coverage
                + 0.15 * exact_identifier_match
                + 0.05 * exact_phrase_match
            )

            scored_candidates.append(
                (
                    candidate,
                    rerank_score,
                )
            )

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].hybrid_rank,
                str(
                    item[0]
                    .document
                    .metadata
                    .get(
                        "chunk_id",
                        "",
                    )
                ),
            )
        )

        selected_candidates = (
            scored_candidates[:top_n]
        )

        return [
            RerankedResult(
                document=candidate.document,
                rank=rank,
                rerank_score=rerank_score,
                reranker_backend=(
                    self.backend_name
                ),
                reranker_model=(
                    self.model_name
                ),
                hybrid_rank=(
                    candidate.hybrid_rank
                ),
                hybrid_score=(
                    candidate.hybrid_score
                ),
                vector_rank=(
                    candidate.vector_rank
                ),
                vector_score=(
                    candidate.vector_score
                ),
                keyword_rank=(
                    candidate.keyword_rank
                ),
                keyword_score=(
                    candidate.keyword_score
                ),
                matched_by=(
                    candidate.matched_by
                ),
            )
            for rank, (
                candidate,
                rerank_score,
            ) in enumerate(
                selected_candidates,
                start=1,
            )
        ]


class SentenceTransformersCrossEncoderReranker:
    """
    Optional real cross-encoder adapter.

    The model is loaded lazily only when this backend is
    selected in the environment configuration.
    """

    backend_name = "sentence_transformers"

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
    ) -> None:
        try:
            from sentence_transformers import (
                CrossEncoder,
            )

        except ImportError as error:
            raise RuntimeError(
                "The sentence_transformers reranker "
                "backend requires the "
                "'sentence-transformers' package."
            ) from error

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        *,
        candidates: list[RerankCandidate],
        access_context: SearchAccessContext,
        top_n: int,
    ) -> list[RerankedResult]:
        """
        Scores each query-candidate pair with a real
        cross-encoder model.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "Reranker query cannot be empty."
            )

        if top_n <= 0:
            raise ValueError(
                "Reranker top_n must be greater "
                "than zero."
            )

        authorized_candidates = (
            validate_rerank_candidates(
                candidates,
                access_context=access_context,
            )
        )

        if not authorized_candidates:
            return []

        query_document_pairs = []

        for candidate in authorized_candidates:
            document = candidate.document
            metadata = document.metadata

            candidate_text = (
                f"Title: "
                f"{metadata.get('title', '')}\n"
                f"Document ID: "
                f"{metadata.get('document_id', '')}\n"
                f"Version: "
                f"{metadata.get('version', '')}\n"
                f"Content:\n"
                f"{document.page_content}"
            )

            query_document_pairs.append(
                (
                    cleaned_query,
                    candidate_text,
                )
            )

        raw_scores = self.model.predict(
            query_document_pairs
        )

        if len(raw_scores) != len(
            authorized_candidates
        ):
            raise RuntimeError(
                "Cross-encoder returned a different "
                "number of scores than candidates."
            )

        scored_candidates = [
            (
                candidate,
                float(score),
            )
            for candidate, score in zip(
                authorized_candidates,
                raw_scores,
                strict=True,
            )
        ]

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].hybrid_rank,
                str(
                    item[0]
                    .document
                    .metadata
                    .get(
                        "chunk_id",
                        "",
                    )
                ),
            )
        )

        selected_candidates = (
            scored_candidates[:top_n]
        )

        return [
            RerankedResult(
                document=candidate.document,
                rank=rank,
                rerank_score=rerank_score,
                reranker_backend=(
                    self.backend_name
                ),
                reranker_model=(
                    self.model_name
                ),
                hybrid_rank=(
                    candidate.hybrid_rank
                ),
                hybrid_score=(
                    candidate.hybrid_score
                ),
                vector_rank=(
                    candidate.vector_rank
                ),
                vector_score=(
                    candidate.vector_score
                ),
                keyword_rank=(
                    candidate.keyword_rank
                ),
                keyword_score=(
                    candidate.keyword_score
                ),
                matched_by=(
                    candidate.matched_by
                ),
            )
            for rank, (
                candidate,
                rerank_score,
            ) in enumerate(
                selected_candidates,
                start=1,
            )
        ]


def build_reranker(
    settings: RerankerSettings,
) -> Reranker:
    """
    Creates the configured reranker implementation.
    """

    if settings.backend == "local_feature":
        return LocalFeatureReranker()

    if (
        settings.backend
        == "sentence_transformers"
    ):
        return (
            SentenceTransformersCrossEncoderReranker(
                model_name=settings.model_name,
                device=settings.device,
            )
        )

    raise RuntimeError(
        f"Unsupported reranker backend: "
        f"'{settings.backend}'."
    )