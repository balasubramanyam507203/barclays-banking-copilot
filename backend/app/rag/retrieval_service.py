from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import (
    EmbeddingSettings,
    HybridRetrievalSettings,
    RerankerSettings,
)
from app.rag.embedding_service import (
    EmbeddingClient,
    embed_query_text,
)
from app.rag.faiss_store import (
    LocalFaissStore,
    SearchAccessContext,
    SearchResult,
)
from app.rag.keyword_index import (
    KeywordSearchResult,
    LocalBm25Index,
)
from app.rag.reranker import (
    RerankCandidate,
    Reranker,
)


MAX_QUERY_LENGTH = 2_000


@dataclass
class FusionAccumulator:
    """
    Internal mutable record used during rank fusion.
    """

    document: Document
    hybrid_score: float = 0.0

    vector_rank: int | None = None
    vector_score: float | None = None

    keyword_rank: int | None = None
    keyword_score: float | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """
    One final authorized and reranked chunk.
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

    citation: str


@dataclass(frozen=True)
class RetrievalResponse:
    """
    Complete retrieval and reranking response.
    """

    query: str
    access_context: SearchAccessContext
    results: list[RetrievedChunk]

    @property
    def result_count(self) -> int:
        return len(self.results)

    @property
    def documents(self) -> list[Document]:
        return [
            result.document
            for result in self.results
        ]


def normalize_query(query: str) -> str:
    """
    Cleans and validates a user query.
    """

    cleaned_query = " ".join(
        query.strip().split()
    )

    if not cleaned_query:
        raise ValueError(
            "Query cannot be empty."
        )

    if len(cleaned_query) > MAX_QUERY_LENGTH:
        raise ValueError(
            "Query cannot contain more than "
            f"{MAX_QUERY_LENGTH} characters."
        )

    return cleaned_query


def build_chunk_citation(
    document: Document,
) -> str:
    """
    Creates a human-readable citation.
    """

    metadata = document.metadata

    title = str(
        metadata.get(
            "title",
            "Untitled document",
        )
    )

    document_id = str(
        metadata.get(
            "document_id",
            "UNKNOWN",
        )
    )

    version = str(
        metadata.get(
            "version",
            "UNKNOWN",
        )
    )

    chunk_number = metadata.get(
        "chunk_number",
        "?"
    )

    total_chunks = metadata.get(
        "total_chunks",
        "?"
    )

    return (
        f"{title} "
        f"({document_id}, version {version}, "
        f"chunk {chunk_number}/{total_chunks})"
    )


def fuse_ranked_results(
    *,
    vector_results: list[SearchResult],
    keyword_results: list[
        KeywordSearchResult
    ],
    candidate_top_k: int,
    settings: HybridRetrievalSettings,
) -> list[RerankCandidate]:
    """
    Combines semantic and keyword rankings using
    weighted reciprocal-rank fusion.

    It returns candidates for the reranking stage.
    """

    if candidate_top_k <= 0:
        raise ValueError(
            "candidate_top_k must be greater "
            "than zero."
        )

    accumulators: dict[
        str,
        FusionAccumulator,
    ] = {}

    for result in vector_results:
        chunk_id = str(
            result.document.metadata["chunk_id"]
        )

        accumulator = accumulators.setdefault(
            chunk_id,
            FusionAccumulator(
                document=result.document
            ),
        )

        accumulator.vector_rank = result.rank
        accumulator.vector_score = result.score

        accumulator.hybrid_score += (
            settings.vector_weight
            / (
                settings.reciprocal_rank_constant
                + result.rank
            )
        )

    for result in keyword_results:
        chunk_id = str(
            result.document.metadata["chunk_id"]
        )

        accumulator = accumulators.setdefault(
            chunk_id,
            FusionAccumulator(
                document=result.document
            ),
        )

        accumulator.keyword_rank = result.rank
        accumulator.keyword_score = result.score

        accumulator.hybrid_score += (
            settings.keyword_weight
            / (
                settings.reciprocal_rank_constant
                + result.rank
            )
        )

    ordered_accumulators = sorted(
        accumulators.values(),
        key=lambda item: (
            -item.hybrid_score,
            (
                item.vector_rank
                if item.vector_rank is not None
                else 1_000_000
            ),
            (
                item.keyword_rank
                if item.keyword_rank is not None
                else 1_000_000
            ),
            str(
                item.document.metadata.get(
                    "chunk_id",
                    "",
                )
            ),
        ),
    )

    selected_accumulators = (
        ordered_accumulators[:candidate_top_k]
    )

    candidates = []

    for hybrid_rank, item in enumerate(
        selected_accumulators,
        start=1,
    ):
        matched_by = []

        if item.vector_rank is not None:
            matched_by.append(
                "semantic"
            )

        if item.keyword_rank is not None:
            matched_by.append(
                "keyword"
            )

        candidates.append(
            RerankCandidate(
                document=item.document,
                hybrid_rank=hybrid_rank,
                hybrid_score=item.hybrid_score,
                vector_rank=item.vector_rank,
                vector_score=item.vector_score,
                keyword_rank=item.keyword_rank,
                keyword_score=item.keyword_score,
                matched_by=tuple(
                    matched_by
                ),
            )
        )

    return candidates


class PermissionAwareRetrievalService:
    """
    Permission-aware retrieval and reranking service.

    Query path:

    1. Query validation
    2. Query embedding
    3. Semantic retrieval
    4. Keyword retrieval
    5. Reciprocal-rank fusion
    6. Reranking
    7. Final authorized evidence
    """

    def __init__(
        self,
        *,
        vector_store: LocalFaissStore,
        keyword_index: LocalBm25Index,
        embedding_client: EmbeddingClient,
        embedding_settings: EmbeddingSettings,
        hybrid_settings: HybridRetrievalSettings,
        reranker: Reranker,
        reranker_settings: RerankerSettings,
        default_top_k: int = 5,
        default_vector_minimum_score: float = 0.0,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError(
                "default_top_k must be greater "
                "than zero."
            )

        if not (
            -1.0
            <= default_vector_minimum_score
            <= 1.0
        ):
            raise ValueError(
                "default_vector_minimum_score must be "
                "between -1.0 and 1.0."
            )

        self.vector_store = vector_store
        self.keyword_index = keyword_index

        self.embedding_client = embedding_client
        self.embedding_settings = embedding_settings

        self.hybrid_settings = hybrid_settings

        self.reranker = reranker
        self.reranker_settings = (
            reranker_settings
        )

        self.default_top_k = default_top_k

        self.default_vector_minimum_score = (
            default_vector_minimum_score
        )

        self._validate_index_compatibility()

    def _validate_index_compatibility(
        self,
    ) -> None:
        """
        Ensures query and indexed embeddings use the
        same model and dimensions.
        """

        if self.vector_store.document_count == 0:
            raise ValueError(
                "The vector store contains no documents."
            )

        if (
            self.vector_store.dimensions
            != self.embedding_settings.dimensions
        ):
            raise ValueError(
                "The configured query embedding "
                "dimensions do not match the FAISS "
                "index dimensions."
            )

        indexed_models = {
            str(
                document.metadata.get(
                    "embedding_model",
                    "",
                )
            ).strip()
            for document
            in self.vector_store.documents
        }

        if "" in indexed_models:
            raise ValueError(
                "One or more indexed chunks do not "
                "contain embedding-model metadata."
            )

        if len(indexed_models) != 1:
            raise ValueError(
                "The FAISS index contains chunks "
                "generated by multiple embedding models."
            )

        indexed_model = next(
            iter(indexed_models)
        )

        if (
            indexed_model
            != self.embedding_settings.model
        ):
            raise ValueError(
                "The query embedding model does not "
                "match the model used to build the index. "
                f"Configured: "
                f"'{self.embedding_settings.model}'. "
                f"Index: '{indexed_model}'."
            )

    def retrieve(
        self,
        query: str,
        *,
        access_context: SearchAccessContext,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """
        Retrieves, fuses, reranks, and returns the final
        authorized evidence chunks.
        """

        cleaned_query = normalize_query(
            query
        )

        effective_top_k = (
            self.default_top_k
            if top_k is None
            else top_k
        )

        if effective_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_vector = embed_query_text(
            cleaned_query,
            embedding_client=(
                self.embedding_client
            ),
            expected_dimensions=(
                self.embedding_settings.dimensions
            ),
        )

        vector_results = (
            self.vector_store.search_by_vector(
                query_vector,
                access_context=access_context,
                k=(
                    self.hybrid_settings
                    .vector_candidate_k
                ),
                minimum_score=(
                    self
                    .default_vector_minimum_score
                ),
            )
        )

        keyword_results = (
            self.keyword_index.search(
                cleaned_query,
                access_context=access_context,
                k=(
                    self.hybrid_settings
                    .keyword_candidate_k
                ),
                minimum_score=(
                    self.hybrid_settings
                    .keyword_minimum_score
                ),
            )
        )

        rerank_candidates = (
            fuse_ranked_results(
                vector_results=vector_results,
                keyword_results=keyword_results,
                candidate_top_k=(
                    self.reranker_settings
                    .candidate_k
                ),
                settings=(
                    self.hybrid_settings
                ),
            )
        )

        reranked_results = self.reranker.rerank(
            cleaned_query,
            candidates=rerank_candidates,
            access_context=access_context,
            top_n=min(
                effective_top_k,
                self.reranker_settings.top_n,
            ),
        )

        retrieved_chunks = [
            RetrievedChunk(
                document=result.document,
                rank=result.rank,
                rerank_score=(
                    result.rerank_score
                ),
                reranker_backend=(
                    result.reranker_backend
                ),
                reranker_model=(
                    result.reranker_model
                ),
                hybrid_rank=(
                    result.hybrid_rank
                ),
                hybrid_score=(
                    result.hybrid_score
                ),
                vector_rank=(
                    result.vector_rank
                ),
                vector_score=(
                    result.vector_score
                ),
                keyword_rank=(
                    result.keyword_rank
                ),
                keyword_score=(
                    result.keyword_score
                ),
                matched_by=(
                    result.matched_by
                ),
                citation=build_chunk_citation(
                    result.document
                ),
            )
            for result in reranked_results
        ]

        return RetrievalResponse(
            query=cleaned_query,
            access_context=access_context,
            results=retrieved_chunks,
        )

    def retrieve_documents(
        self,
        query: str,
        *,
        access_context: SearchAccessContext,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Returns only the final reranked LangChain
        Documents.
        """

        response = self.retrieve(
            query,
            access_context=access_context,
            top_k=top_k,
        )

        return response.documents