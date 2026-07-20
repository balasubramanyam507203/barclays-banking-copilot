import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

from app.rag.faiss_store import (
    SearchAccessContext,
    is_document_authorized,
)


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+"
    r"(?:[.\-_/][a-z0-9]+)*"
)


@dataclass(frozen=True)
class KeywordSearchResult:
    """
    One authorized BM25 keyword-search result.
    """

    document: Document
    score: float
    rank: int


def tokenize_for_keyword_search(
    text: str,
) -> list[str]:
    """
    Converts text into normalized keyword tokens.

    The tokenizer preserves terms such as:

    PAY-POL-1042
    Section 4.3
    KYC
    AML
    """

    return TOKEN_PATTERN.findall(
        text.lower()
    )


def build_searchable_text(
    document: Document,
) -> str:
    """
    Combines searchable metadata and chunk content.

    This allows exact keyword searches to match document
    IDs and titles even when they do not appear inside the
    chunk text.
    """

    metadata = document.metadata

    searchable_metadata = [
        str(
            metadata.get(
                "document_id",
                "",
            )
        ),
        str(
            metadata.get(
                "title",
                "",
            )
        ),
        str(
            metadata.get(
                "version",
                "",
            )
        ),
        str(
            metadata.get(
                "department",
                "",
            )
        ),
        str(
            metadata.get(
                "document_type",
                "",
            )
        ),
    ]

    return "\n".join(
        [
            *searchable_metadata,
            document.page_content,
        ]
    )


def calculate_metadata_exact_match_boost(
    *,
    query: str,
    document: Document,
) -> float:
    """
    Adds controlled boosts for exact document identifiers
    and exact title phrases.

    BM25 still performs the general keyword ranking.
    """

    normalized_query = " ".join(
        query.lower().split()
    )

    metadata = document.metadata

    document_id = str(
        metadata.get(
            "document_id",
            "",
        )
    ).strip().lower()

    title = " ".join(
        str(
            metadata.get(
                "title",
                "",
            )
        ).lower().split()
    )

    version = str(
        metadata.get(
            "version",
            "",
        )
    ).strip().lower()

    boost = 0.0

    if (
        document_id
        and document_id in normalized_query
    ):
        boost += 8.0

    if title and title in normalized_query:
        boost += 4.0

    if (
        version
        and f"version {version}"
        in normalized_query
    ):
        boost += 1.5

    return boost


class LocalBm25Index:
    """
    Local BM25 keyword-search index.

    This is used only for local development.

    In production, OpenSearch performs keyword BM25 and
    vector retrieval inside the same secured search engine.
    """

    def __init__(
        self,
        *,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError(
                "At least one document is required "
                "to build the BM25 index."
            )

        if k1 <= 0:
            raise ValueError(
                "BM25 k1 must be greater than zero."
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "BM25 b must be between zero and one."
            )

        self.documents = documents
        self.k1 = k1
        self.b = b

    def search(
        self,
        query: str,
        *,
        access_context: SearchAccessContext,
        k: int = 20,
        minimum_score: float = 0.0,
    ) -> list[KeywordSearchResult]:
        """
        Searches only documents authorized for the current
        user.

        Local BM25 statistics are calculated from the
        authorized corpus for this request.
        """

        cleaned_query = " ".join(
            query.strip().split()
        )

        if not cleaned_query:
            raise ValueError(
                "Keyword query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "Keyword result count 'k' must be "
                "greater than zero."
            )

        if minimum_score < 0:
            raise ValueError(
                "Keyword minimum score cannot be "
                "negative."
            )

        query_tokens = tokenize_for_keyword_search(
            cleaned_query
        )

        if not query_tokens:
            return []

        authorized_documents = [
            document
            for document in self.documents
            if is_document_authorized(
                document,
                access_context=access_context,
            )
        ]

        if not authorized_documents:
            return []

        tokenized_documents = [
            tokenize_for_keyword_search(
                build_searchable_text(document)
            )
            for document in authorized_documents
        ]

        average_document_length = (
            sum(
                len(tokens)
                for tokens in tokenized_documents
            )
            / len(tokenized_documents)
        )

        if average_document_length <= 0:
            return []

        document_frequencies = (
            self._calculate_document_frequencies(
                tokenized_documents
            )
        )

        scored_documents: list[
            tuple[Document, float]
        ] = []

        for document, document_tokens in zip(
            authorized_documents,
            tokenized_documents,
            strict=True,
        ):
            bm25_score = self._calculate_bm25_score(
                query_tokens=query_tokens,
                document_tokens=document_tokens,
                document_frequencies=(
                    document_frequencies
                ),
                corpus_size=len(
                    authorized_documents
                ),
                average_document_length=(
                    average_document_length
                ),
            )

            metadata_boost = (
                calculate_metadata_exact_match_boost(
                    query=cleaned_query,
                    document=document,
                )
            )

            total_score = (
                bm25_score + metadata_boost
            )

            # Do not return documents with no keyword
            # evidence, even when minimum_score is zero.
            if total_score <= minimum_score:
                continue

            scored_documents.append(
                (
                    document,
                    total_score,
                )
            )

        scored_documents.sort(
            key=lambda item: (
                -item[1],
                str(
                    item[0].metadata.get(
                        "chunk_id",
                        "",
                    )
                ),
            )
        )

        top_documents = scored_documents[:k]

        return [
            KeywordSearchResult(
                document=document,
                score=score,
                rank=rank,
            )
            for rank, (
                document,
                score,
            ) in enumerate(
                top_documents,
                start=1,
            )
        ]

    @staticmethod
    def _calculate_document_frequencies(
        tokenized_documents: list[list[str]],
    ) -> Counter[str]:
        """
        Counts how many documents contain each term.
        """

        document_frequencies: Counter[str] = (
            Counter()
        )

        for tokens in tokenized_documents:
            document_frequencies.update(
                set(tokens)
            )

        return document_frequencies

    def _calculate_bm25_score(
        self,
        *,
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequencies: Counter[str],
        corpus_size: int,
        average_document_length: float,
    ) -> float:
        """
        Calculates one BM25 relevance score.
        """

        term_frequencies = Counter(
            document_tokens
        )

        document_length = len(
            document_tokens
        )

        score = 0.0

        for query_term in set(query_tokens):
            term_frequency = term_frequencies.get(
                query_term,
                0,
            )

            if term_frequency == 0:
                continue

            document_frequency = (
                document_frequencies.get(
                    query_term,
                    0,
                )
            )

            inverse_document_frequency = math.log(
                1
                + (
                    corpus_size
                    - document_frequency
                    + 0.5
                )
                / (
                    document_frequency
                    + 0.5
                )
            )

            length_normalization = (
                1
                - self.b
                + self.b
                * (
                    document_length
                    / average_document_length
                )
            )

            numerator = (
                term_frequency
                * (
                    self.k1 + 1
                )
            )

            denominator = (
                term_frequency
                + self.k1
                * length_normalization
            )

            score += (
                inverse_document_frequency
                * numerator
                / denominator
            )

        return score