import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """
    Configuration used when splitting parent documents.

    Chunk size and overlap are measured in tokens.
    """

    chunk_size_tokens: int = 700
    chunk_overlap_tokens: int = 100
    encoding_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError(
                "chunk_size_tokens must be greater than zero"
            )

        if self.chunk_overlap_tokens < 0:
            raise ValueError(
                "chunk_overlap_tokens cannot be negative"
            )

        if (
            self.chunk_overlap_tokens
            >= self.chunk_size_tokens
        ):
            raise ValueError(
                "chunk_overlap_tokens must be smaller than "
                "chunk_size_tokens"
            )


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


REQUIRED_SECURITY_METADATA = (
    "parent_document_key",
    "document_id",
    "version",
    "allowed_roles",
    "allowed_regions",
    "classification_rank",
    "entitlement_key",
    "retrieval_enabled",
)


@lru_cache(maxsize=10)
def get_token_encoder(
    encoding_name: str,
) -> Any:
    """
    Loads and caches a tiktoken encoder.

    Caching prevents the tokenizer from being loaded
    repeatedly for every chunk.
    """

    return tiktoken.get_encoding(encoding_name)


def count_tokens(
    text: str,
    *,
    encoding_name: str,
) -> int:
    """
    Counts the approximate number of model tokens
    contained in a piece of text.
    """

    encoder = get_token_encoder(encoding_name)

    return len(encoder.encode(text))


def create_chunk_hash(
    chunk_content: str,
) -> str:
    """
    Creates a SHA-256 fingerprint for one chunk.
    """

    normalized_content = chunk_content.strip()

    return hashlib.sha256(
        normalized_content.encode("utf-8")
    ).hexdigest()


def create_chunk_id(
    *,
    parent_document_key: str,
    chunk_index: int,
    chunk_hash: str,
) -> str:
    """
    Creates a deterministic ID for one chunk.

    The same parent, index, and content will produce
    the same chunk ID.
    """

    identity_value = (
        f"{parent_document_key}"
        f"|{chunk_index}"
        f"|{chunk_hash}"
    )

    identity_hash = hashlib.sha256(
        identity_value.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"{parent_document_key}"
        f":chunk-{chunk_index:04d}"
        f":{identity_hash}"
    )


def validate_parent_document(
    document: Document,
) -> None:
    """
    Verifies that a document is safe and ready for
    chunking.
    """

    if not document.page_content.strip():
        raise ValueError(
            "Parent document content cannot be empty"
        )

    missing_fields = [
        field_name
        for field_name in REQUIRED_SECURITY_METADATA
        if field_name not in document.metadata
    ]

    if missing_fields:
        fields_text = ", ".join(missing_fields)

        raise ValueError(
            "Parent document is missing required metadata: "
            f"{fields_text}"
        )

    if (
        document.metadata.get("record_type")
        != "parent_document"
    ):
        raise ValueError(
            "Only parent documents can be passed to "
            "the text splitter"
        )

    if (
        document.metadata.get("retrieval_enabled")
        is not True
    ):
        raise ValueError(
            "Only retrieval-enabled documents can be "
            "chunked for the searchable index"
        )

    if not document.metadata.get("allowed_roles"):
        raise ValueError(
            "Parent document must contain allowed roles"
        )

    if not document.metadata.get("allowed_regions"):
        raise ValueError(
            "Parent document must contain allowed regions"
        )

    if (
        document.metadata.get("classification_rank")
        is None
    ):
        raise ValueError(
            "Parent document must contain a "
            "classification rank"
        )

    if not document.metadata.get("entitlement_key"):
        raise ValueError(
            "Parent document must contain an "
            "entitlement key"
        )


def build_text_splitter(
    config: ChunkingConfig,
) -> RecursiveCharacterTextSplitter:
    """
    Creates a token-aware recursive text splitter.

    The separator order means:

    1. Try to keep paragraphs together.
    2. Then try to keep individual lines together.
    3. Then split using spaces.
    4. Finally split individual characters only when
       absolutely necessary.
    """

    return (
        RecursiveCharacterTextSplitter
        .from_tiktoken_encoder(
            encoding_name=config.encoding_name,
            chunk_size=config.chunk_size_tokens,
            chunk_overlap=(
                config.chunk_overlap_tokens
            ),
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )
    )


def build_chunk_metadata(
    *,
    parent_document: Document,
    chunk_content: str,
    chunk_index: int,
    total_chunks: int,
    config: ChunkingConfig,
) -> dict[str, Any]:
    """
    Copies parent metadata and adds chunk-specific
    metadata.
    """

    parent_metadata = dict(
        parent_document.metadata
    )

    parent_document_key = str(
        parent_metadata["parent_document_key"]
    )

    chunk_hash = create_chunk_hash(
        chunk_content
    )

    chunk_id = create_chunk_id(
        parent_document_key=parent_document_key,
        chunk_index=chunk_index,
        chunk_hash=chunk_hash,
    )

    chunk_token_count = count_tokens(
        chunk_content,
        encoding_name=config.encoding_name,
    )

    return {
        **parent_metadata,

        # Chunk identity
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "chunk_number": chunk_index + 1,
        "total_chunks": total_chunks,
        "chunk_hash": chunk_hash,

        # Chunk configuration
        "chunk_token_count": chunk_token_count,
        "configured_chunk_size_tokens": (
            config.chunk_size_tokens
        ),
        "configured_chunk_overlap_tokens": (
            config.chunk_overlap_tokens
        ),
        "token_encoding": config.encoding_name,
        "chunking_strategy": (
            "recursive_character_tiktoken"
        ),

        # Processing stage
        "record_type": "chunk",
        "embedding_status": "PENDING",
    }


def split_parent_document(
    document: Document,
    *,
    config: ChunkingConfig = (
        DEFAULT_CHUNKING_CONFIG
    ),
) -> list[Document]:
    """
    Splits one parent LangChain Document into multiple
    chunk LangChain Documents.
    """

    validate_parent_document(document)

    text_splitter = build_text_splitter(
        config
    )

    raw_chunks = text_splitter.split_text(
        document.page_content
    )

    chunk_contents = [
        chunk.strip()
        for chunk in raw_chunks
        if chunk.strip()
    ]

    if not chunk_contents:
        raise ValueError(
            "Text splitter did not produce any chunks"
        )

    total_chunks = len(chunk_contents)

    chunk_documents = []

    for chunk_index, chunk_content in enumerate(
        chunk_contents
    ):
        chunk_metadata = build_chunk_metadata(
            parent_document=document,
            chunk_content=chunk_content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            config=config,
        )

        chunk_document = Document(
            page_content=chunk_content,
            metadata=chunk_metadata,
        )

        chunk_documents.append(
            chunk_document
        )

    return chunk_documents


def split_retrievable_documents(
    documents: list[Document],
    *,
    config: ChunkingConfig = (
        DEFAULT_CHUNKING_CONFIG
    ),
) -> list[Document]:
    """
    Splits multiple retrieval-enabled parent documents
    into chunk documents.
    """

    all_chunks = []

    for document in documents:
        document_chunks = split_parent_document(
            document,
            config=config,
        )

        all_chunks.extend(
            document_chunks
        )

    return all_chunks