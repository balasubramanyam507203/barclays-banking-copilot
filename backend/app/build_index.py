from typing import Any

from langchain_core.documents import Document
from pydantic import ValidationError

from app.config import (
    get_embedding_settings,
    get_vector_store_settings,
)
from app.ingestion.document_mapper import (
    map_raw_document_to_banking_document,
)
from app.ingestion.loaders import (
    load_all_text_documents,
)
from app.models.banking_document import (
    BankingDocument,
)
from app.preprocessing.cleaner import (
    clean_banking_document,
)
from app.preprocessing.deduplication import (
    detect_duplicate_and_version_issues,
)
from app.preprocessing.metadata_enricher import (
    enrich_document_metadata,
)
from app.preprocessing.pii_masker import (
    mask_pii_in_document,
)
from app.rag.embedding_service import (
    EmbeddedChunk,
    build_embedding_client,
    embed_chunk_documents,
)
from app.rag.faiss_store import (
    LocalFaissStore,
)
from app.rag.langchain_converter import (
    convert_to_langchain_documents,
    select_retrievable_documents,
)
from app.rag.text_splitter import (
    DEFAULT_CHUNKING_CONFIG,
    split_retrievable_documents,
)


def main() -> None:
    """
    Runs the complete ingestion pipeline and creates
    the local FAISS policy index.

    This program runs when source documents are added,
    changed, deleted, or reprocessed.

    It does not handle user questions.
    """

    document_folder = "../sample_documents"

    raw_documents = (
        load_all_text_documents(
            document_folder
        )
    )

    validated_documents: list[
        BankingDocument
    ] = []

    cleaned_documents: list[
        BankingDocument
    ] = []

    rejected_documents: list[
        dict[str, Any]
    ] = []

    # Step 1:
    # Source-specific mapping and Pydantic validation.
    #
    # Step 2:
    # Cleaning and normalization.
    for raw_document in raw_documents:
        try:
            validated_document = (
                map_raw_document_to_banking_document(
                    raw_document
                )
            )

            validated_documents.append(
                validated_document
            )

            cleaned_document = (
                clean_banking_document(
                    validated_document
                )
            )

            cleaned_documents.append(
                cleaned_document
            )

        except (
            ValueError,
            ValidationError,
        ) as error:
            rejected_documents.append(
                {
                    "file_name": (
                        raw_document["file_name"]
                    ),
                    "stage": (
                        "VALIDATION_OR_CLEANING"
                    ),
                    "reasons": [str(error)],
                }
            )

    # Step 3:
    # Exact duplicate and version-governance checks.
    (
        accepted_documents,
        duplicate_rejections,
    ) = detect_duplicate_and_version_issues(
        cleaned_documents
    )

    rejected_documents.extend(
        duplicate_rejections
    )

    # Step 4:
    # PII detection and masking.
    pii_processed_documents = [
        mask_pii_in_document(document)
        for document in accepted_documents
    ]

    # Step 5:
    # Permission, lifecycle, ownership, citation,
    # and lineage enrichment.
    enriched_documents = [
        enrich_document_metadata(document)
        for document in pii_processed_documents
    ]

    # Step 6:
    # Internal BankingDocument to LangChain Document.
    langchain_documents: list[Document] = (
        convert_to_langchain_documents(
            enriched_documents
        )
    )

    # Step 7:
    # Exclude future-effective, expired, archived,
    # and superseded documents from the searchable index.
    retrievable_documents: list[Document] = (
        select_retrievable_documents(
            langchain_documents
        )
    )

    # Step 8:
    # Parent documents to token-aware chunk documents.
    chunk_documents: list[Document] = (
        split_retrievable_documents(
            retrievable_documents,
            config=DEFAULT_CHUNKING_CONFIG,
        )
    )

    # Step 9:
    # Generate vectors for every eligible chunk.
    embedding_settings = (
        get_embedding_settings()
    )

    embedding_client = (
        build_embedding_client(
            embedding_settings
        )
    )

    embedded_chunks: list[EmbeddedChunk] = (
        embed_chunk_documents(
            chunk_documents,
            embedding_client=(
                embedding_client
            ),
            settings=embedding_settings,
        )
    )

    # Step 10:
    # Create and persist the local FAISS index.
    vector_store_settings = (
        get_vector_store_settings()
    )

    vector_store = (
        LocalFaissStore.from_embedded_chunks(
            embedded_chunks
        )
    )

    vector_store.save(
        vector_store_settings.index_path
    )

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETED")

    print(
        f"\nRaw documents loaded: "
        f"{len(raw_documents)}"
    )

    print(
        f"Validated documents: "
        f"{len(validated_documents)}"
    )

    print(
        f"Cleaned documents: "
        f"{len(cleaned_documents)}"
    )

    print(
        "Accepted after duplicate/version checks: "
        f"{len(accepted_documents)}"
    )

    print(
        f"PII processed documents: "
        f"{len(pii_processed_documents)}"
    )

    print(
        f"Metadata enriched documents: "
        f"{len(enriched_documents)}"
    )

    print(
        f"LangChain parent documents: "
        f"{len(langchain_documents)}"
    )

    print(
        f"Retrievable parent documents: "
        f"{len(retrievable_documents)}"
    )

    print(
        f"Chunk documents created: "
        f"{len(chunk_documents)}"
    )

    print(
        f"Embedded chunks created: "
        f"{len(embedded_chunks)}"
    )

    print(
        f"FAISS vectors stored: "
        f"{vector_store.document_count}"
    )

    print(
        f"Embedding model: "
        f"{embedding_settings.model}"
    )

    print(
        f"Vector dimensions: "
        f"{vector_store.dimensions}"
    )

    print(
        f"Index saved to: "
        f"{vector_store_settings.index_path}"
    )

    print(
        f"Rejected documents: "
        f"{len(rejected_documents)}"
    )

    if rejected_documents:
        print("\nRejected documents:")

        for rejected_document in rejected_documents:
            print("\n" + "-" * 70)

            print(
                f"File: "
                f"{rejected_document['file_name']}"
            )

            for reason in rejected_document[
                "reasons"
            ]:
                print(
                    f"Reason: {reason}"
                )


if __name__ == "__main__":
    main()