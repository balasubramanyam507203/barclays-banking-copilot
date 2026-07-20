from langchain_core.documents import (
    Document,
)
from fastapi import (
    APIRouter,
    HTTPException,
    Path,
    status,
)

from app.api.dependencies import (
    AccessContextDependency,
    ServicesDependency,
)
from app.api.schemas import (
    SourceChunkResponse,
    SourceDocumentResponse,
)
from app.rag.faiss_store import (
    is_document_authorized,
)


router = APIRouter(
    prefix="/sources",
    tags=["Sources"],
)


def get_chunk_number(
    document: Document,
) -> int:
    """
    Safely returns a chunk number for sorting.
    """

    raw_chunk_number = (
        document.metadata.get(
            "chunk_number",
            0,
        )
    )

    try:
        return int(raw_chunk_number)

    except (
        TypeError,
        ValueError,
    ):
        return 0


def find_authorized_document_chunks(
    *,
    document_id: str,
    services: ServicesDependency,
    access_context: AccessContextDependency,
) -> list[Document]:
    """
    Finds chunks only when the employee is authorized.

    Missing and unauthorized documents both return an
    empty list to avoid revealing document existence.
    """

    normalized_document_id = (
        document_id.strip().lower()
    )

    authorized_chunks = [
        document
        for document
        in services.vector_store.documents
        if (
            str(
                document.metadata.get(
                    "document_id",
                    "",
                )
            ).strip().lower()
            == normalized_document_id
        )
        and is_document_authorized(
            document,
            access_context=access_context,
        )
    ]

    return sorted(
        authorized_chunks,
        key=get_chunk_number,
    )


@router.get(
    "/{document_id}",
    response_model=SourceDocumentResponse,
)
def get_source_document(
    services: ServicesDependency,
    access_context: AccessContextDependency,
    document_id: str = Path(
        min_length=1,
        max_length=200,
        description=(
            "Policy document identifier."
        ),
    ),
) -> SourceDocumentResponse:
    """
    Returns authorized source chunks for citation
    inspection.
    """

    authorized_chunks = (
        find_authorized_document_chunks(
            document_id=document_id,
            services=services,
            access_context=access_context,
        )
    )

    if not authorized_chunks:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Source document not found.",
        )

    first_document = (
        authorized_chunks[0]
    )

    metadata = first_document.metadata

    return SourceDocumentResponse(
        document_id=str(
            metadata.get(
                "document_id",
                document_id,
            )
        ),
        title=str(
            metadata.get(
                "title",
                "Unknown",
            )
        ),
        version=str(
            metadata.get(
                "version",
                "Unknown",
            )
        ),
        source=str(
            metadata.get(
                "source",
                "Unknown",
            )
        ),
        chunks=[
            SourceChunkResponse(
                chunk_id=str(
                    document.metadata.get(
                        "chunk_id",
                        "",
                    )
                ),
                chunk_number=get_chunk_number(
                    document
                ),
                total_chunks=int(
                    document.metadata.get(
                        "total_chunks",
                        len(authorized_chunks),
                    )
                ),
                content=(
                    document.page_content
                ),
            )
            for document
            in authorized_chunks
        ],
    )