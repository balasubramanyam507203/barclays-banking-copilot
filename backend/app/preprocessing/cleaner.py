import re
import unicodedata

from app.models.banking_document import BankingDocument


CONTROL_CHARACTER_PATTERN = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
MULTIPLE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def normalize_unicode(text: str) -> str:
    """
    Converts visually equivalent Unicode characters into
    a consistent representation.
    """

    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    """
    Removes invisible control characters that can interfere
    with parsing, chunking, logging, or indexing.
    """

    return CONTROL_CHARACTER_PATTERN.sub("", text)


def normalize_line_endings(text: str) -> str:
    """
    Converts Windows and older Mac line endings into
    the standard newline character.
    """

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_lines(text: str) -> str:
    """
    Removes unnecessary spaces while preserving paragraph
    and section boundaries.
    """

    cleaned_lines = []

    for line in text.splitlines():
        cleaned_line = MULTIPLE_SPACES_PATTERN.sub(
            " ",
            line,
        ).strip()

        cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def reduce_blank_lines(text: str) -> str:
    """
    Replaces three or more consecutive newline characters
    with two newline characters.
    """

    return MULTIPLE_BLANK_LINES_PATTERN.sub("\n\n", text)


def clean_text(text: str) -> str:
    """
    Runs the complete deterministic text-cleaning pipeline.
    """

    cleaned_text = normalize_unicode(text)
    cleaned_text = remove_control_characters(cleaned_text)
    cleaned_text = normalize_line_endings(cleaned_text)
    cleaned_text = normalize_lines(cleaned_text)
    cleaned_text = reduce_blank_lines(cleaned_text)

    return cleaned_text.strip()


def clean_banking_document(
    document: BankingDocument,
) -> BankingDocument:
    """
    Returns a new BankingDocument containing cleaned content.

    The original object is not modified.
    """

    cleaned_content = clean_text(document.content)

    if not cleaned_content:
        raise ValueError(
            f"Document '{document.file_name}' became empty "
            "after cleaning"
        )

    return document.model_copy(
        update={
            "content": cleaned_content,
        }
    )


def clean_banking_documents(
    documents: list[BankingDocument],
) -> list[BankingDocument]:
    """
    Cleans a collection of validated banking documents.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_document = clean_banking_document(document)
        cleaned_documents.append(cleaned_document)

    return cleaned_documents