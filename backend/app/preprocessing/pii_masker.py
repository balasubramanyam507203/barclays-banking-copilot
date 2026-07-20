import re
from collections import Counter
from collections.abc import Callable
from re import Match, Pattern

from app.models.banking_document import BankingDocument


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

SSN_PATTERN = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(\d{3}\)|\d{3})"
    r"[\s.-]\d{3}[\s.-]\d{4}"
    r"(?!\d)"
)

ACCOUNT_NUMBER_PATTERN = re.compile(
    r"(?i)\b"
    r"(account(?:\s+number)?|acct(?:\s+number)?)"
    r"\s*[:#-]?\s*"
    r"([A-Z0-9-]{6,20})"
    r"\b"
)

CUSTOMER_ID_PATTERN = re.compile(
    r"(?i)\b"
    r"(customer(?:\s+id|\s+number|\s+reference)|"
    r"customer\s+identifier)"
    r"\s*[:#-]?\s*"
    r"([A-Z0-9-]{4,30})"
    r"\b"
)

DATE_OF_BIRTH_PATTERN = re.compile(
    r"(?i)\b"
    r"(date\s+of\s+birth|dob)"
    r"\s*[:#-]?\s*"
    r"("
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    r"\d{1,2}/\d{1,2}/\d{2,4}"
    r")"
    r"\b"
)

PERSON_LABEL_PATTERN = re.compile(
    r"(?im)\b"
    r"(customer\s+name|full\s+name)"
    r"\s*:\s*"
    r"("
    r"[A-Z][A-Za-z'.-]+"
    r"(?:\s+[A-Z][A-Za-z'.-]+){1,3}"
    r")"
    r"\b"
)


def replace_simple_pattern(
    *,
    text: str,
    pattern: Pattern[str],
    entity_type: str,
    placeholder: str,
    findings: Counter[str],
) -> str:
    """
    Replaces an unlabeled PII value such as an email,
    phone number, or SSN.
    """

    def replacement(_: Match[str]) -> str:
        findings[entity_type] += 1
        return placeholder

    return pattern.sub(replacement, text)


def replace_labeled_pattern(
    *,
    text: str,
    pattern: Pattern[str],
    entity_type: str,
    placeholder: str,
    findings: Counter[str],
) -> str:
    """
    Replaces the sensitive value while preserving its label.

    Example:
    Account Number: 987654321

    becomes:
    Account Number: [ACCOUNT_NUMBER]
    """

    def replacement(match: Match[str]) -> str:
        findings[entity_type] += 1

        label = match.group(1)

        return f"{label}: {placeholder}"

    return pattern.sub(replacement, text)


def mask_pii_text(
    text: str,
) -> tuple[str, dict[str, int]]:
    """
    Detects and masks supported PII categories.

    Returns:
        masked text
        counts grouped by PII type
    """

    findings: Counter[str] = Counter()
    masked_text = text

    # Labeled identifiers are processed first.
    masked_text = replace_labeled_pattern(
        text=masked_text,
        pattern=ACCOUNT_NUMBER_PATTERN,
        entity_type="ACCOUNT_NUMBER",
        placeholder="[ACCOUNT_NUMBER]",
        findings=findings,
    )

    masked_text = replace_labeled_pattern(
        text=masked_text,
        pattern=CUSTOMER_ID_PATTERN,
        entity_type="CUSTOMER_ID",
        placeholder="[CUSTOMER_ID]",
        findings=findings,
    )

    masked_text = replace_labeled_pattern(
        text=masked_text,
        pattern=DATE_OF_BIRTH_PATTERN,
        entity_type="DATE_OF_BIRTH",
        placeholder="[DATE_OF_BIRTH]",
        findings=findings,
    )

    masked_text = replace_labeled_pattern(
        text=masked_text,
        pattern=PERSON_LABEL_PATTERN,
        entity_type="PERSON",
        placeholder="[PERSON]",
        findings=findings,
    )

    # Unlabeled structured values.
    masked_text = replace_simple_pattern(
        text=masked_text,
        pattern=SSN_PATTERN,
        entity_type="SSN",
        placeholder="[SSN]",
        findings=findings,
    )

    masked_text = replace_simple_pattern(
        text=masked_text,
        pattern=EMAIL_PATTERN,
        entity_type="EMAIL_ADDRESS",
        placeholder="[EMAIL_ADDRESS]",
        findings=findings,
    )

    masked_text = replace_simple_pattern(
        text=masked_text,
        pattern=PHONE_PATTERN,
        entity_type="PHONE_NUMBER",
        placeholder="[PHONE_NUMBER]",
        findings=findings,
    )

    return masked_text, dict(findings)


def mask_pii_in_document(
    document: BankingDocument,
) -> BankingDocument:
    """
    Returns a new BankingDocument containing masked content
    and non-sensitive PII detection metadata.
    """

    masked_content, findings = mask_pii_text(
        document.content
    )

    pii_count = sum(findings.values())
    pii_detected = pii_count > 0

    return document.model_copy(
        update={
            "content": masked_content,
            "pii_detected": pii_detected,
            "pii_masked": pii_detected,
            "pii_types": sorted(findings.keys()),
            "pii_count": pii_count,
        }
    )


def mask_pii_in_documents(
    documents: list[BankingDocument],
) -> list[BankingDocument]:
    """
    Masks PII across a collection of banking documents.
    """

    return [
        mask_pii_in_document(document)
        for document in documents
    ]