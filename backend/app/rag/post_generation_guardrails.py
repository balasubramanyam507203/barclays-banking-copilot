import re
from dataclasses import dataclass

from app.rag.context_builder import PromptPackage


CITATION_PATTERN = re.compile(
    r"\[\s*SOURCE\s+(\d+)\s*\]",
    re.IGNORECASE,
)

CITATION_ONLY_PATTERN = re.compile(
    r"^\s*"
    r"(?:"
    r"\[\s*SOURCE\s+\d+\s*\]"
    r"\s*[,;]?\s*"
    r")+"
    r"$",
    re.IGNORECASE,
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^\s*#{1,6}\s+"
)

SOURCE_BLOCK_PATTERN = re.compile(
    r"(?ms)"
    r"^\[\s*(SOURCE\s+\d+)\s*\]\s*\n"
    r"(.*?)"
    r"(?="
    r"^\[\s*SOURCE\s+\d+\s*\]\s*\n"
    r"|\Z"
    r")"
)

SOURCES_SECTION_PATTERN = re.compile(
    r"(?im)"
    r"^\s*(?:#+\s*)?"
    r"sources\s*:?\s*$"
)

SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+|\n+"
)

TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+"
    r"(?:[.\-_/][a-z0-9]+)*"
)

NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?%?\b"
)


REQUIREMENT_PATTERN = re.compile(
    r"(?i)\b"
    r"(?:"
    r"must"
    r"|shall"
    r"|required"
    r"|requires"
    r"|require"
    r"|requirement"
    r"|mandatory"
    r"|needs?\s+to"
    r"|has\s+to"
    r")"
    r"\b"
)


RELAXATION_PATTERN = re.compile(
    r"(?i)\b"
    r"(?:"
    r"not\s+required"
    r"|is\s+not\s+required"
    r"|isn't\s+required"
    r"|no\s+need"
    r"|need\s+not"
    r"|does\s+not\s+require"
    r"|do\s+not\s+require"
    r"|optional"
    r"|without"
    r"|bypass(?:es|ed|ing)?"
    r"|skip(?:s|ped|ping)?"
    r")"
    r"\b"
)


PERMISSION_PATTERN = re.compile(
    r"(?i)\b"
    r"(?:"
    r"allow(?:s|ed|ing)?"
    r"|permit(?:s|ted|ting)?"
    r"|may"
    r"|can"
    r")"
    r"\b"
)


PROHIBITION_PATTERN = re.compile(
    r"(?i)\b"
    r"(?:"
    r"must\s+not"
    r"|shall\s+not"
    r"|not\s+allowed"
    r"|prohibited"
    r"|forbidden"
    r"|cannot"
    r"|may\s+not"
    r")"
    r"\b"
)


AUTOMATIC_APPROVAL_PATTERN = re.compile(
    r"(?i)\b"
    r"(?:"
    r"automatic(?:ally)?\s+"
    r"(?:payment\s+)?approval"
    r"|approve(?:d|s|ing)?\s+automatically"
    r")"
    r"\b"
)


PRE_APPROVAL_CONTROL_PATTERN = re.compile(
    r"(?is)"
    r"(?:"
    r"\b"
    r"(?:"
    r"verification"
    r"|verify"
    r"|review"
    r"|due\s+diligence"
    r"|supporting\s+documentation"
    r")"
    r"\b"
    r".{0,150}?"
    r"\bbefore\s+approval\b"
    r"|"
    r"\bbefore\s+approval\b"
    r".{0,150}?"
    r"\b"
    r"(?:"
    r"verification"
    r"|verify"
    r"|review"
    r"|due\s+diligence"
    r"|supporting\s+documentation"
    r")"
    r"\b"
    r")"
)


CONTROL_TERMS = {
    "approval",
    "authorization",
    "beneficiary",
    "check",
    "customer",
    "documentation",
    "identity",
    "payment",
    "review",
    "transfer",
    "verification",
}


SECRET_PATTERNS = {
    "OPENAI_API_KEY": re.compile(
        r"(?i)\bOPENAI_API_KEY\b"
    ),
    "API_KEY_ASSIGNMENT": re.compile(
        r"(?i)\bapi[_\s-]?key\s*[:=]\s*\S+"
    ),
    "OPENAI_STYLE_SECRET": re.compile(
        r"\bsk-[A-Za-z0-9_-]{16,}\b"
    ),
    "AWS_ACCESS_KEY": re.compile(
        r"\bAKIA[A-Z0-9]{16}\b"
    ),
    "PRIVATE_KEY": re.compile(
        r"-----BEGIN "
        r"(?:RSA |EC |OPENSSH )?"
        r"PRIVATE KEY-----"
    ),
}


PII_PATTERNS = {
    "SSN": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
    "PAYMENT_CARD": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),
    "LABELED_ACCOUNT_NUMBER": re.compile(
        r"(?i)\b"
        r"(?:account|acct)"
        r"(?:\s+number)?"
        r"\s*[:#=-]?\s*"
        r"\d{6,20}\b"
    ),
}


INSTRUCTION_LEAKAGE_PATTERNS = {
    "SYSTEM_PROMPT_DISCLOSURE": re.compile(
        r"(?i)\b"
        r"(?:the|my|hidden|internal)"
        r"\s+system prompt\b"
    ),
    "DEVELOPER_MESSAGE_DISCLOSURE": re.compile(
        r"(?i)\bdeveloper message\b"
    ),
    "HIDDEN_INSTRUCTIONS_DISCLOSURE": re.compile(
        r"(?i)\bhidden instructions\b"
    ),
    "PROMPT_DUMP": re.compile(
        r"(?i)\b"
        r"(?:begin|here is|below is)"
        r"\s+(?:the\s+)?"
        r"(?:system prompt|internal instructions)"
    ),
}


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "must",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


MINIMUM_CLAIM_TOKEN_COUNT = 3
MINIMUM_SUPPORT_SCORE = 0.20


@dataclass(frozen=True)
class ClaimVerification:
    """
    Verification result for one factual claim.
    """

    claim: str
    citations: tuple[str, ...]

    supported: bool
    support_score: float

    errors: tuple[str, ...]


@dataclass(frozen=True)
class PostGenerationGuardrailResult:
    """
    Complete post-generation guardrail result.
    """

    passed: bool

    claims_checked: int
    supported_claims: int

    claim_results: tuple[
        ClaimVerification,
        ...
    ]

    secrets_detected: tuple[str, ...]
    pii_detected: tuple[str, ...]

    instruction_leakage_detected: tuple[
        str,
        ...
    ]

    errors: tuple[str, ...]


def normalize_citation_label(
    citation_number: str,
) -> str:
    return f"SOURCE {int(citation_number)}"


def extract_citation_labels(
    text: str,
) -> tuple[str, ...]:
    labels = {
        normalize_citation_label(
            citation_number
        )
        for citation_number
        in CITATION_PATTERN.findall(text)
    }

    return tuple(
        sorted(
            labels,
            key=lambda label: int(
                label.split()[-1]
            ),
        )
    )


def normalize_token(
    token: str,
) -> str:
    normalized = token.lower().strip()

    if (
        normalized.endswith("ies")
        and len(normalized) > 4
    ):
        return normalized[:-3] + "y"

    if (
        normalized.endswith("ments")
        and len(normalized) > 6
    ):
        return normalized[:-1]

    if (
        normalized.endswith("ing")
        and len(normalized) > 5
    ):
        return normalized[:-3]

    if (
        normalized.endswith("ed")
        and len(normalized) > 4
    ):
        return normalized[:-2]

    if (
        normalized.endswith("es")
        and len(normalized) > 4
    ):
        return normalized[:-2]

    if (
        normalized.endswith("s")
        and len(normalized) > 3
    ):
        return normalized[:-1]

    return normalized


def tokenize_meaningful_text(
    text: str,
) -> set[str]:
    tokens = TOKEN_PATTERN.findall(
        text.lower()
    )

    return {
        normalize_token(token)
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    }


def remove_sources_section(
    answer: str,
) -> str:
    match = SOURCES_SECTION_PATTERN.search(
        answer
    )

    if match is None:
        return answer.strip()

    return answer[
        :match.start()
    ].strip()


def get_sources_section(
    answer: str,
) -> str:
    match = SOURCES_SECTION_PATTERN.search(
        answer
    )

    if match is None:
        return ""

    return answer[
        match.end():
    ].strip()


def clean_claim_text(
    claim: str,
) -> str:
    cleaned_claim = re.sub(
        r"^\s*(?:[-*•]|\d+[.)])\s*",
        "",
        claim,
    )

    return cleaned_claim.strip()


def looks_like_factual_claim(
    text: str,
) -> bool:
    cleaned_text = clean_claim_text(
        text
    )

    if not cleaned_text:
        return False

    if cleaned_text.endswith(":"):
        return False

    if cleaned_text.lower().startswith(
        "sources"
    ):
        return False

    meaningful_tokens = (
        tokenize_meaningful_text(
            cleaned_text
        )
    )

    return (
        len(meaningful_tokens)
        >= MINIMUM_CLAIM_TOKEN_COUNT
    )


def extract_factual_claims(
    answer: str,
) -> list[str]:
    """
    Splits the answer body into factual claims.

    A citation placed immediately after a sentence,
    such as:

    The review is required. [SOURCE 1]

    is attached to the preceding sentence. This does
    not permit one citation to support multiple earlier
    sentences.
    """

    answer_body = remove_sources_section(
        answer
    )

    raw_segments = (
        SENTENCE_BOUNDARY_PATTERN.split(
            answer_body
        )
    )

    claims: list[str] = []

    for segment in raw_segments:
        stripped_segment = segment.strip()

        if not stripped_segment:
            continue

        if MARKDOWN_HEADING_PATTERN.match(
            stripped_segment
        ):
            continue

        cleaned_segment = clean_claim_text(
            stripped_segment
        )

        if CITATION_ONLY_PATTERN.fullmatch(
            cleaned_segment
        ):
            if claims:
                claims[-1] = (
                    f"{claims[-1]} "
                    f"{cleaned_segment}"
                )

            continue

        if looks_like_factual_claim(
            cleaned_segment
        ):
            claims.append(
                cleaned_segment
            )

    return claims


def extract_evidence_blocks(
    prompt_package: PromptPackage,
) -> dict[str, str]:
    evidence_blocks = {}

    for label, content in (
        SOURCE_BLOCK_PATTERN.findall(
            prompt_package.context_text
        )
    ):
        normalized_label = " ".join(
            label.upper().split()
        )

        evidence_blocks[
            normalized_label
        ] = content.strip()

    if (
        not evidence_blocks
        and len(prompt_package.citations) == 1
        and prompt_package.context_text.strip()
    ):
        only_label = (
            prompt_package.citations[0]
            .label
            .upper()
        )

        evidence_blocks[only_label] = (
            prompt_package.context_text.strip()
        )

    return evidence_blocks


def calculate_support_score(
    *,
    claim: str,
    evidence: str,
) -> float:
    claim_without_citations = (
        CITATION_PATTERN.sub(
            "",
            claim,
        )
    )

    claim_tokens = tokenize_meaningful_text(
        claim_without_citations
    )

    evidence_tokens = tokenize_meaningful_text(
        evidence
    )

    if not claim_tokens:
        return 1.0

    matching_tokens = (
        claim_tokens & evidence_tokens
    )

    return (
        len(matching_tokens)
        / len(claim_tokens)
    )


def detect_claim_evidence_contradictions(
    *,
    claim: str,
    evidence: str,
) -> tuple[str, ...]:
    errors = []

    claim_without_citations = (
        CITATION_PATTERN.sub(
            "",
            claim,
        )
    )

    claim_tokens = tokenize_meaningful_text(
        claim_without_citations
    )

    evidence_tokens = tokenize_meaningful_text(
        evidence
    )

    shared_control_terms = (
        claim_tokens
        & evidence_tokens
        & CONTROL_TERMS
    )

    claim_relaxes_control = bool(
        RELAXATION_PATTERN.search(
            claim_without_citations
        )
    )

    evidence_requires_control = bool(
        REQUIREMENT_PATTERN.search(
            evidence
        )
    )

    if (
        claim_relaxes_control
        and evidence_requires_control
        and shared_control_terms
    ):
        errors.append(
            "The claim says a required control can be "
            "skipped, treated as optional, or performed "
            "without verification, but the cited evidence "
            "states that the control is required."
        )

    claim_requires_control = bool(
        REQUIREMENT_PATTERN.search(
            claim_without_citations
        )
    )

    evidence_relaxes_control = bool(
        RELAXATION_PATTERN.search(
            evidence
        )
    )

    if (
        claim_requires_control
        and evidence_relaxes_control
        and shared_control_terms
    ):
        errors.append(
            "The claim describes a mandatory requirement, "
            "but the cited evidence describes that control "
            "as optional or unnecessary."
        )

    claim_permits_action = bool(
        PERMISSION_PATTERN.search(
            claim_without_citations
        )
    )

    evidence_prohibits_action = bool(
        PROHIBITION_PATTERN.search(
            evidence
        )
    )

    if (
        claim_permits_action
        and evidence_prohibits_action
        and shared_control_terms
    ):
        errors.append(
            "The claim permits an action that the cited "
            "evidence prohibits."
        )

    claim_prohibits_action = bool(
        PROHIBITION_PATTERN.search(
            claim_without_citations
        )
    )

    evidence_permits_action = bool(
        PERMISSION_PATTERN.search(
            evidence
        )
    )

    if (
        claim_prohibits_action
        and evidence_permits_action
        and shared_control_terms
    ):
        errors.append(
            "The claim prohibits an action that the cited "
            "evidence permits."
        )

    claim_allows_automatic_approval = bool(
        AUTOMATIC_APPROVAL_PATTERN.search(
            claim_without_citations
        )
    )

    evidence_requires_preapproval_control = bool(
        PRE_APPROVAL_CONTROL_PATTERN.search(
            evidence
        )
    )

    if (
        claim_allows_automatic_approval
        and evidence_requires_preapproval_control
    ):
        errors.append(
            "The claim allows automatic approval, but the "
            "cited evidence requires verification or "
            "review before approval."
        )

    return tuple(
        dict.fromkeys(errors)
    )


def validate_claim_numbers(
    *,
    claim: str,
    evidence: str,
) -> tuple[str, ...]:
    claim_numbers = set(
        NUMBER_PATTERN.findall(
            CITATION_PATTERN.sub(
                "",
                claim,
            )
        )
    )

    evidence_numbers = set(
        NUMBER_PATTERN.findall(
            evidence
        )
    )

    unsupported_numbers = (
        claim_numbers - evidence_numbers
    )

    return tuple(
        sorted(unsupported_numbers)
    )


def verify_claim(
    claim: str,
    *,
    allowed_citations: set[str],
    evidence_blocks: dict[str, str],
) -> ClaimVerification:
    errors = []

    claim_citations = extract_citation_labels(
        claim
    )

    if not claim_citations:
        errors.append(
            "The factual claim does not contain a "
            "source citation."
        )

        return ClaimVerification(
            claim=claim,
            citations=(),
            supported=False,
            support_score=0.0,
            errors=tuple(errors),
        )

    unknown_citations = [
        citation
        for citation in claim_citations
        if citation not in allowed_citations
    ]

    if unknown_citations:
        errors.append(
            "The claim contains unknown citations: "
            + ", ".join(unknown_citations)
        )

    cited_evidence_parts = [
        evidence_blocks[citation]
        for citation in claim_citations
        if citation in evidence_blocks
    ]

    if not cited_evidence_parts:
        errors.append(
            "No evidence text was found for the "
            "claim's citation labels."
        )

        return ClaimVerification(
            claim=claim,
            citations=claim_citations,
            supported=False,
            support_score=0.0,
            errors=tuple(errors),
        )

    combined_evidence = "\n".join(
        cited_evidence_parts
    )

    support_score = calculate_support_score(
        claim=claim,
        evidence=combined_evidence,
    )

    contradiction_errors = (
        detect_claim_evidence_contradictions(
            claim=claim,
            evidence=combined_evidence,
        )
    )

    errors.extend(
        contradiction_errors
    )

    if support_score < MINIMUM_SUPPORT_SCORE:
        errors.append(
            "The cited evidence does not contain enough "
            "meaningful support for the claim. "
            f"Support score: {support_score:.3f}."
        )

    unsupported_numbers = validate_claim_numbers(
        claim=claim,
        evidence=combined_evidence,
    )

    if unsupported_numbers:
        errors.append(
            "The claim contains numbers not present in "
            "the cited evidence: "
            + ", ".join(unsupported_numbers)
        )

    return ClaimVerification(
        claim=claim,
        citations=claim_citations,
        supported=not errors,
        support_score=support_score,
        errors=tuple(errors),
    )


def detect_pattern_matches(
    text: str,
    *,
    patterns: dict[
        str,
        re.Pattern[str],
    ],
) -> tuple[str, ...]:
    matches = [
        pattern_name
        for pattern_name, pattern
        in patterns.items()
        if pattern.search(text)
    ]

    return tuple(
        sorted(matches)
    )


def verify_sources_section(
    answer: str,
    *,
    citations_used: tuple[str, ...],
) -> tuple[str, ...]:
    errors = []

    sources_section = get_sources_section(
        answer
    )

    if not sources_section:
        errors.append(
            "The answer does not contain a Sources "
            "section."
        )

        return tuple(errors)

    sources_section_citations = set(
        extract_citation_labels(
            sources_section
        )
    )

    missing_citations = [
        citation
        for citation in citations_used
        if citation not in sources_section_citations
    ]

    if missing_citations:
        errors.append(
            "The Sources section is missing citations "
            "used in the answer: "
            + ", ".join(missing_citations)
        )

    unused_citations = [
        citation
        for citation in sources_section_citations
        if citation not in citations_used
    ]

    if unused_citations:
        errors.append(
            "The Sources section contains citations "
            "that were not used in the answer body: "
            + ", ".join(
                sorted(unused_citations)
            )
        )

    return tuple(errors)


def evaluate_post_generation_guardrails(
    answer: str,
    *,
    prompt_package: PromptPackage,
) -> PostGenerationGuardrailResult:
    errors = []

    allowed_citations = {
        citation.label.upper()
        for citation
        in prompt_package.citations
    }

    evidence_blocks = extract_evidence_blocks(
        prompt_package
    )

    factual_claims = extract_factual_claims(
        answer
    )

    claim_results = tuple(
        verify_claim(
            claim,
            allowed_citations=allowed_citations,
            evidence_blocks=evidence_blocks,
        )
        for claim in factual_claims
    )

    unsupported_claims = [
        result
        for result in claim_results
        if not result.supported
    ]

    for result in unsupported_claims:
        errors.append(
            "Unsupported claim: "
            f"'{result.claim}' "
            + " ".join(result.errors)
        )

    citations_used = extract_citation_labels(
        remove_sources_section(answer)
    )

    errors.extend(
        verify_sources_section(
            answer,
            citations_used=citations_used,
        )
    )

    secrets_detected = detect_pattern_matches(
        answer,
        patterns=SECRET_PATTERNS,
    )

    if secrets_detected:
        errors.append(
            "Potential credential or secret disclosure "
            "was detected: "
            + ", ".join(secrets_detected)
        )

    pii_detected = detect_pattern_matches(
        answer,
        patterns=PII_PATTERNS,
    )

    if pii_detected:
        errors.append(
            "Potential unmasked PII was detected in the "
            "generated answer: "
            + ", ".join(pii_detected)
        )

    instruction_leakage_detected = (
        detect_pattern_matches(
            answer,
            patterns=(
                INSTRUCTION_LEAKAGE_PATTERNS
            ),
        )
    )

    if instruction_leakage_detected:
        errors.append(
            "Potential internal-instruction disclosure "
            "was detected: "
            + ", ".join(
                instruction_leakage_detected
            )
        )

    supported_claim_count = sum(
        1
        for result in claim_results
        if result.supported
    )

    return PostGenerationGuardrailResult(
        passed=not errors,
        claims_checked=len(claim_results),
        supported_claims=(
            supported_claim_count
        ),
        claim_results=claim_results,
        secrets_detected=secrets_detected,
        pii_detected=pii_detected,
        instruction_leakage_detected=(
            instruction_leakage_detected
        ),
        errors=tuple(errors),
    )