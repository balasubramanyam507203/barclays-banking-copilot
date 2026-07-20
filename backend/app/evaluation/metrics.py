import re
from collections.abc import Iterable

from app.evaluation.models import (
    RankedRetrievalMetrics,
)


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+"
)


def normalize_text(
    text: str,
) -> str:
    """
    Produces normalized text for deterministic phrase
    comparisons.
    """

    tokens = TOKEN_PATTERN.findall(
        text.lower()
    )

    return " ".join(tokens)


def tokenize(
    text: str,
) -> set[str]:
    """
    Returns unique lowercase alphanumeric tokens.
    """

    return set(
        TOKEN_PATTERN.findall(
            text.lower()
        )
    )


def ordered_unique(
    values: Iterable[str],
) -> list[str]:
    """
    Removes duplicate values while preserving rank.
    """

    unique_values = []
    observed_values = set()

    for value in values:
        normalized_value = value.strip()

        if not normalized_value:
            continue

        if normalized_value in observed_values:
            continue

        observed_values.add(
            normalized_value
        )

        unique_values.append(
            normalized_value
        )

    return unique_values


def calculate_ranked_metrics(
    *,
    ranked_document_ids: list[str],
    expected_document_ids: list[str],
) -> RankedRetrievalMetrics:
    """
    Calculates precision, recall and reciprocal rank.

    Precision:
        Relevant retrieved documents divided by all
        retrieved documents.

    Recall:
        Expected documents found divided by all
        expected documents.

    Reciprocal rank:
        1 divided by the rank of the first relevant
        document.
    """

    ranked_ids = ordered_unique(
        ranked_document_ids
    )

    expected_ids = set(
        ordered_unique(
            expected_document_ids
        )
    )

    if not expected_ids:
        return RankedRetrievalMetrics(
            precision=1.0
            if not ranked_ids
            else 0.0,
            recall=1.0,
            reciprocal_rank=1.0
            if not ranked_ids
            else 0.0,
        )

    relevant_retrieved = [
        document_id
        for document_id in ranked_ids
        if document_id in expected_ids
    ]

    precision = (
        len(relevant_retrieved)
        / len(ranked_ids)
        if ranked_ids
        else 0.0
    )

    recall = (
        len(set(relevant_retrieved))
        / len(expected_ids)
    )

    reciprocal_rank = 0.0

    for rank, document_id in enumerate(
        ranked_ids,
        start=1,
    ):
        if document_id in expected_ids:
            reciprocal_rank = 1.0 / rank
            break

    return RankedRetrievalMetrics(
        precision=precision,
        recall=recall,
        reciprocal_rank=reciprocal_rank,
    )


def calculate_phrase_match_score(
    *,
    answer: str,
    expected_phrase: str,
) -> float:
    """
    Measures how many expected-phrase tokens appear in
    the answer.

    This permits small grammatical differences while
    remaining deterministic.
    """

    expected_tokens = tokenize(
        expected_phrase
    )

    if not expected_tokens:
        return 1.0

    answer_tokens = tokenize(answer)

    matching_tokens = (
        expected_tokens
        & answer_tokens
    )

    return (
        len(matching_tokens)
        / len(expected_tokens)
    )


def calculate_expected_fact_coverage(
    *,
    answer: str,
    expected_phrases: list[str],
    minimum_phrase_match: float,
) -> tuple[
    float,
    dict[str, float],
]:
    """
    Measures the percentage of expected facts found in
    the generated answer.
    """

    if not expected_phrases:
        return 1.0, {}

    phrase_scores = {
        phrase: calculate_phrase_match_score(
            answer=answer,
            expected_phrase=phrase,
        )
        for phrase in expected_phrases
    }

    matched_phrase_count = sum(
        1
        for score in phrase_scores.values()
        if score >= minimum_phrase_match
    )

    coverage = (
        matched_phrase_count
        / len(expected_phrases)
    )

    return coverage, phrase_scores


def find_forbidden_phrases(
    *,
    answer: str,
    forbidden_phrases: list[str],
) -> list[str]:
    """
    Returns forbidden phrases found in the answer.
    """

    normalized_answer = normalize_text(
        answer
    )

    detected_phrases = []

    for phrase in forbidden_phrases:
        normalized_phrase = normalize_text(
            phrase
        )

        if (
            normalized_phrase
            and normalized_phrase
            in normalized_answer
        ):
            detected_phrases.append(
                phrase
            )

    return detected_phrases


def calculate_estimated_cost(
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    input_cost_per_million_usd: float,
    output_cost_per_million_usd: float,
) -> float:
    """
    Estimates generation cost using configurable rates.

    Rates remain environment configuration because model
    and internal-gateway pricing can change.
    """

    normalized_input_tokens = (
        input_tokens or 0
    )

    normalized_output_tokens = (
        output_tokens or 0
    )

    input_cost = (
        normalized_input_tokens
        / 1_000_000
        * input_cost_per_million_usd
    )

    output_cost = (
        normalized_output_tokens
        / 1_000_000
        * output_cost_per_million_usd
    )

    return input_cost + output_cost