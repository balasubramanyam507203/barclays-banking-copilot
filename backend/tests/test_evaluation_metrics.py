import pytest

from app.evaluation.metrics import (
    calculate_estimated_cost,
    calculate_expected_fact_coverage,
    calculate_phrase_match_score,
    calculate_ranked_metrics,
    find_forbidden_phrases,
)


def test_ranked_metrics() -> None:
    metrics = calculate_ranked_metrics(
        ranked_document_ids=[
            "OTHER-POLICY",
            "KYC-POL-2031",
            "SECONDARY-POLICY",
        ],
        expected_document_ids=[
            "KYC-POL-2031"
        ],
    )

    assert metrics.precision == pytest.approx(
        1 / 3
    )

    assert metrics.recall == 1.0

    assert metrics.reciprocal_rank == 0.5


def test_ranked_metrics_missing_document(
) -> None:
    metrics = calculate_ranked_metrics(
        ranked_document_ids=[
            "OTHER-POLICY"
        ],
        expected_document_ids=[
            "KYC-POL-2031"
        ],
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_phrase_match_score() -> None:
    score = calculate_phrase_match_score(
        answer=(
            "The analyst must complete customer "
            "identity verification before approval."
        ),
        expected_phrase=(
            "customer identity verification"
        ),
    )

    assert score == 1.0


def test_expected_fact_coverage() -> None:
    coverage, phrase_scores = (
        calculate_expected_fact_coverage(
            answer=(
                "The analyst verifies customer "
                "identity and source of funds."
            ),
            expected_phrases=[
                "customer identity",
                "source of funds",
            ],
            minimum_phrase_match=0.8,
        )
    )

    assert coverage == 1.0

    assert phrase_scores[
        "customer identity"
    ] == 1.0

    assert phrase_scores[
        "source of funds"
    ] == 1.0


def test_partial_fact_coverage() -> None:
    coverage, _ = (
        calculate_expected_fact_coverage(
            answer=(
                "The analyst verifies customer "
                "identity."
            ),
            expected_phrases=[
                "customer identity",
                "source of funds",
            ],
            minimum_phrase_match=0.8,
        )
    )

    assert coverage == 0.5


def test_forbidden_phrase_detection() -> None:
    matches = find_forbidden_phrases(
        answer=(
            "The policy incorrectly states that "
            "verification is optional."
        ),
        forbidden_phrases=[
            "verification is optional"
        ],
    )

    assert matches == [
        "verification is optional"
    ]


def test_estimated_cost() -> None:
    cost = calculate_estimated_cost(
        input_tokens=1_000_000,
        output_tokens=500_000,
        input_cost_per_million_usd=2.0,
        output_cost_per_million_usd=8.0,
    )

    assert cost == pytest.approx(6.0)