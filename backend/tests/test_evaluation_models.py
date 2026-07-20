import pytest
from pydantic import ValidationError

from app.evaluation.models import (
    EvaluationDataset,
)


def test_loads_valid_dataset() -> None:
    dataset = EvaluationDataset.model_validate(
        {
            "dataset_name": "Policy evaluation",
            "version": "1.0.0",
            "cases": [
                {
                    "case_id": "case-1",
                    "question": (
                        "What verification is required?"
                    ),
                    "should_abstain": False,
                    "expected_document_ids": [
                        "KYC-POL-2031"
                    ]
                }
            ]
        }
    )

    assert len(dataset.cases) == 1

    assert (
        dataset
        .thresholds
        .retrieval_top_k
        == 5
    )


def test_answerable_case_requires_document(
) -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": (
                    "Policy evaluation"
                ),
                "version": "1.0.0",
                "cases": [
                    {
                        "case_id": "case-1",
                        "question": (
                            "What verification "
                            "is required?"
                        ),
                        "should_abstain": False,
                        "expected_document_ids": []
                    }
                ]
            }
        )


def test_abstention_case_can_have_no_document(
) -> None:
    dataset = EvaluationDataset.model_validate(
        {
            "dataset_name": "Policy evaluation",
            "version": "1.0.0",
            "cases": [
                {
                    "case_id": "case-1",
                    "question": (
                        "What is my account balance?"
                    ),
                    "should_abstain": True,
                    "expected_document_ids": []
                }
            ]
        }
    )

    assert (
        dataset.cases[0].should_abstain
        is True
    )


def test_duplicate_case_ids_fail() -> None:
    with pytest.raises(
        ValidationError
    ):
        EvaluationDataset.model_validate(
            {
                "dataset_name": (
                    "Policy evaluation"
                ),
                "version": "1.0.0",
                "cases": [
                    {
                        "case_id": "duplicate",
                        "question": "Question one",
                        "should_abstain": True
                    },
                    {
                        "case_id": "duplicate",
                        "question": "Question two",
                        "should_abstain": True
                    }
                ]
            }
        )