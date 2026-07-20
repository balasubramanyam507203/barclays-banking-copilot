import argparse
import os
from pathlib import Path

from app.api.services import (
    build_application_services,
)
from app.evaluation.models import (
    EvaluationDataset,
)
from app.evaluation.reporting import (
    write_json_report,
    write_markdown_report,
)
from app.evaluation.runner import (
    EvaluationRunner,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "policy_eval_v1.json"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "reports"
)


def read_non_negative_float(
    environment_variable: str,
    default: float = 0.0,
) -> float:
    """
    Reads a non-negative evaluation cost setting.
    """

    raw_value = os.getenv(
        environment_variable,
        str(default),
    ).strip()

    try:
        value = float(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"{environment_variable} must be a "
            "number."
        ) from error

    if value < 0:
        raise RuntimeError(
            f"{environment_variable} cannot be "
            "negative."
        )

    return value


def load_dataset(
    dataset_path: Path,
) -> EvaluationDataset:
    """
    Loads and validates the evaluation dataset.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Evaluation dataset does not exist at "
            f"'{dataset_path}'."
        )

    return EvaluationDataset.model_validate_json(
        dataset_path.read_text(
            encoding="utf-8"
        )
    )


def build_argument_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Enterprise Banking Policy "
            "Copilot evaluation suite."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to a versioned evaluation "
            "dataset JSON file."
        ),
    )

    parser.add_argument(
        "--report-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
        help=(
            "Directory where evaluation reports "
            "will be written."
        ),
    )

    return parser


def display_summary(
    report,
) -> None:
    summary = report.summary

    print("\n" + "=" * 70)
    print("RAG EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"\nDataset: {report.dataset_name} "
        f"v{report.dataset_version}"
    )

    print(
        f"Cases: {summary.total_cases}"
    )

    print(
        f"Passed: {summary.passed_cases}"
    )

    print(
        f"Failed: {summary.failed_cases}"
    )

    print(
        f"Pass rate: "
        f"{summary.pass_rate * 100:.1f}%"
    )

    print(
        "Retrieval recall: "
        f"{summary.average_retrieval_recall * 100:.1f}%"
    )

    print(
        "Citation precision: "
        f"{summary.average_citation_precision * 100:.1f}%"
    )

    print(
        "Citation recall: "
        f"{summary.average_citation_recall * 100:.1f}%"
    )

    print(
        "Expected fact coverage: "
        f"{summary.average_fact_coverage * 100:.1f}%"
    )

    print(
        "Abstention accuracy: "
        f"{summary.abstention_accuracy * 100:.1f}%"
    )

    print(
        "Guardrail pass rate: "
        f"{summary.guardrail_pass_rate * 100:.1f}%"
    )

    print(
        "Average total latency: "
        f"{summary.average_total_latency_ms:.1f} ms"
    )

    print(
        f"Total tokens: {summary.total_tokens}"
    )

    print(
        "Estimated cost: "
        f"${summary.total_estimated_cost_usd:.6f}"
    )


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    dataset_path = (
        arguments.dataset.resolve()
    )

    report_directory = (
        arguments
        .report_directory
        .resolve()
    )

    print(
        "\nStarting Enterprise Banking "
        "Policy Copilot evaluation..."
    )

    print(
        f"Dataset: {dataset_path}"
    )

    dataset = load_dataset(
        dataset_path
    )

    services = (
        build_application_services()
    )

    try:
        runner = EvaluationRunner(
            services=services,
            input_cost_per_million_usd=(
                read_non_negative_float(
                    "EVAL_INPUT_COST_PER_1M_USD"
                )
            ),
            output_cost_per_million_usd=(
                read_non_negative_float(
                    "EVAL_OUTPUT_COST_PER_1M_USD"
                )
            ),
        )

        report = runner.run_dataset(
            dataset
        )

        json_report_path = (
            report_directory
            / "latest_evaluation.json"
        )

        markdown_report_path = (
            report_directory
            / "latest_evaluation.md"
        )

        write_json_report(
            report,
            json_report_path,
        )

        write_markdown_report(
            report,
            markdown_report_path,
        )

        display_summary(report)

        print(
            f"\nJSON report: "
            f"{json_report_path}"
        )

        print(
            f"Markdown report: "
            f"{markdown_report_path}"
        )

        if report.summary.failed_cases:
            raise SystemExit(1)

    finally:
        services.database_service.dispose()


if __name__ == "__main__":
    main()