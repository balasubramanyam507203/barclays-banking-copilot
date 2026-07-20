import json
from pathlib import Path

from app.evaluation.models import (
    EvaluationReport,
)


def write_json_report(
    report: EvaluationReport,
    output_path: Path,
) -> None:
    """
    Writes the complete machine-readable report.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report.model_dump(
                mode="json"
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


def escape_markdown_cell(
    value: str,
) -> str:
    """
    Prevents report text from breaking Markdown tables.
    """

    return (
        value
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def format_percentage(
    value: float,
) -> str:
    return f"{value * 100:.1f}%"


def write_markdown_report(
    report: EvaluationReport,
    output_path: Path,
) -> None:
    """
    Writes a human-readable evaluation summary.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = report.summary

    lines = [
        f"# {report.dataset_name}",
        "",
        f"Dataset version: `{report.dataset_version}`",
        "",
        f"Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary.total_cases}",
        f"- Passed: {summary.passed_cases}",
        f"- Failed: {summary.failed_cases}",
        (
            "- Overall pass rate: "
            f"{format_percentage(summary.pass_rate)}"
        ),
        (
            "- Retrieval recall: "
            f"{format_percentage(summary.average_retrieval_recall)}"
        ),
        (
            "- Mean reciprocal rank: "
            f"{summary.average_reciprocal_rank:.3f}"
        ),
        (
            "- Citation precision: "
            f"{format_percentage(summary.average_citation_precision)}"
        ),
        (
            "- Citation recall: "
            f"{format_percentage(summary.average_citation_recall)}"
        ),
        (
            "- Expected fact coverage: "
            f"{format_percentage(summary.average_fact_coverage)}"
        ),
        (
            "- Abstention accuracy: "
            f"{format_percentage(summary.abstention_accuracy)}"
        ),
        (
            "- Guardrail pass rate: "
            f"{format_percentage(summary.guardrail_pass_rate)}"
        ),
        (
            "- Average retrieval latency: "
            f"{summary.average_retrieval_latency_ms:.1f} ms"
        ),
        (
            "- Average generation latency: "
            f"{summary.average_generation_latency_ms:.1f} ms"
        ),
        (
            "- Average total latency: "
            f"{summary.average_total_latency_ms:.1f} ms"
        ),
        (
            "- Total tokens: "
            f"{summary.total_tokens}"
        ),
        (
            "- Estimated cost: "
            f"${summary.total_estimated_cost_usd:.6f}"
        ),
        "",
        "## Case Results",
        "",
        (
            "| Case | Result | Abstention | Retrieval Recall | "
            "Citation Recall | Fact Coverage | Latency |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for result in report.results:
        lines.append(
            "| "
            + escape_markdown_cell(
                result.case_id
            )
            + " | "
            + (
                "PASS"
                if result.passed
                else "FAIL"
            )
            + " | "
            + (
                "Correct"
                if result
                .metrics
                .abstention_correct
                else "Incorrect"
            )
            + " | "
            + format_percentage(
                result
                .metrics
                .retrieval
                .recall
            )
            + " | "
            + format_percentage(
                result
                .metrics
                .citations
                .recall
            )
            + " | "
            + format_percentage(
                result
                .metrics
                .expected_fact_coverage
            )
            + " | "
            + (
                f"{result.metrics.total_latency_ms:.1f} ms"
            )
            + " |"
        )

    failed_results = [
        result
        for result in report.results
        if not result.passed
    ]

    if failed_results:
        lines.extend(
            [
                "",
                "## Failures",
                "",
            ]
        )

        for result in failed_results:
            lines.append(
                f"### {result.case_id}"
            )

            lines.append("")

            lines.append(
                f"Question: {result.question}"
            )

            lines.append("")

            if result.error:
                lines.append(
                    f"Application error: `{result.error}`"
                )

                lines.append("")

            for reason in result.failure_reasons:
                lines.append(
                    f"- {reason}"
                )

            lines.extend(
                [
                    "",
                    "Retrieved documents: "
                    + (
                        ", ".join(
                            result
                            .retrieved_document_ids
                        )
                        or "None"
                    ),
                    "",
                    "Cited documents: "
                    + (
                        ", ".join(
                            result
                            .cited_document_ids
                        )
                        or "None"
                    ),
                    "",
                ]
            )

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )