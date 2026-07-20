from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Any

from app.api.services import (
    ApplicationServices,
)
from app.evaluation.metrics import (
    calculate_estimated_cost,
    calculate_expected_fact_coverage,
    calculate_ranked_metrics,
    find_forbidden_phrases,
    ordered_unique,
)
from app.evaluation.models import (
    EvaluationCase,
    EvaluationCaseMetrics,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationReport,
    EvaluationSummary,
    EvaluationThresholds,
    TokenUsageMetrics,
)
from app.rag.context_builder import (
    PromptPackage,
    build_prompt_package,
)
from app.rag.faiss_store import (
    SearchAccessContext,
)
from app.rag.generation_service import (
    GroundedAnswerResult,
)


class EvaluationRunner:
    """
    Runs the existing production-style RAG services
    against a versioned evaluation dataset.
    """

    def __init__(
        self,
        *,
        services: ApplicationServices,
        input_cost_per_million_usd: float = 0.0,
        output_cost_per_million_usd: float = 0.0,
    ) -> None:
        if input_cost_per_million_usd < 0:
            raise ValueError(
                "Input token cost cannot be negative."
            )

        if output_cost_per_million_usd < 0:
            raise ValueError(
                "Output token cost cannot be negative."
            )

        self.services = services

        self.input_cost_per_million_usd = (
            input_cost_per_million_usd
        )

        self.output_cost_per_million_usd = (
            output_cost_per_million_usd
        )

    def run_dataset(
        self,
        dataset: EvaluationDataset,
    ) -> EvaluationReport:
        """
        Runs all evaluation cases.
        """

        results = [
            self.run_case(
                case,
                thresholds=dataset.thresholds,
            )
            for case in dataset.cases
        ]

        summary = self.build_summary(
            results
        )

        return EvaluationReport(
            dataset_name=dataset.dataset_name,
            dataset_version=dataset.version,
            generated_at=datetime.now(
                timezone.utc
            ),
            thresholds=dataset.thresholds,
            summary=summary,
            results=results,
            metadata={
                "embedding_model": (
                    self.services.embedding_model
                ),
                "generation_model": (
                    self.services.generation_model
                ),
                "reranker_backend": (
                    self.services.reranker_backend
                ),
            },
        )

    def run_case(
        self,
        case: EvaluationCase,
        *,
        thresholds: EvaluationThresholds,
    ) -> EvaluationCaseResult:
        """
        Runs retrieval, generation and metrics for one
        question.
        """

        case_started_at = perf_counter()

        try:
            access_context = SearchAccessContext(
                role=case.access.role,
                region=case.access.region,
                clearance_rank=(
                    case.access.clearance_rank
                ),
            )

            retrieval_started_at = (
                perf_counter()
            )

            retrieval_response = (
                self.services
                .retrieval_service
                .retrieve(
                    case.question,
                    access_context=(
                        access_context
                    ),
                    top_k=(
                        thresholds.retrieval_top_k
                    ),
                )
            )

            retrieval_latency_ms = (
                perf_counter()
                - retrieval_started_at
            ) * 1_000

            retrieved_document_ids = (
                self.collect_retrieved_document_ids(
                    retrieval_response
                )
            )

            prompt_package = (
                build_prompt_package(
                    retrieval_response,
                    settings=(
                        self.services
                        .prompt_settings
                    ),
                )
            )

            generation_started_at = (
                perf_counter()
            )

            answer_result = (
                self.services
                .generation_service
                .generate(
                    prompt_package
                )
            )

            generation_latency_ms = (
                perf_counter()
                - generation_started_at
            ) * 1_000

            total_latency_ms = (
                perf_counter()
                - case_started_at
            ) * 1_000

            return self.build_case_result(
                case=case,
                thresholds=thresholds,
                prompt_package=prompt_package,
                answer_result=answer_result,
                retrieved_document_ids=(
                    retrieved_document_ids
                ),
                retrieval_latency_ms=(
                    retrieval_latency_ms
                ),
                generation_latency_ms=(
                    generation_latency_ms
                ),
                total_latency_ms=(
                    total_latency_ms
                ),
            )

        except Exception as error:
            total_latency_ms = (
                perf_counter()
                - case_started_at
            ) * 1_000

            return EvaluationCaseResult(
                case_id=case.case_id,
                question=case.question,
                should_abstain=(
                    case.should_abstain
                ),
                tags=case.tags,
                expected_document_ids=(
                    case.expected_document_ids
                ),
                passed=False,
                failure_reasons=[
                    "The evaluation case raised an "
                    "application error."
                ],
                metrics=EvaluationCaseMetrics(
                    total_latency_ms=(
                        total_latency_ms
                    )
                ),
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

    def build_case_result(
        self,
        *,
        case: EvaluationCase,
        thresholds: EvaluationThresholds,
        prompt_package: PromptPackage,
        answer_result: GroundedAnswerResult,
        retrieved_document_ids: list[str],
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
    ) -> EvaluationCaseResult:
        """
        Calculates metrics and determines pass/fail.

        Only citations from a guardrail-approved,
        non-abstention answer are counted as released
        citations.

        Citations from a rejected internal draft must
        not be treated as citations shown to the user.
        """

        approved_citation_labels = (
            answer_result.citations_used
            if (
                not answer_result.abstained
                and answer_result
                .citation_validation_passed
                and answer_result
                .post_generation_guardrails_passed
            )
            else ()
        )

        cited_document_ids = (
            self.collect_cited_document_ids(
                prompt_package=prompt_package,
                citations_used=(
                    approved_citation_labels
                ),
            )
        )

        retrieval_metrics = (
            calculate_ranked_metrics(
                ranked_document_ids=(
                    retrieved_document_ids
                ),
                expected_document_ids=(
                    case.expected_document_ids
                ),
            )
        )

        citation_metrics = (
            calculate_ranked_metrics(
                ranked_document_ids=(
                    cited_document_ids
                ),
                expected_document_ids=(
                    case.expected_document_ids
                ),
            )
        )

        (
            expected_fact_coverage,
            phrase_match_scores,
        ) = calculate_expected_fact_coverage(
            answer=answer_result.answer,
            expected_phrases=(
                case.expected_answer_phrases
            ),
            minimum_phrase_match=(
                thresholds
                .expected_phrase_token_match
            ),
        )

        detected_forbidden_phrases = (
            find_forbidden_phrases(
                answer=answer_result.answer,
                forbidden_phrases=(
                    case
                    .forbidden_answer_phrases
                ),
            )
        )

        abstention_correct = (
            answer_result.abstained
            == case.should_abstain
        )

        guardrails_passed = (
            answer_result
            .citation_validation_passed
            and answer_result
            .post_generation_guardrails_passed
        )

        estimated_cost_usd = (
            calculate_estimated_cost(
                input_tokens=(
                    answer_result.input_tokens
                ),
                output_tokens=(
                    answer_result.output_tokens
                ),
                input_cost_per_million_usd=(
                    self
                    .input_cost_per_million_usd
                ),
                output_cost_per_million_usd=(
                    self
                    .output_cost_per_million_usd
                ),
            )
        )

        metrics = EvaluationCaseMetrics(
            retrieval=retrieval_metrics,
            citations=citation_metrics,
            expected_fact_coverage=(
                expected_fact_coverage
            ),
            phrase_match_scores=(
                phrase_match_scores
            ),
            forbidden_phrase_passed=(
                not detected_forbidden_phrases
            ),
            abstention_correct=(
                abstention_correct
            ),
            citation_validation_passed=(
                answer_result
                .citation_validation_passed
            ),
            post_generation_guardrails_passed=(
                answer_result
                .post_generation_guardrails_passed
            ),
            retrieval_latency_ms=(
                retrieval_latency_ms
            ),
            generation_latency_ms=(
                generation_latency_ms
            ),
            total_latency_ms=(
                total_latency_ms
            ),
            usage=TokenUsageMetrics(
                input_tokens=(
                    answer_result.input_tokens
                ),
                output_tokens=(
                    answer_result.output_tokens
                ),
                total_tokens=(
                    answer_result.total_tokens
                ),
            ),
            estimated_cost_usd=(
                estimated_cost_usd
            ),
        )

        failure_reasons = (
            self.determine_failure_reasons(
                case=case,
                thresholds=thresholds,
                metrics=metrics,
                answer_result=answer_result,
                cited_document_ids=(
                    cited_document_ids
                ),
                detected_forbidden_phrases=(
                    detected_forbidden_phrases
                ),
                guardrails_passed=(
                    guardrails_passed
                ),
            )
        )

        return EvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            should_abstain=(
                case.should_abstain
            ),
            tags=case.tags,
            expected_document_ids=(
                case.expected_document_ids
            ),
            retrieved_document_ids=(
                retrieved_document_ids
            ),
            cited_document_ids=(
                cited_document_ids
            ),
            answer=answer_result.answer,
            abstained=answer_result.abstained,
            model_called=(
                answer_result.model_called
            ),
            passed=not failure_reasons,
            failure_reasons=failure_reasons,
            metrics=metrics,
        )

    @staticmethod
    def determine_failure_reasons(
        *,
        case: EvaluationCase,
        thresholds: EvaluationThresholds,
        metrics: EvaluationCaseMetrics,
        answer_result: GroundedAnswerResult,
        cited_document_ids: list[str],
        detected_forbidden_phrases: list[str],
        guardrails_passed: bool,
    ) -> list[str]:
        """
        Applies deterministic pass/fail rules.
        """

        failure_reasons = []

        if not metrics.abstention_correct:
            if case.should_abstain:
                failure_reasons.append(
                    "The system answered a question that "
                    "was expected to abstain."
                )
            else:
                failure_reasons.append(
                    "The system abstained from an "
                    "answerable question."
                )

        if case.should_abstain:
            if cited_document_ids:
                failure_reasons.append(
                    "The abstention response contained "
                    "policy citations."
                )

        else:
            if (
                metrics.retrieval.recall
                < thresholds
                .minimum_retrieval_recall
            ):
                failure_reasons.append(
                    "Retrieval recall was below the "
                    "required threshold."
                )

            if (
                metrics.citations.precision
                < thresholds
                .minimum_citation_precision
            ):
                failure_reasons.append(
                    "Citation precision was below the "
                    "required threshold."
                )

            if (
                metrics.citations.recall
                < thresholds
                .minimum_citation_recall
            ):
                failure_reasons.append(
                    "Citation recall was below the "
                    "required threshold."
                )

            if (
                metrics.expected_fact_coverage
                < thresholds
                .minimum_fact_coverage
            ):
                failure_reasons.append(
                    "Expected answer fact coverage was "
                    "below the required threshold."
                )

        if detected_forbidden_phrases:
            failure_reasons.append(
                "The answer contained forbidden phrases: "
                + ", ".join(
                    detected_forbidden_phrases
                )
            )

        if (
            thresholds.require_guardrails
            and not guardrails_passed
        ):
            failure_reasons.append(
                "The answer did not pass all generation "
                "guardrails."
            )

        if (
            not case.should_abstain
            and not answer_result.answer.strip()
        ):
            failure_reasons.append(
                "The answer was empty."
            )

        return failure_reasons

    @staticmethod
    def collect_retrieved_document_ids(
        retrieval_response: Any,
    ) -> list[str]:
        """
        Returns document IDs in retrieval and
        reranking order.
        """

        document_ids = []

        for result in retrieval_response.results:
            document_id = str(
                result.document.metadata.get(
                    "document_id",
                    "",
                )
            ).strip()

            if document_id:
                document_ids.append(
                    document_id
                )

        return ordered_unique(
            document_ids
        )

    @staticmethod
    def collect_cited_document_ids(
        *,
        prompt_package: PromptPackage,
        citations_used: tuple[str, ...],
    ) -> list[str]:
        """
        Maps generated source labels back to document
        IDs.
        """

        citation_document_map = {
            citation.label.upper():
                citation.document_id
            for citation
            in prompt_package.citations
        }

        cited_document_ids = [
            citation_document_map[
                citation_label.upper()
            ]
            for citation_label
            in citations_used
            if citation_label.upper()
            in citation_document_map
        ]

        return ordered_unique(
            cited_document_ids
        )

    @staticmethod
    def build_summary(
        results: list[EvaluationCaseResult],
    ) -> EvaluationSummary:
        """
        Aggregates all case-level metrics.
        """

        total_cases = len(results)

        passed_cases = sum(
            1
            for result in results
            if result.passed
        )

        failed_cases = (
            total_cases - passed_cases
        )

        answerable_results = [
            result
            for result in results
            if not result.should_abstain
        ]

        def average(
            values: list[float],
        ) -> float:
            return (
                mean(values)
                if values
                else 0.0
            )

        total_input_tokens = sum(
            result.metrics.usage.input_tokens
            or 0
            for result in results
        )

        total_output_tokens = sum(
            result.metrics.usage.output_tokens
            or 0
            for result in results
        )

        total_tokens = sum(
            result.metrics.usage.total_tokens
            or 0
            for result in results
        )

        return EvaluationSummary(
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            pass_rate=(
                passed_cases / total_cases
                if total_cases
                else 0.0
            ),
            average_retrieval_precision=average(
                [
                    result
                    .metrics
                    .retrieval
                    .precision
                    for result
                    in answerable_results
                ]
            ),
            average_retrieval_recall=average(
                [
                    result
                    .metrics
                    .retrieval
                    .recall
                    for result
                    in answerable_results
                ]
            ),
            average_reciprocal_rank=average(
                [
                    result
                    .metrics
                    .retrieval
                    .reciprocal_rank
                    for result
                    in answerable_results
                ]
            ),
            average_citation_precision=average(
                [
                    result
                    .metrics
                    .citations
                    .precision
                    for result
                    in answerable_results
                ]
            ),
            average_citation_recall=average(
                [
                    result
                    .metrics
                    .citations
                    .recall
                    for result
                    in answerable_results
                ]
            ),
            average_fact_coverage=average(
                [
                    result
                    .metrics
                    .expected_fact_coverage
                    for result
                    in answerable_results
                ]
            ),
            abstention_accuracy=average(
                [
                    (
                        1.0
                        if result
                        .metrics
                        .abstention_correct
                        else 0.0
                    )
                    for result in results
                ]
            ),
            guardrail_pass_rate=average(
                [
                    (
                        1.0
                        if (
                            result
                            .metrics
                            .citation_validation_passed
                            and result
                            .metrics
                            .post_generation_guardrails_passed
                        )
                        else 0.0
                    )
                    for result in results
                ]
            ),
            average_retrieval_latency_ms=average(
                [
                    result
                    .metrics
                    .retrieval_latency_ms
                    for result in results
                ]
            ),
            average_generation_latency_ms=average(
                [
                    result
                    .metrics
                    .generation_latency_ms
                    for result in results
                ]
            ),
            average_total_latency_ms=average(
                [
                    result
                    .metrics
                    .total_latency_ms
                    for result in results
                ]
            ),
            total_input_tokens=(
                total_input_tokens
            ),
            total_output_tokens=(
                total_output_tokens
            ),
            total_tokens=total_tokens,
            total_estimated_cost_usd=sum(
                result
                .metrics
                .estimated_cost_usd
                for result in results
            ),
        )