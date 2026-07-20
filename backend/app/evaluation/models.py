from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class EvaluationAccessContext(BaseModel):
    """
    Permissions used while evaluating one question.

    This allows the evaluation suite to verify that
    different roles receive different evidence.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    role: str = Field(
        default="compliance_analyst",
        min_length=1,
        max_length=100,
    )

    region: str = Field(
        default="US",
        min_length=1,
        max_length=50,
    )

    clearance_rank: int = Field(
        default=2,
        ge=1,
        le=10,
    )


class EvaluationCase(BaseModel):
    """
    One expected RAG behavior.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    case_id: str = Field(
        min_length=1,
        max_length=150,
    )

    question: str = Field(
        min_length=1,
        max_length=2_000,
    )

    should_abstain: bool = False

    expected_document_ids: list[str] = Field(
        default_factory=list
    )

    expected_answer_phrases: list[str] = Field(
        default_factory=list
    )

    forbidden_answer_phrases: list[str] = Field(
        default_factory=list
    )

    access: EvaluationAccessContext = Field(
        default_factory=EvaluationAccessContext
    )

    tags: list[str] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_expected_behavior(
        self,
    ) -> "EvaluationCase":
        """
        Answerable cases must identify at least one
        expected source document.
        """

        self.expected_document_ids = [
            document_id.strip()
            for document_id
            in self.expected_document_ids
            if document_id.strip()
        ]

        self.expected_answer_phrases = [
            phrase.strip()
            for phrase
            in self.expected_answer_phrases
            if phrase.strip()
        ]

        self.forbidden_answer_phrases = [
            phrase.strip()
            for phrase
            in self.forbidden_answer_phrases
            if phrase.strip()
        ]

        self.tags = [
            tag.strip()
            for tag in self.tags
            if tag.strip()
        ]

        if (
            not self.should_abstain
            and not self.expected_document_ids
        ):
            raise ValueError(
                "A non-abstention evaluation case must "
                "contain at least one expected document "
                "ID."
            )

        return self


class EvaluationThresholds(BaseModel):
    """
    Pass/fail thresholds for the evaluation suite.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    minimum_retrieval_recall: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    minimum_citation_precision: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    minimum_citation_recall: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    minimum_fact_coverage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    expected_phrase_token_match: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )

    require_guardrails: bool = True


class EvaluationDataset(BaseModel):
    """
    Complete versioned evaluation dataset.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    dataset_name: str = Field(
        min_length=1,
        max_length=200,
    )

    version: str = Field(
        min_length=1,
        max_length=50,
    )

    description: str = ""

    thresholds: EvaluationThresholds = Field(
        default_factory=EvaluationThresholds
    )

    cases: list[EvaluationCase] = Field(
        min_length=1
    )

    @model_validator(mode="after")
    def validate_case_ids(
        self,
    ) -> "EvaluationDataset":
        case_ids = [
            case.case_id
            for case in self.cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Evaluation case IDs must be unique."
            )

        return self


class RankedRetrievalMetrics(BaseModel):
    """
    Metrics for an ordered collection of document IDs.
    """

    precision: float = 0.0
    recall: float = 0.0
    reciprocal_rank: float = 0.0


class TokenUsageMetrics(BaseModel):
    """
    Token usage returned by the generation provider.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class EvaluationCaseMetrics(BaseModel):
    """
    Calculated metrics for one case.
    """

    retrieval: RankedRetrievalMetrics = Field(
        default_factory=RankedRetrievalMetrics
    )

    citations: RankedRetrievalMetrics = Field(
        default_factory=RankedRetrievalMetrics
    )

    expected_fact_coverage: float = 0.0

    phrase_match_scores: dict[str, float] = Field(
        default_factory=dict
    )

    forbidden_phrase_passed: bool = True

    abstention_correct: bool = False

    citation_validation_passed: bool = False
    post_generation_guardrails_passed: bool = False

    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    usage: TokenUsageMetrics = Field(
        default_factory=TokenUsageMetrics
    )

    estimated_cost_usd: float = 0.0


class EvaluationCaseResult(BaseModel):
    """
    Detailed outcome for one evaluation question.
    """

    case_id: str
    question: str
    should_abstain: bool

    tags: list[str] = Field(
        default_factory=list
    )

    expected_document_ids: list[str] = Field(
        default_factory=list
    )

    retrieved_document_ids: list[str] = Field(
        default_factory=list
    )

    cited_document_ids: list[str] = Field(
        default_factory=list
    )

    answer: str = ""
    abstained: bool = False
    model_called: bool = False

    passed: bool = False

    failure_reasons: list[str] = Field(
        default_factory=list
    )

    metrics: EvaluationCaseMetrics = Field(
        default_factory=EvaluationCaseMetrics
    )

    error: str | None = None


class EvaluationSummary(BaseModel):
    """
    Aggregated evaluation metrics.
    """

    total_cases: int
    passed_cases: int
    failed_cases: int

    pass_rate: float

    average_retrieval_precision: float
    average_retrieval_recall: float
    average_reciprocal_rank: float

    average_citation_precision: float
    average_citation_recall: float

    average_fact_coverage: float

    abstention_accuracy: float
    guardrail_pass_rate: float

    average_retrieval_latency_ms: float
    average_generation_latency_ms: float
    average_total_latency_ms: float

    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    total_estimated_cost_usd: float


class EvaluationReport(BaseModel):
    """
    Complete serializable evaluation report.
    """

    dataset_name: str
    dataset_version: str

    generated_at: datetime

    thresholds: EvaluationThresholds

    summary: EvaluationSummary

    results: list[EvaluationCaseResult]

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )