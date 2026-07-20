from dataclasses import dataclass
from typing import Any

from app.config import (
    GenerationSettings,
)
from app.rag.context_builder import (
    CitationRecord,
    PromptPackage,
)
from app.rag.generation_service import (
    GroundedAnswerGenerationService,
    extract_citation_labels,
    validate_generated_answer,
)


@dataclass
class FakeAIMessage:
    """
    Fake LangChain AI response.
    """

    text: str

    usage_metadata: dict[str, Any]
    response_metadata: dict[str, Any]
    additional_kwargs: dict[str, Any]

    @property
    def content(self) -> str:
        return self.text


class FakeChatModel:
    """
    Fake generation client.
    """

    def __init__(
        self,
        answer: str,
    ) -> None:
        self.answer = answer
        self.call_count = 0
        self.last_input: Any = None

    def invoke(
        self,
        input: Any,
        **kwargs: Any,
    ) -> FakeAIMessage:
        self.call_count += 1
        self.last_input = input

        return FakeAIMessage(
            text=self.answer,
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 50,
                "total_tokens": 250,
            },
            response_metadata={
                "model_name": (
                    "fake-generation-model"
                ),
                "finish_reason": "stop",
            },
            additional_kwargs={
                "refusal": None,
            },
        )


def create_generation_settings(
) -> GenerationSettings:
    return GenerationSettings(
        api_key="test-api-key",
        model="fake-generation-model",
        max_output_tokens=500,
        temperature=0.0,
        timeout_seconds=30.0,
        max_retries=2,
        use_responses_api=True,
        base_url=None,
    )


def create_prompt_package(
    *,
    should_abstain: bool = False,
) -> PromptPackage:
    citations = []
    evidence_count = 0
    context_text = ""

    if not should_abstain:
        citations = [
            CitationRecord(
                label="SOURCE 1",
                citation=(
                    "Payment Review Policy "
                    "(PAY-POL-1042, version 1.0, "
                    "chunk 1/1)"
                ),
                chunk_id="payment-chunk",
                document_id="PAY-POL-1042",
                title="Payment Review Policy",
                version="1.0",
                source=(
                    "local://PAY-POL-1042.txt"
                ),
            )
        ]

        evidence_count = 1

        context_text = (
            "[SOURCE 1]\n"
            "Title: Payment Review Policy\n"
            "Document ID: PAY-POL-1042\n"
            "Version: 1.0\n"
            "Policy evidence:\n"
            "High-risk payments require enhanced "
            "verification before approval."
        )

    return PromptPackage(
        system_prompt=(
            "Answer only from supplied evidence."
        ),
        user_prompt=(
            "Question and authorized evidence."
        ),
        context_text=context_text,
        citations=citations,
        evidence_count=evidence_count,
        should_abstain=should_abstain,
        context_token_count=50,
        system_prompt_token_count=20,
        user_prompt_token_count=80,
        estimated_input_token_count=100,
        answer_max_tokens=500,
    )


def test_extract_citation_labels() -> None:
    labels = extract_citation_labels(
        (
            "Rule one [SOURCE 2]. "
            "Rule two [source 1]. "
            "Repeated [SOURCE 2]."
        )
    )

    assert labels == (
        "SOURCE 1",
        "SOURCE 2",
    )


def test_valid_grounded_answer_passes() -> None:
    answer = (
        "High-risk payments require enhanced "
        "verification before approval "
        "[SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    chat_model = FakeChatModel(
        answer
    )

    service = GroundedAnswerGenerationService(
        chat_model=chat_model,
        settings=create_generation_settings(),
    )

    result = service.generate(
        create_prompt_package()
    )

    assert chat_model.call_count == 1

    assert (
        result.citation_validation_passed
        is True
    )

    assert (
        result.post_generation_guardrails_passed
        is True
    )

    assert result.abstained is False

    assert result.citations_used == (
        "SOURCE 1",
    )

    assert result.claims_checked == 1
    assert result.supported_claims == 1

    assert result.input_tokens == 200
    assert result.output_tokens == 50
    assert result.total_tokens == 250


def test_unknown_citation_is_rejected() -> None:
    answer = (
        "High-risk payments require review "
        "[SOURCE 99].\n\n"
        "Sources:\n"
        "- [SOURCE 99]"
    )

    chat_model = FakeChatModel(
        answer
    )

    service = GroundedAnswerGenerationService(
        chat_model=chat_model,
        settings=create_generation_settings(),
    )

    result = service.generate(
        create_prompt_package()
    )

    assert (
        result.citation_validation_passed
        is False
    )

    assert result.abstained is True

    assert (
        "did not pass source citation validation"
        in result.answer
    )


def test_missing_citation_is_rejected() -> None:
    answer = (
        "High-risk payments require enhanced "
        "verification.\n\n"
        "Sources:\n"
        "Payment policy."
    )

    chat_model = FakeChatModel(
        answer
    )

    service = GroundedAnswerGenerationService(
        chat_model=chat_model,
        settings=create_generation_settings(),
    )

    result = service.generate(
        create_prompt_package()
    )

    assert (
        result.citation_validation_passed
        is False
    )

    assert result.abstained is True


def test_unsupported_claim_fails_guardrails() -> None:
    answer = (
        "The policy allows automatic approval without "
        "customer verification [SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    chat_model = FakeChatModel(
        answer
    )

    service = GroundedAnswerGenerationService(
        chat_model=chat_model,
        settings=create_generation_settings(),
    )

    result = service.generate(
        create_prompt_package()
    )

    assert (
        result.citation_validation_passed
        is True
    )

    assert (
        result.post_generation_guardrails_passed
        is False
    )

    assert result.abstained is True
    assert result.guardrail_errors


def test_no_evidence_skips_model_call() -> None:
    chat_model = FakeChatModel(
        "This answer should never be used."
    )

    service = GroundedAnswerGenerationService(
        chat_model=chat_model,
        settings=create_generation_settings(),
    )

    result = service.generate(
        create_prompt_package(
            should_abstain=True
        )
    )

    assert chat_model.call_count == 0

    assert result.model_called is False
    assert result.abstained is True

    assert (
        result.post_generation_guardrails_passed
        is True
    )

    assert (
        "cannot confirm this"
        in result.answer.lower()
    )


def test_missing_sources_section_is_rejected() -> None:
    answer = (
        "High-risk payments require enhanced "
        "verification [SOURCE 1]."
    )

    validation = validate_generated_answer(
        answer,
        prompt_package=(
            create_prompt_package()
        ),
    )

    assert validation.passed is False

    assert any(
        "Sources section" in error
        for error in validation.errors
    )