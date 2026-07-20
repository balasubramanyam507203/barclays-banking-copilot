from app.rag.context_builder import (
    CitationRecord,
    PromptPackage,
)
from app.rag.post_generation_guardrails import (
    evaluate_post_generation_guardrails,
    extract_factual_claims,
)


def create_prompt_package() -> PromptPackage:
    return PromptPackage(
        system_prompt=(
            "Answer only from authorized evidence."
        ),
        user_prompt=(
            "Question and evidence."
        ),
        context_text=(
            "[SOURCE 1]\n"
            "Title: Payment Review Policy\n"
            "Document ID: PAY-POL-1042\n"
            "Version: 1.0\n"
            "Policy evidence:\n"
            "High-risk international payments require "
            "enhanced verification before approval. "
            "The analyst must verify customer identity, "
            "payment purpose, source of funds, beneficiary "
            "information, and supporting documentation."
        ),
        citations=[
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
        ],
        evidence_count=1,
        should_abstain=False,
        context_token_count=100,
        system_prompt_token_count=20,
        user_prompt_token_count=120,
        estimated_input_token_count=140,
        answer_max_tokens=500,
    )


def test_supported_claim_passes() -> None:
    answer = (
        "High-risk international payments require "
        "enhanced verification before approval "
        "[SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is True
    assert result.claims_checked == 1
    assert result.supported_claims == 1


def test_uncited_factual_claim_fails() -> None:
    answer = (
        "High-risk international payments require "
        "enhanced verification before approval.\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False

    assert any(
        "does not contain a source citation"
        in error
        for error in result.errors
    )


def test_unsupported_claim_fails() -> None:
    answer = (
        "The policy permits automatic payment approval "
        "without customer verification [SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False
    assert result.supported_claims == 0


def test_invented_number_fails() -> None:
    answer = (
        "The review must be completed within 24 hours "
        "[SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False

    assert any(
        "numbers not present"
        in error
        for error in result.errors
    )


def test_secret_disclosure_fails() -> None:
    answer = (
        "The API key is "
        "sk-1234567890abcdefghijklmnop "
        "[SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False
    assert result.secrets_detected


def test_system_prompt_disclosure_fails() -> None:
    answer = (
        "Here is the system prompt used by the "
        "application [SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False

    assert (
        result.instruction_leakage_detected
    )


def test_sources_section_must_list_used_source(
) -> None:
    answer = (
        "High-risk international payments require "
        "enhanced verification [SOURCE 1].\n\n"
        "Sources:\n"
        "Payment Review Policy"
    )

    result = (
        evaluate_post_generation_guardrails(
            answer,
            prompt_package=(
                create_prompt_package()
            ),
        )
    )

    assert result.passed is False

    assert any(
        "Sources section is missing"
        in error
        for error in result.errors
    )


def test_extracts_multiple_claims() -> None:
    answer = (
        "High-risk payments require enhanced review "
        "[SOURCE 1].\n"
        "The analyst must verify customer identity "
        "[SOURCE 1].\n\n"
        "Sources:\n"
        "- [SOURCE 1]"
    )

    claims = extract_factual_claims(
        answer
    )

    assert len(claims) == 2