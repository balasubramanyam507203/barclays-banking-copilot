from app.config import (
    get_embedding_settings,
    get_generation_settings,
    get_hybrid_retrieval_settings,
    get_prompt_settings,
    get_reranker_settings,
    get_vector_store_settings,
)
from app.rag.context_builder import (
    build_prompt_package,
)
from app.rag.embedding_service import (
    build_embedding_client,
)
from app.rag.faiss_store import (
    LocalFaissStore,
    SearchAccessContext,
)
from app.rag.generation_service import (
    GroundedAnswerGenerationService,
    GroundedAnswerResult,
    build_generation_client,
)
from app.rag.keyword_index import (
    LocalBm25Index,
)
from app.rag.reranker import (
    build_reranker,
)
from app.rag.retrieval_service import (
    PermissionAwareRetrievalService,
)


def display_optional_token_count(
    value: int | None,
) -> str:
    """
    Formats an optional token count.
    """

    if value is None:
        return "Not reported"

    return str(value)


def display_grounded_answer(
    result: GroundedAnswerResult,
) -> None:
    """
    Displays only the final trusted answer.

    Invalid raw model responses are not shown.
    """

    print("\n" + "=" * 70)
    print("FINAL GUARDED ANSWER")

    print(
        f"\n{result.answer}"
    )

    print("\n" + "-" * 70)
    print("GENERATION AND GUARDRAIL DIAGNOSTICS")

    print(
        f"\nModel called: "
        f"{result.model_called}"
    )

    print(
        f"Model name: "
        f"{result.model_name}"
    )

    print(
        f"Abstained: "
        f"{result.abstained}"
    )

    print(
        "Citation validation passed: "
        f"{result.citation_validation_passed}"
    )

    print(
        "Post-generation guardrails passed: "
        f"{result.post_generation_guardrails_passed}"
    )

    print(
        f"Citations used: "
        f"{list(result.citations_used)}"
    )

    print(
        f"Claims checked: "
        f"{result.claims_checked}"
    )

    print(
        f"Supported claims: "
        f"{result.supported_claims}"
    )

    print(
        f"Input tokens: "
        f"{display_optional_token_count(result.input_tokens)}"
    )

    print(
        f"Output tokens: "
        f"{display_optional_token_count(result.output_tokens)}"
    )

    print(
        f"Total tokens: "
        f"{display_optional_token_count(result.total_tokens)}"
    )

    print(
        f"Finish reason: "
        f"{result.finish_reason or 'Not reported'}"
    )

    if result.validation_errors:
        print(
            "\nCitation-validation errors:"
        )

        for error in result.validation_errors:
            print(
                f"- {error}"
            )

    if result.guardrail_errors:
        print(
            "\nPost-generation guardrail errors:"
        )

        for error in result.guardrail_errors:
            print(
                f"- {error}"
            )


def main() -> None:
    """
    Runs the complete guarded query-time RAG pipeline.
    """

    print(
        "\nStarting Enterprise Banking "
        "Policy Copilot..."
    )

    embedding_settings = (
        get_embedding_settings()
    )

    vector_store_settings = (
        get_vector_store_settings()
    )

    hybrid_settings = (
        get_hybrid_retrieval_settings()
    )

    reranker_settings = (
        get_reranker_settings()
    )

    prompt_settings = (
        get_prompt_settings()
    )

    generation_settings = (
        get_generation_settings()
    )

    index_path = (
        vector_store_settings.index_path
    )

    if not index_path.exists():
        print(
            "\nThe local FAISS index does not exist."
        )

        print(
            "Build it first with:"
        )

        print(
            "\npython -m app.build_index"
        )

        return

    print(
        f"\nLoading FAISS index from: "
        f"{index_path}"
    )

    vector_store = LocalFaissStore.load(
        index_path
    )

    keyword_index = LocalBm25Index(
        documents=vector_store.documents
    )

    embedding_client = (
        build_embedding_client(
            embedding_settings
        )
    )

    print(
        f"Loading reranker backend: "
        f"{reranker_settings.backend}"
    )

    reranker = build_reranker(
        reranker_settings
    )

    retrieval_service = (
        PermissionAwareRetrievalService(
            vector_store=vector_store,
            keyword_index=keyword_index,
            embedding_client=embedding_client,
            embedding_settings=(
                embedding_settings
            ),
            hybrid_settings=hybrid_settings,
            reranker=reranker,
            reranker_settings=(
                reranker_settings
            ),
            default_top_k=(
                reranker_settings.top_n
            ),
            default_vector_minimum_score=(
                vector_store_settings
                .retrieval_min_score
            ),
        )
    )

    generation_client = (
        build_generation_client(
            generation_settings
        )
    )

    generation_service = (
        GroundedAnswerGenerationService(
            chat_model=generation_client,
            settings=generation_settings,
        )
    )

    # Later these values come from verified JWT claims.
    access_context = SearchAccessContext(
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    print("\n" + "=" * 70)
    print("ENTERPRISE BANKING POLICY COPILOT")

    print(
        f"\nIndexed chunks: "
        f"{vector_store.document_count}"
    )

    print(
        f"Embedding model: "
        f"{embedding_settings.model}"
    )

    print(
        f"Reranker backend: "
        f"{reranker_settings.backend}"
    )

    print(
        f"Generation model: "
        f"{generation_settings.model}"
    )

    print(
        "Post-generation guardrails: Enabled"
    )

    print(
        f"User role: "
        f"{access_context.role}"
    )

    print(
        f"User region: "
        f"{access_context.region}"
    )

    print(
        f"User clearance rank: "
        f"{access_context.clearance_rank}"
    )

    print(
        "\nType a policy question or policy ID."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:
        try:
            query = input(
                "\nQuestion: "
            )

        except EOFError:
            print(
                "\nExiting Banking Policy Copilot."
            )
            break

        if query.strip().lower() in {
            "exit",
            "quit",
        }:
            print(
                "\nExiting Banking Policy Copilot."
            )
            break

        try:
            retrieval_response = (
                retrieval_service.retrieve(
                    query,
                    access_context=(
                        access_context
                    ),
                )
            )

            prompt_package = (
                build_prompt_package(
                    retrieval_response,
                    settings=prompt_settings,
                )
            )

            answer_result = (
                generation_service.generate(
                    prompt_package
                )
            )

            display_grounded_answer(
                answer_result
            )

        except ValueError as error:
            print(
                f"\nRequest failed: {error}"
            )

        except Exception as error:
            print(
                "\nUnexpected application error: "
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()