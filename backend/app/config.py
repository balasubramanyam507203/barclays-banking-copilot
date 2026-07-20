import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class EmbeddingSettings:
    """
    Configuration for document and query embeddings.
    """

    api_key: str = field(repr=False)
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    base_url: str | None = None


@dataclass(frozen=True)
class VectorStoreSettings:
    """
    Configuration for the local FAISS index.
    """

    index_path: Path = Path(
        "local_indexes/policy_faiss"
    )

    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.0


@dataclass(frozen=True)
class HybridRetrievalSettings:
    """
    Configuration for semantic and keyword retrieval.
    """

    vector_candidate_k: int = 20
    keyword_candidate_k: int = 20

    keyword_minimum_score: float = 0.0

    reciprocal_rank_constant: int = 60

    vector_weight: float = 1.0
    keyword_weight: float = 1.0


@dataclass(frozen=True)
class RerankerSettings:
    """
    Configuration for the second-stage reranker.
    """

    backend: str = "local_feature"

    candidate_k: int = 10
    top_n: int = 5

    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

    device: str = "cpu"


@dataclass(frozen=True)
class PromptSettings:
    """
    Configuration for context assembly and prompt
    construction.
    """

    max_context_tokens: int = 3_000
    max_context_chunks: int = 5

    answer_max_tokens: int = 700

    token_encoding: str = "cl100k_base"

    minimum_evidence_chunks: int = 1


@dataclass(frozen=True)
class GenerationSettings:
    """
    Configuration for grounded answer generation.

    The same OpenAI key is used locally for embeddings
    and generation.

    Production can route both through an approved model
    gateway using base_url.
    """

    api_key: str = field(repr=False)

    model: str = "gpt-4.1-mini"
    max_output_tokens: int = 700

    temperature: float | None = 0.0

    timeout_seconds: float = 60.0
    max_retries: int = 2

    use_responses_api: bool = True

    base_url: str | None = None


def get_required_environment_variable(
    variable_name: str,
) -> str:
    """
    Returns a required environment variable.
    """

    value = os.getenv(variable_name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Required environment variable "
            f"'{variable_name}' is missing. "
            "Add it to the backend/.env file."
        )

    return value.strip()


def get_optional_environment_variable(
    variable_name: str,
) -> str | None:
    """
    Returns an optional environment variable.
    """

    value = os.getenv(variable_name)

    if value is None:
        return None

    cleaned_value = value.strip()

    return cleaned_value or None


def get_positive_integer_environment_variable(
    variable_name: str,
    default_value: int,
) -> int:
    """
    Reads and validates a positive integer.
    """

    raw_value = os.getenv(
        variable_name,
        str(default_value),
    ).strip()

    try:
        parsed_value = int(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            f"must be an integer, but received "
            f"'{raw_value}'."
        ) from error

    if parsed_value <= 0:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            "must be greater than zero."
        )

    return parsed_value


def get_non_negative_integer_environment_variable(
    variable_name: str,
    default_value: int,
) -> int:
    """
    Reads and validates a non-negative integer.
    """

    raw_value = os.getenv(
        variable_name,
        str(default_value),
    ).strip()

    try:
        parsed_value = int(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            f"must be an integer, but received "
            f"'{raw_value}'."
        ) from error

    if parsed_value < 0:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            "cannot be negative."
        )

    return parsed_value


def get_float_environment_variable(
    variable_name: str,
    default_value: float,
) -> float:
    """
    Reads and validates a floating-point value.
    """

    raw_value = os.getenv(
        variable_name,
        str(default_value),
    ).strip()

    try:
        return float(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            f"must be a number, but received "
            f"'{raw_value}'."
        ) from error


def get_positive_float_environment_variable(
    variable_name: str,
    default_value: float,
) -> float:
    """
    Reads and validates a positive float.
    """

    parsed_value = get_float_environment_variable(
        variable_name,
        default_value,
    )

    if parsed_value <= 0:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            "must be greater than zero."
        )

    return parsed_value


def get_optional_float_environment_variable(
    variable_name: str,
    default_value: float | None,
) -> float | None:
    """
    Reads an optional float.

    Values such as none, null, and an empty string
    return None.
    """

    raw_value = os.getenv(variable_name)

    if raw_value is None:
        return default_value

    cleaned_value = raw_value.strip().lower()

    if cleaned_value in {
        "",
        "none",
        "null",
    }:
        return None

    try:
        return float(cleaned_value)

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{variable_name}' "
            f"must be a number or 'none', but received "
            f"'{raw_value}'."
        ) from error


def get_boolean_environment_variable(
    variable_name: str,
    default_value: bool,
) -> bool:
    """
    Reads and validates a Boolean environment value.
    """

    raw_value = os.getenv(variable_name)

    if raw_value is None:
        return default_value

    cleaned_value = raw_value.strip().lower()

    true_values = {
        "true",
        "1",
        "yes",
        "y",
        "on",
    }

    false_values = {
        "false",
        "0",
        "no",
        "n",
        "off",
    }

    if cleaned_value in true_values:
        return True

    if cleaned_value in false_values:
        return False

    raise RuntimeError(
        f"Environment variable '{variable_name}' "
        "must contain true or false."
    )


def get_embedding_settings() -> EmbeddingSettings:
    """
    Builds embedding settings.
    """

    api_key = get_required_environment_variable(
        "OPENAI_API_KEY"
    )

    model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    ).strip()

    if not model:
        raise RuntimeError(
            "OPENAI_EMBEDDING_MODEL cannot be empty."
        )

    dimensions = (
        get_positive_integer_environment_variable(
            "OPENAI_EMBEDDING_DIMENSIONS",
            1536,
        )
    )

    batch_size = (
        get_positive_integer_environment_variable(
            "EMBEDDING_BATCH_SIZE",
            64,
        )
    )

    return EmbeddingSettings(
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        base_url=(
            get_optional_environment_variable(
                "OPENAI_BASE_URL"
            )
        ),
    )


def get_vector_store_settings() -> VectorStoreSettings:
    """
    Builds local FAISS settings.
    """

    raw_index_path = os.getenv(
        "FAISS_INDEX_PATH",
        "local_indexes/policy_faiss",
    ).strip()

    if not raw_index_path:
        raise RuntimeError(
            "FAISS_INDEX_PATH cannot be empty."
        )

    retrieval_top_k = (
        get_positive_integer_environment_variable(
            "RETRIEVAL_TOP_K",
            5,
        )
    )

    retrieval_min_score = (
        get_float_environment_variable(
            "RETRIEVAL_MIN_SCORE",
            0.0,
        )
    )

    if not -1.0 <= retrieval_min_score <= 1.0:
        raise RuntimeError(
            "RETRIEVAL_MIN_SCORE must be between "
            "-1.0 and 1.0."
        )

    return VectorStoreSettings(
        index_path=Path(raw_index_path),
        retrieval_top_k=retrieval_top_k,
        retrieval_min_score=(
            retrieval_min_score
        ),
    )


def get_hybrid_retrieval_settings(
) -> HybridRetrievalSettings:
    """
    Builds hybrid retrieval settings.
    """

    vector_candidate_k = (
        get_positive_integer_environment_variable(
            "HYBRID_VECTOR_CANDIDATE_K",
            20,
        )
    )

    keyword_candidate_k = (
        get_positive_integer_environment_variable(
            "HYBRID_KEYWORD_CANDIDATE_K",
            20,
        )
    )

    keyword_minimum_score = (
        get_float_environment_variable(
            "KEYWORD_MINIMUM_SCORE",
            0.0,
        )
    )

    if keyword_minimum_score < 0:
        raise RuntimeError(
            "KEYWORD_MINIMUM_SCORE cannot be negative."
        )

    reciprocal_rank_constant = (
        get_positive_integer_environment_variable(
            "HYBRID_RRF_CONSTANT",
            60,
        )
    )

    vector_weight = (
        get_positive_float_environment_variable(
            "HYBRID_VECTOR_WEIGHT",
            1.0,
        )
    )

    keyword_weight = (
        get_positive_float_environment_variable(
            "HYBRID_KEYWORD_WEIGHT",
            1.0,
        )
    )

    return HybridRetrievalSettings(
        vector_candidate_k=vector_candidate_k,
        keyword_candidate_k=keyword_candidate_k,
        keyword_minimum_score=(
            keyword_minimum_score
        ),
        reciprocal_rank_constant=(
            reciprocal_rank_constant
        ),
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )


def get_reranker_settings() -> RerankerSettings:
    """
    Builds reranker settings.
    """

    backend = os.getenv(
        "RERANKER_BACKEND",
        "local_feature",
    ).strip().lower()

    allowed_backends = {
        "local_feature",
        "sentence_transformers",
    }

    if backend not in allowed_backends:
        raise RuntimeError(
            "RERANKER_BACKEND must be one of: "
            "local_feature, sentence_transformers."
        )

    candidate_k = (
        get_positive_integer_environment_variable(
            "RERANKER_CANDIDATE_K",
            10,
        )
    )

    top_n = (
        get_positive_integer_environment_variable(
            "RERANKER_TOP_N",
            5,
        )
    )

    if top_n > candidate_k:
        raise RuntimeError(
            "RERANKER_TOP_N cannot be greater than "
            "RERANKER_CANDIDATE_K."
        )

    model_name = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
    ).strip()

    if not model_name:
        raise RuntimeError(
            "RERANKER_MODEL cannot be empty."
        )

    device = os.getenv(
        "RERANKER_DEVICE",
        "cpu",
    ).strip().lower()

    if not device:
        raise RuntimeError(
            "RERANKER_DEVICE cannot be empty."
        )

    return RerankerSettings(
        backend=backend,
        candidate_k=candidate_k,
        top_n=top_n,
        model_name=model_name,
        device=device,
    )


def get_prompt_settings() -> PromptSettings:
    """
    Builds prompt and context settings.
    """

    max_context_tokens = (
        get_positive_integer_environment_variable(
            "CONTEXT_MAX_TOKENS",
            3_000,
        )
    )

    max_context_chunks = (
        get_positive_integer_environment_variable(
            "CONTEXT_MAX_CHUNKS",
            5,
        )
    )

    answer_max_tokens = (
        get_positive_integer_environment_variable(
            "ANSWER_MAX_TOKENS",
            700,
        )
    )

    token_encoding = os.getenv(
        "CONTEXT_TOKEN_ENCODING",
        "cl100k_base",
    ).strip()

    if not token_encoding:
        raise RuntimeError(
            "CONTEXT_TOKEN_ENCODING cannot be empty."
        )

    minimum_evidence_chunks = (
        get_positive_integer_environment_variable(
            "MINIMUM_EVIDENCE_CHUNKS",
            1,
        )
    )

    if (
        minimum_evidence_chunks
        > max_context_chunks
    ):
        raise RuntimeError(
            "MINIMUM_EVIDENCE_CHUNKS cannot be greater "
            "than CONTEXT_MAX_CHUNKS."
        )

    return PromptSettings(
        max_context_tokens=max_context_tokens,
        max_context_chunks=max_context_chunks,
        answer_max_tokens=answer_max_tokens,
        token_encoding=token_encoding,
        minimum_evidence_chunks=(
            minimum_evidence_chunks
        ),
    )


def get_generation_settings() -> GenerationSettings:
    """
    Builds answer-generation settings.
    """

    api_key = get_required_environment_variable(
        "OPENAI_API_KEY"
    )

    model = os.getenv(
        "GENERATION_MODEL",
        "gpt-4.1-mini",
    ).strip()

    if not model:
        raise RuntimeError(
            "GENERATION_MODEL cannot be empty."
        )

    max_output_tokens = (
        get_positive_integer_environment_variable(
            "ANSWER_MAX_TOKENS",
            700,
        )
    )

    temperature = (
        get_optional_float_environment_variable(
            "GENERATION_TEMPERATURE",
            0.0,
        )
    )

    timeout_seconds = (
        get_positive_float_environment_variable(
            "GENERATION_TIMEOUT_SECONDS",
            60.0,
        )
    )

    max_retries = (
        get_non_negative_integer_environment_variable(
            "GENERATION_MAX_RETRIES",
            2,
        )
    )

    use_responses_api = (
        get_boolean_environment_variable(
            "GENERATION_USE_RESPONSES_API",
            True,
        )
    )

    return GenerationSettings(
        api_key=api_key,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        use_responses_api=use_responses_api,
        base_url=(
            get_optional_environment_variable(
                "OPENAI_BASE_URL"
            )
        ),
    )