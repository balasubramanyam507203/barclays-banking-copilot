from __future__ import annotations

from opentelemetry import metrics


_meter = metrics.get_meter(
    "enterprise-banking-policy-copilot.rag",
    "1.0.0",
)


RAG_REQUESTS = _meter.create_counter(
    name="rag_requests_total",
    description=(
        "Number of completed RAG chat requests."
    ),
    unit="1",
)

RAG_ERRORS = _meter.create_counter(
    name="rag_errors_total",
    description=(
        "Number of RAG pipeline failures."
    ),
    unit="1",
)

RAG_ABSTENTIONS = _meter.create_counter(
    name="rag_abstentions_total",
    description=(
        "Number of guarded RAG abstentions."
    ),
    unit="1",
)

RAG_GUARDRAIL_FAILURES = _meter.create_counter(
    name="rag_guardrail_failures_total",
    description=(
        "Number of answers rejected by generation "
        "guardrails."
    ),
    unit="1",
)

RAG_CACHE_LOOKUPS = _meter.create_counter(
    name="rag_cache_lookups_total",
    description=(
        "Number of Redis RAG cache lookups."
    ),
    unit="1",
)

RAG_CACHE_HITS = _meter.create_counter(
    name="rag_cache_hits_total",
    description=(
        "Number of successful Redis RAG cache hits."
    ),
    unit="1",
)

RAG_CACHE_MISSES = _meter.create_counter(
    name="rag_cache_misses_total",
    description=(
        "Number of Redis RAG cache misses."
    ),
    unit="1",
)

RAG_CACHE_WRITES = _meter.create_counter(
    name="rag_cache_writes_total",
    description=(
        "Number of guarded RAG responses written to Redis."
    ),
    unit="1",
)

RAG_CACHE_ERRORS = _meter.create_counter(
    name="rag_cache_errors_total",
    description=(
        "Number of Redis cache errors."
    ),
    unit="1",
)

RAG_RETRIEVAL_LATENCY = _meter.create_histogram(
    name="rag_retrieval_latency_ms",
    description=(
        "Retrieval and reranking latency."
    ),
    unit="ms",
)

RAG_GENERATION_LATENCY = _meter.create_histogram(
    name="rag_generation_latency_ms",
    description=(
        "LLM generation and post-generation "
        "guardrail latency."
    ),
    unit="ms",
)

RAG_TOTAL_LATENCY = _meter.create_histogram(
    name="rag_total_latency_ms",
    description=(
        "End-to-end guarded RAG latency."
    ),
    unit="ms",
)


def build_metric_attributes(
    *,
    role: str,
    status: str,
    cache_hit: bool,
) -> dict[str, str | bool]:
    """Builds low-cardinality metric attributes."""

    return {
        "user.role": role,
        "rag.status": status,
        "cache.hit": cache_hit,
    }


def record_rag_success(
    *,
    role: str,
    status: str,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    abstained: bool,
    guardrail_passed: bool,
    cache_hit: bool = False,
) -> None:
    """Records metrics for a completed RAG request."""

    attributes = build_metric_attributes(
        role=role,
        status=status,
        cache_hit=cache_hit,
    )

    RAG_REQUESTS.add(
        1,
        attributes,
    )

    RAG_RETRIEVAL_LATENCY.record(
        retrieval_latency_ms,
        attributes,
    )

    RAG_GENERATION_LATENCY.record(
        generation_latency_ms,
        attributes,
    )

    RAG_TOTAL_LATENCY.record(
        total_latency_ms,
        attributes,
    )

    if abstained:
        RAG_ABSTENTIONS.add(
            1,
            attributes,
        )

    if not guardrail_passed:
        RAG_GUARDRAIL_FAILURES.add(
            1,
            attributes,
        )


def record_rag_error(
    *,
    role: str,
    stage: str,
    error_type: str,
) -> None:
    """Records a RAG pipeline error."""

    RAG_ERRORS.add(
        1,
        {
            "user.role": role,
            "rag.error_stage": stage,
            "error.type": error_type,
        },
    )


def record_cache_lookup(
    *,
    hit: bool,
) -> None:
    """Records a Redis cache lookup result."""

    attributes = {
        "cache.hit": hit,
    }

    RAG_CACHE_LOOKUPS.add(
        1,
        attributes,
    )

    if hit:
        RAG_CACHE_HITS.add(
            1,
            attributes,
        )
    else:
        RAG_CACHE_MISSES.add(
            1,
            attributes,
        )


def record_cache_write() -> None:
    """Records a successful cache write."""

    RAG_CACHE_WRITES.add(1)


def record_cache_error(
    *,
    operation: str,
    error_type: str,
) -> None:
    """Records a Redis cache error."""

    RAG_CACHE_ERRORS.add(
        1,
        {
            "cache.operation": operation,
            "error.type": error_type,
        },
    )
