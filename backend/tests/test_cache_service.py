from __future__ import annotations

from typing import Any

from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.models import CachedRagResponse
from app.cache.service import RedisRagResponseCache
from app.cache.settings import CacheSettings


class FakeRedis:
    def __init__(
        self,
        *,
        ping_error: Exception | None = None,
    ) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.deleted_keys: list[str] = []
        self.closed = False
        self.ping_error = ping_error

    def ping(self) -> bool:
        if self.ping_error is not None:
            raise self.ping_error

        return True

    def get(self, name: str) -> str | None:
        return self.values.get(name)

    def set(
        self,
        name: str,
        value: str,
        *,
        ex: int,
    ) -> bool:
        self.values[name] = value
        self.expirations[name] = ex
        return True

    def delete(self, *names: str) -> int:
        deleted_count = 0

        for name in names:
            self.deleted_keys.append(name)

            if name in self.values:
                del self.values[name]
                deleted_count += 1

        return deleted_count

    def close(self) -> None:
        self.closed = True


def build_settings(
    *,
    enabled: bool = True,
) -> CacheSettings:
    return CacheSettings(
        enabled=enabled,
        redis_url="redis://127.0.0.1:6379/0",
        ttl_seconds=900,
        key_prefix=(
            "enterprise-banking-policy-copilot"
        ),
        schema_version="v1",
        prompt_version="v1",
        connect_timeout_seconds=1.0,
        socket_timeout_seconds=1.0,
        health_check_interval_seconds=30,
        cache_abstentions=True,
    )


def build_cache(
    client: Any,
    *,
    index_fingerprint: str = "index-a",
) -> RedisRagResponseCache:
    return RedisRagResponseCache(
        settings=build_settings(),
        index_fingerprint=index_fingerprint,
        embedding_model="embedding-model",
        generation_model="generation-model",
        reranker_backend="cross-encoder",
        client=client,
    )


def build_payload() -> CachedRagResponse:
    return CachedRagResponse(
        schema_version="v1",
        status="answered",
        answer=(
            "Identity verification is required "
            "[SOURCE 1].\n\nSources:\n[SOURCE 1]"
        ),
        abstained=False,
        citations_used=["SOURCE 1"],
        sources=[
            {
                "label": "SOURCE 1",
                "document_id": "KYC-POL-2031",
            }
        ],
        evidence_count=1,
        guardrails={
            "citation_validation_passed": True,
            "post_generation_guardrails_passed": True,
            "claims_checked": 1,
            "supported_claims": 1,
        },
    )


def test_cache_key_normalizes_question_whitespace_and_case(
) -> None:
    cache = build_cache(FakeRedis())

    first_key = cache.build_key(
        question="What  Identity Verification is Required?",
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    second_key = cache.build_key(
        question=" what identity verification is required? ",
        role="compliance_analyst",
        region="us",
        clearance_rank=2,
    )

    assert first_key == second_key


def test_cache_key_isolates_access_context() -> None:
    cache = build_cache(FakeRedis())

    analyst_key = cache.build_key(
        question="What verification is required?",
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    support_key = cache.build_key(
        question="What verification is required?",
        role="customer_support",
        region="US",
        clearance_rank=1,
    )

    assert analyst_key != support_key


def test_cache_key_changes_when_index_changes() -> None:
    first_cache = build_cache(
        FakeRedis(),
        index_fingerprint="index-a",
    )

    second_cache = build_cache(
        FakeRedis(),
        index_fingerprint="index-b",
    )

    key_arguments = {
        "question": "How should a complaint be handled?",
        "role": "customer_support",
        "region": "US",
        "clearance_rank": 1,
    }

    assert (
        first_cache.build_key(**key_arguments)
        != second_cache.build_key(**key_arguments)
    )


def test_cache_set_and_get_use_ttl() -> None:
    fake_redis = FakeRedis()
    cache = build_cache(fake_redis)

    assert cache.initialize() is True

    cache_key = cache.build_key(
        question="What verification is required?",
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
    )

    payload = build_payload()

    assert cache.set(cache_key, payload) is True
    assert fake_redis.expirations[cache_key] == 900

    cached_payload = cache.get(cache_key)

    assert cached_payload is not None
    assert cached_payload.answer == payload.answer
    assert cached_payload.citations_used == [
        "SOURCE 1"
    ]


def test_invalid_cached_json_fails_open() -> None:
    fake_redis = FakeRedis()
    cache = build_cache(fake_redis)

    assert cache.initialize() is True

    cache_key = "test:invalid"
    fake_redis.values[cache_key] = "not valid json"

    assert cache.get(cache_key) is None


def test_unavailable_redis_does_not_raise() -> None:
    fake_redis = FakeRedis(
        ping_error=RedisConnectionError(
            "Redis unavailable"
        )
    )

    cache = build_cache(fake_redis)

    assert cache.initialize() is False
    assert cache.available is False
    assert cache.get("missing") is None
