from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError
from redis import Redis
from redis.exceptions import RedisError

from app.cache.models import CachedRagResponse
from app.cache.settings import CacheSettings
from app.observability.metrics import (
    record_cache_error,
    record_cache_lookup,
    record_cache_write,
)


logger = logging.getLogger(__name__)


WHITESPACE_PATTERN = re.compile(r"\s+")


class RedisClientProtocol(Protocol):
    def ping(self) -> Any:
        ...

    def get(self, name: str) -> Any:
        ...

    def set(
        self,
        name: str,
        value: str,
        *,
        ex: int,
    ) -> Any:
        ...

    def delete(self, *names: str) -> Any:
        ...

    def close(self) -> Any:
        ...


def build_redis_client(
    settings: CacheSettings,
) -> Redis:
    """Creates one shared redis-py client and pool."""

    return Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=(
            settings.connect_timeout_seconds
        ),
        socket_timeout=(
            settings.socket_timeout_seconds
        ),
        health_check_interval=(
            settings.health_check_interval_seconds
        ),
    )


def normalize_question(question: str) -> str:
    """Normalizes harmless formatting differences."""

    return WHITESPACE_PATTERN.sub(
        " ",
        question.strip().lower(),
    )


class RedisRagResponseCache:
    """
    Permission-aware Redis cache for approved RAG output.

    Cache failures are fail-open: Redis unavailability never
    prevents the primary RAG pipeline from answering.
    """

    def __init__(
        self,
        *,
        settings: CacheSettings,
        index_fingerprint: str,
        embedding_model: str,
        generation_model: str,
        reranker_backend: str,
        client: RedisClientProtocol | None = None,
    ) -> None:
        self.settings = settings
        self.index_fingerprint = index_fingerprint
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.reranker_backend = reranker_backend

        self._client = (
            client
            if client is not None
            else build_redis_client(settings)
        )

        self._available = False

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def available(self) -> bool:
        return (
            self.settings.enabled
            and self._available
        )

    def initialize(self) -> bool:
        """Pings Redis without making startup fail."""

        if not self.settings.enabled:
            logger.info(
                "Redis response cache is disabled.",
                extra={
                    "event": "cache_disabled",
                },
            )
            return False

        try:
            self._client.ping()
            self._available = True

            logger.info(
                "Redis response cache is ready.",
                extra={
                    "event": "cache_ready",
                    "cache_ttl_seconds": (
                        self.settings.ttl_seconds
                    ),
                    "cache_schema_version": (
                        self.settings.schema_version
                    ),
                    "cache_prompt_version": (
                        self.settings.prompt_version
                    ),
                },
            )

        except (RedisError, OSError, TimeoutError) as error:
            self._available = False
            record_cache_error(
                operation="initialize",
                error_type=type(error).__name__,
            )

            logger.warning(
                "Redis response cache is unavailable; "
                "the RAG pipeline will continue without "
                "cache.",
                extra={
                    "event": "cache_unavailable",
                    "error_type": type(error).__name__,
                },
            )

        return self._available

    def close(self) -> None:
        """Closes Redis resources without failing shutdown."""

        try:
            self._client.close()

        except (RedisError, OSError) as error:
            record_cache_error(
                operation="close",
                error_type=type(error).__name__,
            )

            logger.warning(
                "Redis response cache could not close "
                "cleanly.",
                extra={
                    "event": "cache_close_failed",
                    "error_type": type(error).__name__,
                },
            )

        finally:
            self._available = False

    def build_key(
        self,
        *,
        question: str,
        role: str,
        region: str,
        clearance_rank: int,
    ) -> str:
        """
        Builds a hashed cache key that isolates access
        contexts and runtime versions.
        """

        canonical_scope = {
            "question": normalize_question(question),
            "role": role.strip().lower(),
            "region": region.strip().lower(),
            "clearance_rank": int(clearance_rank),
            "index_fingerprint": self.index_fingerprint,
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "reranker_backend": self.reranker_backend,
            "schema_version": (
                self.settings.schema_version
            ),
            "prompt_version": (
                self.settings.prompt_version
            ),
        }

        canonical_json = json.dumps(
            canonical_scope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

        digest = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()

        return (
            f"{self.settings.key_prefix}:"
            f"rag:{self.settings.schema_version}:"
            f"{digest}"
        )

    def get(
        self,
        cache_key: str,
    ) -> CachedRagResponse | None:
        """Returns a validated cached response or None."""

        if not self.available:
            return None

        try:
            raw_value = self._client.get(
                cache_key
            )

            if raw_value is None:
                record_cache_lookup(hit=False)
                return None

            cached_response = (
                CachedRagResponse.model_validate_json(
                    raw_value
                )
            )

            if (
                cached_response.schema_version
                != self.settings.schema_version
            ):
                self._client.delete(cache_key)
                record_cache_lookup(hit=False)
                return None

            record_cache_lookup(hit=True)
            return cached_response

        except (
            RedisError,
            OSError,
            TimeoutError,
            ValidationError,
            TypeError,
            ValueError,
        ) as error:
            record_cache_error(
                operation="get",
                error_type=type(error).__name__,
            )

            logger.warning(
                "Redis cache lookup failed; continuing "
                "with the primary RAG pipeline.",
                extra={
                    "event": "cache_lookup_failed",
                    "error_type": type(error).__name__,
                },
            )

            return None

    def set(
        self,
        cache_key: str,
        response: CachedRagResponse,
    ) -> bool:
        """Stores a response with an expiration time."""

        if not self.available:
            return False

        if (
            response.abstained
            and not self.settings.cache_abstentions
        ):
            return False

        try:
            self._client.set(
                cache_key,
                response.model_dump_json(),
                ex=self.settings.ttl_seconds,
            )

            record_cache_write()
            return True

        except (
            RedisError,
            OSError,
            TimeoutError,
        ) as error:
            record_cache_error(
                operation="set",
                error_type=type(error).__name__,
            )

            logger.warning(
                "Redis cache write failed; response will "
                "still be returned normally.",
                extra={
                    "event": "cache_write_failed",
                    "error_type": type(error).__name__,
                },
            )

            return False
