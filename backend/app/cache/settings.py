from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class CacheSettings:
    """Redis cache configuration."""

    enabled: bool
    redis_url: str
    ttl_seconds: int
    key_prefix: str
    schema_version: str
    prompt_version: str
    connect_timeout_seconds: float
    socket_timeout_seconds: float
    health_check_interval_seconds: int
    cache_abstentions: bool


def parse_boolean(
    raw_value: str,
    *,
    variable_name: str,
) -> bool:
    normalized_value = raw_value.strip().lower()

    if normalized_value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized_value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise RuntimeError(
        f"{variable_name} must be true or false."
    )


def parse_positive_integer(
    raw_value: str,
    *,
    variable_name: str,
) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable_name} must be an integer."
        ) from error

    if value <= 0:
        raise RuntimeError(
            f"{variable_name} must be greater than 0."
        )

    return value


def parse_positive_float(
    raw_value: str,
    *,
    variable_name: str,
) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{variable_name} must be a number."
        ) from error

    if value <= 0:
        raise RuntimeError(
            f"{variable_name} must be greater than 0."
        )

    return value


def validate_redis_url(redis_url: str) -> str:
    cleaned_url = redis_url.strip()

    if not cleaned_url:
        raise RuntimeError(
            "REDIS_URL cannot be empty when caching is enabled."
        )

    parsed_url = urlparse(cleaned_url)

    if parsed_url.scheme not in {
        "redis",
        "rediss",
        "unix",
    }:
        raise RuntimeError(
            "REDIS_URL must use redis://, rediss://, or unix://."
        )

    return cleaned_url


def normalize_key_component(
    raw_value: str,
    *,
    variable_name: str,
) -> str:
    cleaned_value = raw_value.strip().strip(":")

    if not cleaned_value:
        raise RuntimeError(
            f"{variable_name} cannot be empty."
        )

    return cleaned_value


def get_cache_settings() -> CacheSettings:
    """Loads Redis cache settings from environment variables."""

    enabled = parse_boolean(
        os.getenv(
            "CACHE_ENABLED",
            "false",
        ),
        variable_name="CACHE_ENABLED",
    )

    raw_redis_url = os.getenv(
        "REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )

    redis_url = (
        validate_redis_url(raw_redis_url)
        if enabled
        else raw_redis_url.strip()
    )

    return CacheSettings(
        enabled=enabled,
        redis_url=redis_url,
        ttl_seconds=parse_positive_integer(
            os.getenv(
                "CACHE_TTL_SECONDS",
                "900",
            ),
            variable_name="CACHE_TTL_SECONDS",
        ),
        key_prefix=normalize_key_component(
            os.getenv(
                "CACHE_KEY_PREFIX",
                "enterprise-banking-policy-copilot",
            ),
            variable_name="CACHE_KEY_PREFIX",
        ),
        schema_version=normalize_key_component(
            os.getenv(
                "CACHE_SCHEMA_VERSION",
                "v1",
            ),
            variable_name="CACHE_SCHEMA_VERSION",
        ),
        prompt_version=normalize_key_component(
            os.getenv(
                "CACHE_PROMPT_VERSION",
                "v1",
            ),
            variable_name="CACHE_PROMPT_VERSION",
        ),
        connect_timeout_seconds=(
            parse_positive_float(
                os.getenv(
                    "CACHE_CONNECT_TIMEOUT_SECONDS",
                    "1.0",
                ),
                variable_name=(
                    "CACHE_CONNECT_TIMEOUT_SECONDS"
                ),
            )
        ),
        socket_timeout_seconds=(
            parse_positive_float(
                os.getenv(
                    "CACHE_SOCKET_TIMEOUT_SECONDS",
                    "1.0",
                ),
                variable_name=(
                    "CACHE_SOCKET_TIMEOUT_SECONDS"
                ),
            )
        ),
        health_check_interval_seconds=(
            parse_positive_integer(
                os.getenv(
                    "CACHE_HEALTH_CHECK_INTERVAL_SECONDS",
                    "30",
                ),
                variable_name=(
                    "CACHE_HEALTH_CHECK_INTERVAL_SECONDS"
                ),
            )
        ),
        cache_abstentions=parse_boolean(
            os.getenv(
                "CACHE_ABSTENTIONS",
                "true",
            ),
            variable_name="CACHE_ABSTENTIONS",
        ),
    )
