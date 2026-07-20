import logging
import os
from dataclasses import dataclass


VALID_LOG_FORMATS = {
    "json",
    "text",
}


@dataclass(frozen=True)
class ApiSettings:
    """FastAPI and observability configuration."""

    title: str
    version: str
    prefix: str

    service_name: str
    environment: str

    cors_origins: tuple[str, ...]

    log_level: str
    log_format: str

    otel_enabled: bool
    otel_console_exporter: bool
    otel_otlp_endpoint: str | None
    otel_excluded_urls: tuple[str, ...]
    otel_metric_export_interval_ms: int


def normalize_api_prefix(
    prefix: str,
) -> str:
    """Normalizes the API prefix."""

    cleaned_prefix = prefix.strip()

    if not cleaned_prefix:
        raise RuntimeError(
            "API_PREFIX cannot be empty."
        )

    if not cleaned_prefix.startswith("/"):
        cleaned_prefix = f"/{cleaned_prefix}"

    return cleaned_prefix.rstrip("/")


def parse_cors_origins(
    raw_origins: str,
) -> tuple[str, ...]:
    """Parses comma-separated frontend origins."""

    origins = tuple(
        origin.strip().rstrip("/")
        for origin in raw_origins.split(",")
        if origin.strip()
    )

    if not origins:
        raise RuntimeError(
            "API_CORS_ORIGINS must contain at least "
            "one allowed frontend origin."
        )

    if "*" in origins:
        raise RuntimeError(
            "API_CORS_ORIGINS cannot use '*' because "
            "the API allows credentialed requests."
        )

    return origins


def parse_csv_values(
    raw_values: str,
) -> tuple[str, ...]:
    """Parses comma-separated non-empty values."""

    return tuple(
        value.strip()
        for value in raw_values.split(",")
        if value.strip()
    )


def parse_boolean(
    raw_value: str,
    *,
    variable_name: str,
) -> bool:
    """Parses an environment variable as a boolean."""

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
    """Parses a strictly positive integer."""

    try:
        parsed_value = int(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"{variable_name} must be an integer."
        ) from error

    if parsed_value <= 0:
        raise RuntimeError(
            f"{variable_name} must be greater than 0."
        )

    return parsed_value


def normalize_log_level(
    raw_level: str,
) -> str:
    """Validates a Python logging level."""

    normalized_level = raw_level.strip().upper()

    if normalized_level not in logging.getLevelNamesMapping():
        raise RuntimeError(
            "OBSERVABILITY_LOG_LEVEL is not a valid "
            f"logging level: '{raw_level}'."
        )

    return normalized_level


def normalize_optional_value(
    raw_value: str | None,
) -> str | None:
    """Returns a stripped string or None."""

    if raw_value is None:
        return None

    cleaned_value = raw_value.strip()

    return cleaned_value or None


def get_api_settings() -> ApiSettings:
    """Loads API and observability environment values."""

    title = os.getenv(
        "API_TITLE",
        "Enterprise Banking Policy Copilot",
    ).strip()

    version = os.getenv(
        "API_VERSION",
        "1.0.0",
    ).strip()

    prefix = normalize_api_prefix(
        os.getenv(
            "API_PREFIX",
            "/api/v1",
        )
    )

    service_name = os.getenv(
        "API_SERVICE_NAME",
        "enterprise-banking-policy-copilot",
    ).strip()

    environment = os.getenv(
        "APP_ENVIRONMENT",
        "development",
    ).strip().lower()

    raw_cors_origins = os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:3000",
    )

    log_level = normalize_log_level(
        os.getenv(
            "OBSERVABILITY_LOG_LEVEL",
            "INFO",
        )
    )

    log_format = os.getenv(
        "OBSERVABILITY_LOG_FORMAT",
        "json",
    ).strip().lower()

    if log_format not in VALID_LOG_FORMATS:
        raise RuntimeError(
            "OBSERVABILITY_LOG_FORMAT must be "
            "'json' or 'text'."
        )

    if not title:
        raise RuntimeError(
            "API_TITLE cannot be empty."
        )

    if not version:
        raise RuntimeError(
            "API_VERSION cannot be empty."
        )

    if not service_name:
        raise RuntimeError(
            "API_SERVICE_NAME cannot be empty."
        )

    if not environment:
        raise RuntimeError(
            "APP_ENVIRONMENT cannot be empty."
        )

    return ApiSettings(
        title=title,
        version=version,
        prefix=prefix,
        service_name=service_name,
        environment=environment,
        cors_origins=parse_cors_origins(
            raw_cors_origins
        ),
        log_level=log_level,
        log_format=log_format,
        otel_enabled=parse_boolean(
            os.getenv(
                "OTEL_ENABLED",
                "false",
            ),
            variable_name="OTEL_ENABLED",
        ),
        otel_console_exporter=parse_boolean(
            os.getenv(
                "OTEL_CONSOLE_EXPORTER",
                "false",
            ),
            variable_name=(
                "OTEL_CONSOLE_EXPORTER"
            ),
        ),
        otel_otlp_endpoint=normalize_optional_value(
            os.getenv(
                "OTEL_EXPORTER_OTLP_ENDPOINT"
            )
        ),
        otel_excluded_urls=parse_csv_values(
            os.getenv(
                "OTEL_PYTHON_FASTAPI_EXCLUDED_URLS",
                (
                    "/api/v1/health/live,"
                    "/api/v1/health/ready"
                ),
            )
        ),
        otel_metric_export_interval_ms=(
            parse_positive_integer(
                os.getenv(
                    "OTEL_METRIC_EXPORT_INTERVAL_MS",
                    "60000",
                ),
                variable_name=(
                    "OTEL_METRIC_EXPORT_INTERVAL_MS"
                ),
            )
        ),
    )
