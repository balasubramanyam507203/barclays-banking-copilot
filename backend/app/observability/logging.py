from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone
from typing import Any

from app.observability.context import (
    get_conversation_id,
    get_request_id,
    get_trace_identifiers,
    get_user_id,
)


STANDARD_LOG_RECORD_FIELDS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


ALLOWED_EXTRA_FIELDS = {
    "event",
    "request_method",
    "request_path",
    "response_status_code",
    "duration_ms",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "evidence_count",
    "citations_count",
    "claims_checked",
    "supported_claims",
    "abstained",
    "model_called",
    "guardrail_passed",
    "role",
    "region",
    "clearance_rank",
    "error_type",
    "error_stage",
    "indexed_chunks",
    "embedding_model",
    "generation_model",
    "reranker_backend",
    "auth_mode",
}


def utc_timestamp(
    created: float,
) -> str:
    """Formats a LogRecord timestamp as UTC ISO-8601."""

    return (
        datetime.fromtimestamp(
            created,
            tz=timezone.utc,
        )
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class JsonLogFormatter(logging.Formatter):
    """Formats application logs as one JSON object per line."""

    def __init__(
        self,
        *,
        service_name: str,
        environment: str,
    ) -> None:
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        trace_id, span_id = (
            get_trace_identifiers()
        )

        payload: dict[str, Any] = {
            "timestamp": utc_timestamp(
                record.created
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "request_id": get_request_id(),
            "trace_id": trace_id,
            "span_id": span_id,
            "user_id": get_user_id(),
            "conversation_id": (
                get_conversation_id()
            ),
        }

        for field_name in ALLOWED_EXTRA_FIELDS:
            if hasattr(record, field_name):
                payload[field_name] = getattr(
                    record,
                    field_name,
                )

        if record.exc_info:
            payload["exception"] = (
                self.formatException(
                    record.exc_info
                )
            )

        if record.stack_info:
            payload["stack"] = (
                self.formatStack(
                    record.stack_info
                )
            )

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


class TextLogFormatter(logging.Formatter):
    """Readable development formatter with correlation IDs."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        trace_id, _ = get_trace_identifiers()

        timestamp = utc_timestamp(
            record.created
        )

        request_id = (
            get_request_id() or "-"
        )

        trace_value = trace_id or "-"

        base_message = (
            f"{timestamp} "
            f"{record.levelname:<8} "
            f"{record.name} "
            f"request_id={request_id} "
            f"trace_id={trace_value} "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            return (
                base_message
                + "\n"
                + self.formatException(
                    record.exc_info
                )
            )

        return base_message


def configure_logging(
    *,
    service_name: str,
    environment: str,
    log_level: str,
    log_format: str,
) -> None:
    """Configures root, Uvicorn, and application loggers."""

    selected_formatter = (
        "json"
        if log_format == "json"
        else "text"
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JsonLogFormatter,
                    "service_name": service_name,
                    "environment": environment,
                },
                "text": {
                    "()": TextLogFormatter,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": (
                        selected_formatter
                    ),
                    "level": log_level,
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": log_level,
                    "propagate": False,
                },
                "httpx": {
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
                "httpcore": {
                    "handlers": ["console"],
                    "level": "WARNING",
                    "propagate": False,
                },
            },
        }
    )
