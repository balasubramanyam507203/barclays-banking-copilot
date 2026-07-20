import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability.context import (
    reset_request_id,
    set_request_id,
)
from app.observability.logging import (
    JsonLogFormatter,
)
from app.observability.middleware import (
    RequestContextMiddleware,
)


def test_json_formatter_contains_request_context() -> None:
    formatter = JsonLogFormatter(
        service_name="policy-copilot-test",
        environment="test",
    )

    request_token = set_request_id(
        "request-123"
    )

    try:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Request completed",
            args=(),
            exc_info=None,
        )

        record.event = "test_event"
        record.duration_ms = 12.5

        payload = json.loads(
            formatter.format(record)
        )

    finally:
        reset_request_id(request_token)

    assert payload["message"] == (
        "Request completed"
    )
    assert payload["request_id"] == (
        "request-123"
    )
    assert payload["event"] == "test_event"
    assert payload["duration_ms"] == 12.5
    assert payload["service"] == (
        "policy-copilot-test"
    )
    assert payload["environment"] == "test"


def build_test_application() -> FastAPI:
    application = FastAPI()

    application.add_middleware(
        RequestContextMiddleware
    )

    @application.get("/test")
    def test_route() -> dict[str, str]:
        return {
            "status": "ok",
        }

    return application


def test_request_id_is_preserved() -> None:
    application = build_test_application()

    with TestClient(application) as client:
        response = client.get(
            "/test",
            headers={
                "X-Request-ID": "client-request-123",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == (
        "client-request-123"
    )


def test_invalid_request_id_is_replaced() -> None:
    application = build_test_application()

    with TestClient(application) as client:
        response = client.get(
            "/test",
            headers={
                "X-Request-ID": "invalid request id",
            },
        )

    response_request_id = response.headers[
        "X-Request-ID"
    ]

    assert response.status_code == 200
    assert response_request_id != (
        "invalid request id"
    )
    assert len(response_request_id) == 36
