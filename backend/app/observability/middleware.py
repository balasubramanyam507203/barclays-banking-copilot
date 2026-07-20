from __future__ import annotations

import logging
import re
from time import perf_counter
from typing import Any

from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from app.observability.context import (
    create_request_id,
    reset_request_id,
    set_request_id,
)


logger = logging.getLogger(__name__)


REQUEST_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)


class RequestContextMiddleware:
    """
    Creates or accepts a safe request ID, binds it to a
    ContextVar, writes it to the response, and logs request
    completion without logging query strings or bodies.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        request_header_name: str = (
            "x-request-id"
        ),
        response_header_name: bytes = (
            b"x-request-id"
        ),
    ) -> None:
        self.app = app
        self.request_header_name = (
            request_header_name.lower().encode(
                "latin-1"
            )
        )
        self.response_header_name = (
            response_header_name
        )

    @staticmethod
    def select_request_id(
        scope: Scope,
        *,
        request_header_name: bytes,
    ) -> str:
        incoming_request_id = None

        for header_name, header_value in scope.get(
            "headers",
            [],
        ):
            if (
                header_name.lower()
                == request_header_name
            ):
                incoming_request_id = (
                    header_value.decode(
                        "latin-1"
                    ).strip()
                )
                break

        if (
            incoming_request_id
            and REQUEST_ID_PATTERN.fullmatch(
                incoming_request_id
            )
        ):
            return incoming_request_id

        return create_request_id()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        request_id = self.select_request_id(
            scope,
            request_header_name=(
                self.request_header_name
            ),
        )

        request_id_token = set_request_id(
            request_id
        )

        started_at = perf_counter()
        response_status_code = 500

        async def send_with_request_id(
            message: Message,
        ) -> None:
            nonlocal response_status_code

            if message["type"] == "http.response.start":
                response_status_code = int(
                    message.get(
                        "status",
                        500,
                    )
                )

                headers = list(
                    message.get(
                        "headers",
                        [],
                    )
                )

                headers = [
                    (
                        header_name,
                        header_value,
                    )
                    for header_name, header_value
                    in headers
                    if header_name.lower()
                    != self.response_header_name
                ]

                headers.append(
                    (
                        self.response_header_name,
                        request_id.encode(
                            "latin-1"
                        ),
                    )
                )

                message["headers"] = headers

            await send(message)

        method = str(
            scope.get(
                "method",
                "",
            )
        )

        path = str(
            scope.get(
                "path",
                "",
            )
        )

        try:
            await self.app(
                scope,
                receive,
                send_with_request_id,
            )

        except Exception as error:
            duration_ms = (
                perf_counter() - started_at
            ) * 1_000

            logger.exception(
                "HTTP request failed.",
                extra={
                    "event": "http_request_failed",
                    "request_method": method,
                    "request_path": path,
                    "response_status_code": 500,
                    "duration_ms": round(
                        duration_ms,
                        3,
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                },
            )

            raise

        else:
            duration_ms = (
                perf_counter() - started_at
            ) * 1_000

            logger.info(
                "HTTP request completed.",
                extra={
                    "event": (
                        "http_request_completed"
                    ),
                    "request_method": method,
                    "request_path": path,
                    "response_status_code": (
                        response_status_code
                    ),
                    "duration_ms": round(
                        duration_ms,
                        3,
                    ),
                },
            )

        finally:
            reset_request_id(
                request_id_token
            )
