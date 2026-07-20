from __future__ import annotations

from contextvars import ContextVar, Token
from uuid import uuid4

from opentelemetry import trace


_request_id_context: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

_user_id_context: ContextVar[str | None] = ContextVar(
    "user_id",
    default=None,
)

_conversation_id_context: ContextVar[str | None] = ContextVar(
    "conversation_id",
    default=None,
)


def create_request_id() -> str:
    """Creates a new request/correlation identifier."""

    return str(uuid4())


def get_request_id() -> str | None:
    """Returns the current request identifier."""

    return _request_id_context.get()


def get_or_create_request_id() -> str:
    """Returns the current request ID or creates one."""

    request_id = get_request_id()

    if request_id:
        return request_id

    request_id = create_request_id()
    _request_id_context.set(request_id)

    return request_id


def set_request_id(
    request_id: str,
) -> Token[str | None]:
    """Binds a request ID to the current execution context."""

    return _request_id_context.set(request_id)


def reset_request_id(
    token: Token[str | None],
) -> None:
    """Restores the previous request ID context."""

    _request_id_context.reset(token)


def get_user_id() -> str | None:
    """Returns the current authenticated subject."""

    return _user_id_context.get()


def set_user_id(
    user_id: str | None,
) -> Token[str | None]:
    """Binds an authenticated subject to the context."""

    return _user_id_context.set(user_id)


def reset_user_id(
    token: Token[str | None],
) -> None:
    """Restores the previous user context."""

    _user_id_context.reset(token)


def get_conversation_id() -> str | None:
    """Returns the current conversation identifier."""

    return _conversation_id_context.get()


def set_conversation_id(
    conversation_id: str | None,
) -> Token[str | None]:
    """Binds a conversation ID to the current context."""

    return _conversation_id_context.set(
        conversation_id
    )


def reset_conversation_id(
    token: Token[str | None],
) -> None:
    """Restores the previous conversation context."""

    _conversation_id_context.reset(token)


def get_trace_identifiers() -> tuple[str | None, str | None]:
    """Returns active OpenTelemetry trace and span IDs."""

    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()

    if not span_context.is_valid:
        return None, None

    trace_id = format(
        span_context.trace_id,
        "032x",
    )

    span_id = format(
        span_context.span_id,
        "016x",
    )

    return trace_id, span_id
