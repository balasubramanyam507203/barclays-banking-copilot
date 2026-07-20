from collections.abc import Iterator
from typing import Annotated, TypeAlias

from fastapi import (
    Depends,
    HTTPException,
    Request,
    Security,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from sqlalchemy.orm import Session

from app.api.services import (
    ApplicationServices,
)
from app.rag.faiss_store import (
    SearchAccessContext,
)
from app.security.authentication import (
    AuthenticatedPrincipal,
    AuthenticationError,
)


bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="EmployeeAccessToken",
    description=(
        "Validated local or Amazon Cognito "
        "access token."
    ),
)


def get_application_services(
    request: Request,
) -> ApplicationServices:
    """
    Returns the long-lived services loaded during
    FastAPI application startup.
    """

    services = getattr(
        request.app.state,
        "services",
        None,
    )

    if services is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "The policy service is not ready."
            ),
        )

    return services


ServicesDependency: TypeAlias = Annotated[
    ApplicationServices,
    Depends(get_application_services),
]


CredentialsDependency: TypeAlias = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(bearer_scheme),
]


def get_current_principal(
    credentials: CredentialsDependency,
    services: ServicesDependency,
) -> AuthenticatedPrincipal:
    """
    Verifies the bearer access token and returns the
    authenticated employee identity.
    """

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "A valid bearer access token is "
                "required."
            ),
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    try:
        return (
            services
            .authentication_service
            .verify_access_token(
                credentials.credentials
            )
        )

    except AuthenticationError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=str(error),
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from error


CurrentPrincipalDependency: TypeAlias = Annotated[
    AuthenticatedPrincipal,
    Depends(get_current_principal),
]


def get_access_context(
    principal: CurrentPrincipalDependency,
) -> SearchAccessContext:
    """
    Builds retrieval permissions only from verified
    authentication claims.
    """

    return SearchAccessContext(
        role=principal.role,
        region=principal.region,
        clearance_rank=(
            principal.clearance_rank
        ),
    )


AccessContextDependency: TypeAlias = Annotated[
    SearchAccessContext,
    Depends(get_access_context),
]


def get_database_session(
    services: ServicesDependency,
) -> Iterator[Session]:
    """
    Provides one SQLAlchemy session for one API request.
    """

    with (
        services
        .database_service
        .session()
    ) as session:
        yield session


DatabaseSessionDependency: TypeAlias = Annotated[
    Session,
    Depends(get_database_session),
]