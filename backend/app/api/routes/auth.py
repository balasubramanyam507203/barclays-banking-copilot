from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.api.dependencies import (
    CurrentPrincipalDependency,
    ServicesDependency,
)
from app.api.schemas import (
    AccessTokenResponse,
    CurrentUserResponse,
    DevelopmentLoginRequest,
)
from app.security.authentication import (
    AuthenticatedPrincipal,
    AuthenticationError,
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


def build_user_response(
    principal: AuthenticatedPrincipal,
) -> CurrentUserResponse:
    return CurrentUserResponse(
        subject=principal.subject,
        username=principal.username,
        role=principal.role,
        region=principal.region,
        clearance_rank=(
            principal.clearance_rank
        ),
        groups=list(principal.groups),
    )


@router.post(
    "/dev-login",
    response_model=AccessTokenResponse,
)
def development_login(
    payload: DevelopmentLoginRequest,
    services: ServicesDependency,
) -> AccessTokenResponse:
    """
    Local-only login endpoint.

    It is unavailable when AUTH_MODE=cognito.
    """

    if services.auth_mode != "local_jwt":
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Endpoint not found.",
        )

    try:
        issued_token = (
            services
            .authentication_service
            .issue_development_token(
                profile_name=payload.profile,
                password=payload.password,
            )
        )

    except AuthenticationError as error:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=str(error),
            headers={
                "WWW-Authenticate": "Bearer"
            },
        ) from error

    return AccessTokenResponse(
        access_token=(
            issued_token.access_token
        ),
        expires_in=(
            issued_token.expires_in_seconds
        ),
        user=build_user_response(
            issued_token.principal
        ),
    )


@router.get(
    "/me",
    response_model=CurrentUserResponse,
)
def get_current_user(
    principal: CurrentPrincipalDependency,
) -> CurrentUserResponse:
    """
    Returns identity derived from the verified token.
    """

    return build_user_response(
        principal
    )