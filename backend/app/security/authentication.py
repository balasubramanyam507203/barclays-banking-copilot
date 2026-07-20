from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt
from jwt import PyJWKClient


LOCAL_ALGORITHM = "HS256"
COGNITO_ALGORITHM = "RS256"


@dataclass(frozen=True)
class AccessPolicy:
    """
    Permission mapping for one authenticated group.
    """

    role: str
    clearance_rank: int


COGNITO_GROUP_POLICIES: dict[str, AccessPolicy] = {
    "CustomerSupport": AccessPolicy(
        role="customer_support",
        clearance_rank=1,
    ),
    "ComplianceAnalysts": AccessPolicy(
        role="compliance_analyst",
        clearance_rank=2,
    ),
    "SecurityInvestigators": AccessPolicy(
        role="security_investigator",
        clearance_rank=3,
    ),
}


@dataclass(frozen=True)
class DevelopmentProfile:
    """
    Local-only employee identity.
    """

    subject: str
    username: str
    role: str
    region: str
    clearance_rank: int
    groups: tuple[str, ...]


DEVELOPMENT_PROFILES: dict[
    str,
    DevelopmentProfile,
] = {
    "compliance_analyst": DevelopmentProfile(
        subject="local-compliance-001",
        username="compliance.analyst",
        role="compliance_analyst",
        region="US",
        clearance_rank=2,
        groups=("ComplianceAnalysts",),
    ),
    "customer_support": DevelopmentProfile(
        subject="local-support-001",
        username="customer.support",
        role="customer_support",
        region="US",
        clearance_rank=1,
        groups=("CustomerSupport",),
    ),
    "security_investigator": DevelopmentProfile(
        subject="local-security-001",
        username="security.investigator",
        role="security_investigator",
        region="US",
        clearance_rank=3,
        groups=("SecurityInvestigators",),
    ),
}


@dataclass(frozen=True)
class AuthenticationSettings:
    """
    Authentication configuration.

    local_jwt:
        Local development tokens signed with HS256.

    cognito:
        Production access tokens signed by Cognito
        using RS256 and verified through JWKS.
    """

    environment: str
    mode: str

    local_secret: str | None
    local_issuer: str
    local_audience: str
    local_expiration_minutes: int
    development_password: str | None

    cognito_region: str | None
    cognito_user_pool_id: str | None
    cognito_app_client_id: str | None
    cognito_required_scope: str | None

    @property
    def cognito_issuer(self) -> str:
        if (
            self.cognito_region is None
            or self.cognito_user_pool_id is None
        ):
            raise RuntimeError(
                "Cognito issuer settings are incomplete."
            )

        return (
            "https://cognito-idp."
            f"{self.cognito_region}.amazonaws.com/"
            f"{self.cognito_user_pool_id}"
        )

    @property
    def cognito_jwks_url(self) -> str:
        return (
            f"{self.cognito_issuer}/"
            ".well-known/jwks.json"
        )


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """
    Trusted user identity produced after token
    verification.
    """

    subject: str
    username: str

    role: str
    region: str
    clearance_rank: int

    groups: tuple[str, ...]


@dataclass(frozen=True)
class IssuedAccessToken:
    """
    Local development access token result.
    """

    access_token: str
    expires_in_seconds: int
    principal: AuthenticatedPrincipal


class AuthenticationError(Exception):
    """
    Raised when authentication fails.
    """


def get_optional_environment_value(
    name: str,
) -> str | None:
    value = os.getenv(name)

    if value is None:
        return None

    cleaned = value.strip()

    return cleaned or None


def get_required_environment_value(
    name: str,
) -> str:
    value = get_optional_environment_value(
        name
    )

    if value is None:
        raise RuntimeError(
            f"Required environment variable "
            f"'{name}' is missing."
        )

    return value


def get_positive_integer(
    name: str,
    default: int,
) -> int:
    raw_value = os.getenv(
        name,
        str(default),
    ).strip()

    try:
        value = int(raw_value)

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{name}' must "
            "be an integer."
        ) from error

    if value <= 0:
        raise RuntimeError(
            f"Environment variable '{name}' must "
            "be greater than zero."
        )

    return value


def get_authentication_settings(
) -> AuthenticationSettings:
    environment = os.getenv(
        "APP_ENVIRONMENT",
        "development",
    ).strip().lower()

    mode = os.getenv(
        "AUTH_MODE",
        "local_jwt",
    ).strip().lower()

    if mode not in {
        "local_jwt",
        "cognito",
    }:
        raise RuntimeError(
            "AUTH_MODE must be either "
            "'local_jwt' or 'cognito'."
        )

    settings = AuthenticationSettings(
        environment=environment,
        mode=mode,
        local_secret=(
            get_optional_environment_value(
                "LOCAL_JWT_SECRET"
            )
        ),
        local_issuer=os.getenv(
            "LOCAL_JWT_ISSUER",
            "banking-policy-copilot-local",
        ).strip(),
        local_audience=os.getenv(
            "LOCAL_JWT_AUDIENCE",
            "banking-policy-copilot-api",
        ).strip(),
        local_expiration_minutes=(
            get_positive_integer(
                "LOCAL_JWT_EXPIRATION_MINUTES",
                60,
            )
        ),
        development_password=(
            get_optional_environment_value(
                "DEV_AUTH_PASSWORD"
            )
        ),
        cognito_region=(
            get_optional_environment_value(
                "COGNITO_REGION"
            )
        ),
        cognito_user_pool_id=(
            get_optional_environment_value(
                "COGNITO_USER_POOL_ID"
            )
        ),
        cognito_app_client_id=(
            get_optional_environment_value(
                "COGNITO_APP_CLIENT_ID"
            )
        ),
        cognito_required_scope=(
            get_optional_environment_value(
                "COGNITO_REQUIRED_SCOPE"
            )
        ),
    )

    validate_authentication_settings(
        settings
    )

    return settings


def validate_authentication_settings(
    settings: AuthenticationSettings,
) -> None:
    if settings.mode == "local_jwt":
        if settings.environment == "production":
            raise RuntimeError(
                "local_jwt authentication cannot be "
                "used in production."
            )

        if (
            settings.local_secret is None
            or len(settings.local_secret) < 32
        ):
            raise RuntimeError(
                "LOCAL_JWT_SECRET must contain at "
                "least 32 characters."
            )

        if settings.development_password is None:
            raise RuntimeError(
                "DEV_AUTH_PASSWORD is required for "
                "local authentication."
            )

        return

    missing_settings = [
        name
        for name, value in {
            "COGNITO_REGION":
                settings.cognito_region,
            "COGNITO_USER_POOL_ID":
                settings.cognito_user_pool_id,
            "COGNITO_APP_CLIENT_ID":
                settings.cognito_app_client_id,
        }.items()
        if value is None
    ]

    if missing_settings:
        raise RuntimeError(
            "Missing Cognito settings: "
            + ", ".join(missing_settings)
        )


def normalize_groups(
    raw_groups: Any,
) -> tuple[str, ...]:
    if isinstance(raw_groups, str):
        groups = [
            group.strip()
            for group in raw_groups.split(",")
            if group.strip()
        ]

        return tuple(sorted(set(groups)))

    if isinstance(raw_groups, list):
        groups = [
            str(group).strip()
            for group in raw_groups
            if str(group).strip()
        ]

        return tuple(sorted(set(groups)))

    return ()


class AuthenticationService:
    """
    Issues development tokens and verifies trusted
    access tokens.
    """

    def __init__(
        self,
        settings: AuthenticationSettings,
    ) -> None:
        self.settings = settings
        self.mode = settings.mode

        self._jwk_client: PyJWKClient | None = None

        if settings.mode == "cognito":
            self._jwk_client = PyJWKClient(
                settings.cognito_jwks_url
            )

    def issue_development_token(
        self,
        *,
        profile_name: str,
        password: str,
    ) -> IssuedAccessToken:
        """
        Issues a local token.

        This method is unavailable in Cognito mode.
        """

        if self.settings.mode != "local_jwt":
            raise AuthenticationError(
                "Development login is disabled."
            )

        expected_password = (
            self.settings.development_password
        )

        if expected_password is None:
            raise AuthenticationError(
                "Development login is unavailable."
            )

        if not secrets.compare_digest(
            password,
            expected_password,
        ):
            raise AuthenticationError(
                "Invalid development credentials."
            )

        profile = DEVELOPMENT_PROFILES.get(
            profile_name
        )

        if profile is None:
            raise AuthenticationError(
                "Unknown development profile."
            )

        now = datetime.now(
            timezone.utc
        )

        expiration = now + timedelta(
            minutes=(
                self.settings
                .local_expiration_minutes
            )
        )

        payload = {
            "sub": profile.subject,
            "username": profile.username,
            "role": profile.role,
            "region": profile.region,
            "clearance_rank":
                profile.clearance_rank,
            "groups": list(profile.groups),
            "iss": self.settings.local_issuer,
            "aud": self.settings.local_audience,
            "token_use": "access",
            "scope": "policy.read policy.chat",
            "iat": now,
            "exp": expiration,
            "jti": str(uuid4()),
        }

        secret = self.settings.local_secret

        if secret is None:
            raise RuntimeError(
                "Local JWT secret is unavailable."
            )

        token = jwt.encode(
            payload,
            secret,
            algorithm=LOCAL_ALGORITHM,
        )

        principal = AuthenticatedPrincipal(
            subject=profile.subject,
            username=profile.username,
            role=profile.role,
            region=profile.region,
            clearance_rank=(
                profile.clearance_rank
            ),
            groups=profile.groups,
        )

        return IssuedAccessToken(
            access_token=token,
            expires_in_seconds=(
                self.settings
                .local_expiration_minutes
                * 60
            ),
            principal=principal,
        )

    def verify_access_token(
        self,
        token: str,
    ) -> AuthenticatedPrincipal:
        """
        Verifies one bearer access token.
        """

        if not token.strip():
            raise AuthenticationError(
                "Access token cannot be empty."
            )

        try:
            if self.settings.mode == "local_jwt":
                return self._verify_local_token(
                    token
                )

            return self._verify_cognito_token(
                token
            )

        except AuthenticationError:
            raise

        except jwt.ExpiredSignatureError as error:
            raise AuthenticationError(
                "The access token has expired."
            ) from error

        except jwt.InvalidTokenError as error:
            raise AuthenticationError(
                "The access token is invalid."
            ) from error

        except Exception as error:
            raise AuthenticationError(
                "The access token could not be "
                "verified."
            ) from error

    def _verify_local_token(
        self,
        token: str,
    ) -> AuthenticatedPrincipal:
        secret = self.settings.local_secret

        if secret is None:
            raise RuntimeError(
                "Local JWT secret is unavailable."
            )

        payload = jwt.decode(
            token,
            secret,
            algorithms=[LOCAL_ALGORITHM],
            issuer=self.settings.local_issuer,
            audience=self.settings.local_audience,
            options={
                "require": [
                    "sub",
                    "iss",
                    "aud",
                    "iat",
                    "exp",
                    "token_use",
                ]
            },
        )

        if payload.get("token_use") != "access":
            raise AuthenticationError(
                "The token is not an access token."
            )

        role = str(
            payload.get(
                "role",
                "",
            )
        ).strip()

        region = str(
            payload.get(
                "region",
                "",
            )
        ).strip().upper()

        raw_clearance = payload.get(
            "clearance_rank"
        )

        try:
            clearance_rank = int(
                raw_clearance
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise AuthenticationError(
                "Token clearance is invalid."
            ) from error

        if not role or not region:
            raise AuthenticationError(
                "Required access claims are missing."
            )

        return AuthenticatedPrincipal(
            subject=str(payload["sub"]),
            username=str(
                payload.get(
                    "username",
                    payload["sub"],
                )
            ),
            role=role,
            region=region,
            clearance_rank=clearance_rank,
            groups=normalize_groups(
                payload.get("groups")
            ),
        )

    def _verify_cognito_token(
        self,
        token: str,
    ) -> AuthenticatedPrincipal:
        if self._jwk_client is None:
            raise RuntimeError(
                "Cognito JWKS client is unavailable."
            )

        signing_key = (
            self._jwk_client
            .get_signing_key_from_jwt(token)
        )

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=[COGNITO_ALGORITHM],
            issuer=self.settings.cognito_issuer,
            options={
                "verify_aud": False,
                "require": [
                    "sub",
                    "iss",
                    "iat",
                    "exp",
                    "token_use",
                    "client_id",
                ],
            },
        )

        if payload.get("token_use") != "access":
            raise AuthenticationError(
                "Only Cognito access tokens are "
                "accepted."
            )

        if (
            payload.get("client_id")
            != self.settings.cognito_app_client_id
        ):
            raise AuthenticationError(
                "The token was issued for a different "
                "application client."
            )

        required_scope = (
            self.settings.cognito_required_scope
        )

        if required_scope:
            token_scopes = set(
                str(
                    payload.get(
                        "scope",
                        "",
                    )
                ).split()
            )

            if required_scope not in token_scopes:
                raise AuthenticationError(
                    "The token does not contain the "
                    "required API scope."
                )

        return self._build_cognito_principal(
            payload
        )

    @staticmethod
    def _build_cognito_principal(
        payload: dict[str, Any],
    ) -> AuthenticatedPrincipal:
        groups = normalize_groups(
            payload.get("cognito:groups")
        )

        matching_policies = [
            (
                group,
                COGNITO_GROUP_POLICIES[group],
            )
            for group in groups
            if group in COGNITO_GROUP_POLICIES
        ]

        if not matching_policies:
            raise AuthenticationError(
                "The user does not belong to an "
                "authorized Cognito group."
            )

        selected_group, selected_policy = max(
            matching_policies,
            key=lambda item: (
                item[1].clearance_rank,
                item[0],
            ),
        )

        region = str(
            payload.get(
                "custom:region",
                payload.get(
                    "region",
                    "",
                ),
            )
        ).strip().upper()

        if not region:
            raise AuthenticationError(
                "The token does not contain an "
                "authorized region claim."
            )

        username = str(
            payload.get(
                "username",
                payload.get(
                    "cognito:username",
                    payload["sub"],
                ),
            )
        )

        return AuthenticatedPrincipal(
            subject=str(payload["sub"]),
            username=username,
            role=selected_policy.role,
            region=region,
            clearance_rank=(
                selected_policy.clearance_rank
            ),
            groups=groups or (
                selected_group,
            ),
        )