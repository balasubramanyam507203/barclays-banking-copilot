import os
from dataclasses import dataclass

from sqlalchemy.engine import make_url


@dataclass(frozen=True)
class DatabaseSettings:
    """
    SQL database configuration.

    SQLite remains supported for isolated tests and
    lightweight local development.

    PostgreSQL is the normal application database after
    Step 24.
    """

    url: str
    echo: bool
    auto_create_schema: bool

    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout_seconds: int = 30
    pool_recycle_seconds: int = 1_800

    connect_timeout_seconds: int = 10
    statement_timeout_ms: int = 30_000

    application_name: str = (
        "enterprise-banking-policy-copilot"
    )

    @property
    def driver_name(self) -> str:
        return make_url(
            self.url
        ).drivername

    @property
    def is_sqlite(self) -> bool:
        return self.driver_name.startswith(
            "sqlite"
        )

    @property
    def is_postgresql(self) -> bool:
        return self.driver_name.startswith(
            "postgresql"
        )


def get_boolean_environment_variable(
    name: str,
    default: bool,
) -> bool:
    """
    Reads a Boolean environment variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in {
        "true",
        "1",
        "yes",
        "on",
    }:
        return True

    if normalized_value in {
        "false",
        "0",
        "no",
        "off",
    }:
        return False

    raise RuntimeError(
        f"Environment variable '{name}' must "
        "contain true or false."
    )


def get_integer_environment_variable(
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    """
    Reads and validates an integer environment
    variable.
    """

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(
            raw_value.strip()
        )

    except ValueError as error:
        raise RuntimeError(
            f"Environment variable '{name}' must "
            "contain an integer."
        ) from error

    if value < minimum:
        raise RuntimeError(
            f"Environment variable '{name}' must be "
            f"greater than or equal to {minimum}."
        )

    return value


def normalize_database_url(
    database_url: str,
) -> str:
    """
    Validates the SQLAlchemy database URL.

    PostgreSQL URLs must explicitly use the Psycopg 3
    driver so local and production environments use the
    same supported driver.
    """

    cleaned_url = database_url.strip()

    if not cleaned_url:
        raise RuntimeError(
            "DATABASE_URL cannot be empty."
        )

    parsed_url = make_url(
        cleaned_url
    )

    if (
        parsed_url.drivername == "postgresql"
        or parsed_url.drivername == "postgres"
    ):
        raise RuntimeError(
            "DATABASE_URL must use the Psycopg 3 "
            "SQLAlchemy driver format: "
            "'postgresql+psycopg://...'."
        )

    if not (
        parsed_url.drivername.startswith(
            "sqlite"
        )
        or parsed_url.drivername.startswith(
            "postgresql+psycopg"
        )
    ):
        raise RuntimeError(
            "DATABASE_URL must use SQLite or "
            "PostgreSQL with Psycopg 3."
        )

    return cleaned_url


def get_database_settings() -> DatabaseSettings:
    """
    Builds database settings from environment
    variables.
    """

    database_url = normalize_database_url(
        os.getenv(
            "DATABASE_URL",
            "sqlite:///local_data/"
            "policy_copilot.db",
        )
    )

    parsed_url = make_url(
        database_url
    )

    is_sqlite = (
        parsed_url.drivername.startswith(
            "sqlite"
        )
    )

    return DatabaseSettings(
        url=database_url,
        echo=get_boolean_environment_variable(
            "DATABASE_ECHO",
            False,
        ),
        auto_create_schema=(
            get_boolean_environment_variable(
                "DATABASE_AUTO_CREATE_SCHEMA",
                is_sqlite,
            )
        ),
        pool_size=(
            get_integer_environment_variable(
                "DATABASE_POOL_SIZE",
                5,
                minimum=1,
            )
        ),
        max_overflow=(
            get_integer_environment_variable(
                "DATABASE_MAX_OVERFLOW",
                10,
                minimum=0,
            )
        ),
        pool_timeout_seconds=(
            get_integer_environment_variable(
                "DATABASE_POOL_TIMEOUT_SECONDS",
                30,
                minimum=1,
            )
        ),
        pool_recycle_seconds=(
            get_integer_environment_variable(
                "DATABASE_POOL_RECYCLE_SECONDS",
                1_800,
                minimum=1,
            )
        ),
        connect_timeout_seconds=(
            get_integer_environment_variable(
                "DATABASE_CONNECT_TIMEOUT_SECONDS",
                10,
                minimum=1,
            )
        ),
        statement_timeout_ms=(
            get_integer_environment_variable(
                "DATABASE_STATEMENT_TIMEOUT_MS",
                30_000,
                minimum=1,
            )
        ),
        application_name=(
            os.getenv(
                "DATABASE_APPLICATION_NAME",
                "enterprise-banking-policy-copilot",
            ).strip()
            or "enterprise-banking-policy-copilot"
        ),
    )
