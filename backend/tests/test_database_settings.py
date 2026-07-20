import pytest

from app.database.settings import (
    get_database_settings,
)


def clear_database_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [
        "DATABASE_URL",
        "DATABASE_ECHO",
        "DATABASE_AUTO_CREATE_SCHEMA",
        "DATABASE_POOL_SIZE",
        "DATABASE_MAX_OVERFLOW",
        "DATABASE_POOL_TIMEOUT_SECONDS",
        "DATABASE_POOL_RECYCLE_SECONDS",
        "DATABASE_CONNECT_TIMEOUT_SECONDS",
        "DATABASE_STATEMENT_TIMEOUT_MS",
        "DATABASE_APPLICATION_NAME",
    ]

    for name in names:
        monkeypatch.delenv(
            name,
            raising=False,
        )


def test_sqlite_defaults_enable_schema_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(
        monkeypatch
    )

    settings = get_database_settings()

    assert settings.is_sqlite is True
    assert settings.is_postgresql is False
    assert settings.auto_create_schema is True


def test_postgresql_defaults_require_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(
        monkeypatch
    )

    monkeypatch.setenv(
        "DATABASE_URL",
        (
            "postgresql+psycopg://"
            "policy_copilot:secret@"
            "127.0.0.1:5432/policy_copilot"
        ),
    )

    settings = get_database_settings()

    assert settings.is_postgresql is True
    assert settings.is_sqlite is False
    assert settings.auto_create_schema is False


def test_plain_postgresql_url_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(
        monkeypatch
    )

    monkeypatch.setenv(
        "DATABASE_URL",
        (
            "postgresql://"
            "policy_copilot:secret@"
            "127.0.0.1:5432/policy_copilot"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"postgresql\+psycopg",
    ):
        get_database_settings()


def test_invalid_pool_size_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_database_environment(
        monkeypatch
    )

    monkeypatch.setenv(
        "DATABASE_POOL_SIZE",
        "zero",
    )

    with pytest.raises(
        RuntimeError,
        match="must contain an integer",
    ):
        get_database_settings()
