from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    create_engine,
    inspect,
)


EXPECTED_TABLES = {
    "alembic_version",
    "conversations",
    "conversation_messages",
    "message_feedback",
}


def build_alembic_config() -> Config:
    backend_root = Path(
        __file__
    ).resolve().parents[1]

    config = Config(
        str(
            backend_root
            / "alembic.ini"
        )
    )

    config.set_main_option(
        "script_location",
        str(
            backend_root
            / "migrations"
        ),
    )

    return config


def test_upgrade_and_downgrade_migrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = (
        tmp_path
        / "migration_test.db"
    )

    database_url = (
        f"sqlite:///{database_path}"
    )

    monkeypatch.setenv(
        "DATABASE_URL",
        database_url,
    )

    monkeypatch.setenv(
        "DATABASE_AUTO_CREATE_SCHEMA",
        "false",
    )

    config = build_alembic_config()

    command.upgrade(
        config,
        "head",
    )

    engine = create_engine(
        database_url
    )

    try:
        tables_after_upgrade = set(
            inspect(
                engine
            ).get_table_names()
        )

        assert (
            EXPECTED_TABLES
            <= tables_after_upgrade
        )

    finally:
        engine.dispose()

    command.downgrade(
        config,
        "base",
    )

    engine = create_engine(
        database_url
    )

    try:
        tables_after_downgrade = set(
            inspect(
                engine
            ).get_table_names()
        )

        assert (
            "conversations"
            not in tables_after_downgrade
        )

        assert (
            "conversation_messages"
            not in tables_after_downgrade
        )

        assert (
            "message_feedback"
            not in tables_after_downgrade
        )

    finally:
        engine.dispose()
