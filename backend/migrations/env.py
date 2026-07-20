from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.engine import make_url

from app.database.models import Base
from app.database.settings import (
    get_database_settings,
)


load_dotenv()


config = context.config

if config.config_file_name is not None:
    fileConfig(
        config.config_file_name
    )


target_metadata = Base.metadata


def configure_database_url() -> str:
    """
    Loads DATABASE_URL through the application's
    validated settings and injects it into Alembic.

    Percent signs are escaped because Alembic uses
    ConfigParser interpolation.
    """

    settings = get_database_settings()

    rendered_url = make_url(
        settings.url
    ).render_as_string(
        hide_password=False
    )

    escaped_url = rendered_url.replace(
        "%",
        "%%",
    )

    config.set_main_option(
        "sqlalchemy.url",
        escaped_url,
    )

    return rendered_url


database_url = configure_database_url()


def run_migrations_offline() -> None:
    """
    Runs migrations without opening a database
    connection.
    """

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Runs migrations using a live SQLAlchemy
    connection.
    """

    configuration = (
        config.get_section(
            config.config_ini_section,
        )
        or {}
    )

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=(
                connection.dialect.name
                == "sqlite"
            ),
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()

else:
    run_migrations_online()
