from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import (
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import (
    Engine,
    make_url,
)
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

from app.database.models import Base
from app.database.settings import (
    DatabaseSettings,
)


EXPECTED_APPLICATION_TABLES = {
    "conversations",
    "conversation_messages",
    "message_feedback",
}


class DatabaseService:
    """
    Owns the SQLAlchemy engine and creates database
    sessions for API requests.
    """

    def __init__(
        self,
        settings: DatabaseSettings,
    ) -> None:
        self.settings = settings

        self._prepare_sqlite_directory()

        engine_arguments = (
            self._build_engine_arguments()
        )

        self.engine: Engine = create_engine(
            settings.url,
            **engine_arguments,
        )

        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def _build_engine_arguments(
        self,
    ) -> dict[str, Any]:
        """
        Returns database-specific SQLAlchemy engine
        options.
        """

        engine_arguments: dict[str, Any] = {
            "echo": self.settings.echo,
            "pool_pre_ping": True,
        }

        if self.settings.is_sqlite:
            engine_arguments["connect_args"] = {
                "check_same_thread": False,
            }

            return engine_arguments

        if self.settings.is_postgresql:
            engine_arguments.update(
                {
                    "pool_size": (
                        self.settings.pool_size
                    ),
                    "max_overflow": (
                        self.settings.max_overflow
                    ),
                    "pool_timeout": (
                        self.settings
                        .pool_timeout_seconds
                    ),
                    "pool_recycle": (
                        self.settings
                        .pool_recycle_seconds
                    ),
                    "connect_args": {
                        "connect_timeout": (
                            self.settings
                            .connect_timeout_seconds
                        ),
                        "application_name": (
                            self.settings
                            .application_name
                        ),
                        "options": (
                            "-c statement_timeout="
                            f"{self.settings.statement_timeout_ms}"
                        ),
                    },
                }
            )

        return engine_arguments

    def _prepare_sqlite_directory(
        self,
    ) -> None:
        """
        Creates the parent directory for a local SQLite
        database before SQLAlchemy opens the file.
        """

        database_url = make_url(
            self.settings.url
        )

        if not database_url.drivername.startswith(
            "sqlite"
        ):
            return

        database_path = database_url.database

        if database_path in {
            None,
            "",
            ":memory:",
        }:
            return

        Path(database_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def check_connection(self) -> None:
        """
        Opens one connection and performs a minimal
        database liveness check.
        """

        with self.engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

    def verify_schema(self) -> None:
        """
        Confirms that all application tables exist.

        This prevents the API from starting against an
        unmigrated PostgreSQL database.
        """

        inspector = inspect(
            self.engine
        )

        existing_tables = set(
            inspector.get_table_names()
        )

        missing_tables = (
            EXPECTED_APPLICATION_TABLES
            - existing_tables
        )

        if missing_tables:
            missing_list = ", ".join(
                sorted(missing_tables)
            )

            raise RuntimeError(
                "The database schema is not current. "
                "Missing tables: "
                f"{missing_list}. Run "
                "'alembic upgrade head' from the "
                "backend folder."
            )

    def initialize(self) -> None:
        """
        Initializes database access.

        SQLite tests may create tables directly when
        DATABASE_AUTO_CREATE_SCHEMA=true.

        PostgreSQL should normally set
        DATABASE_AUTO_CREATE_SCHEMA=false and use
        Alembic migrations.
        """

        self.check_connection()

        if self.settings.auto_create_schema:
            Base.metadata.create_all(
                self.engine
            )

        self.verify_schema()

    @contextmanager
    def session(
        self,
    ) -> Iterator[Session]:
        """
        Provides one SQLAlchemy session.

        The transaction is rolled back automatically
        when an exception occurs.
        """

        with self.session_factory() as session:
            try:
                yield session

            except Exception:
                session.rollback()
                raise

    def dispose(self) -> None:
        """
        Releases database connection-pool resources
        during FastAPI shutdown.
        """

        self.engine.dispose()
