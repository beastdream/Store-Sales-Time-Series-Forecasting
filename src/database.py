"""Create and test a PostgreSQL SQLAlchemy connection from environment variables."""

import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from sqlalchemy.exc import SQLAlchemyError


LOGGER = logging.getLogger(__name__)
REQUIRED_DB_VARIABLES = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)

load_dotenv()


def get_engine() -> Engine:
    """Return a PostgreSQL engine configured entirely from environment variables."""
    settings = {name: os.getenv(name) for name in REQUIRED_DB_VARIABLES}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(
            "Missing required database environment variables: " + ", ".join(missing)
        )

    try:
        port = int(settings["DB_PORT"])
    except (TypeError, ValueError) as exc:
        raise ValueError("DB_PORT must be a valid integer") from exc

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=settings["DB_USER"],
        password=settings["DB_PASSWORD"],
        host=settings["DB_HOST"],
        port=port,
        database=settings["DB_NAME"],
    )
    try:
        return create_engine(url, pool_pre_ping=True)
    except (ImportError, ModuleNotFoundError, SQLAlchemyError) as exc:
        raise RuntimeError(
            "Unable to create the database engine; verify the installed driver and settings"
        ) from None


def test_connection() -> bool:
    """Return whether a simple database connection and query succeed."""
    try:
        engine = get_engine()
    except (ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return False

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        LOGGER.error(
            "Database connection failed; verify server availability and credentials"
        )
        return False
    finally:
        engine.dispose()
