"""Tests for the canonical ``src.database`` module."""

from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

import src.database as database


def test_src_database_resolves_to_module_file() -> None:
    expected = Path(__file__).resolve().parents[1] / "src" / "database.py"

    assert Path(database.__file__).resolve() == expected.resolve()


def test_get_engine_uses_environment_without_connecting(monkeypatch) -> None:
    settings = {
        "DB_HOST": "db.example.test",
        "DB_PORT": "5432",
        "DB_NAME": "store_sales",
        "DB_USER": "analyst",
        "DB_PASSWORD": "not-a-real-password",
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    sentinel_engine = object()
    calls = []

    def fake_create_engine(url, **kwargs):
        calls.append((url, kwargs))
        return sentinel_engine

    monkeypatch.setattr(database, "create_engine", fake_create_engine)

    assert database.get_engine() is sentinel_engine
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url.drivername == "postgresql+psycopg2"
    assert url.host == settings["DB_HOST"]
    assert url.port == 5432
    assert url.database == settings["DB_NAME"]
    assert kwargs == {"pool_pre_ping": True}


class _ConnectionContext:
    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class _FakeEngine:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.disposed = False

    def connect(self):
        return _ConnectionContext(self.connection)

    def dispose(self) -> None:
        self.disposed = True


def test_connection_uses_mock_engine(monkeypatch) -> None:
    executed = []

    class FakeConnection:
        def execute(self, statement) -> None:
            executed.append(str(statement))

    engine = _FakeEngine(FakeConnection())
    monkeypatch.setattr(database, "get_engine", lambda: engine)

    assert database.test_connection() is True
    assert executed == ["SELECT 1"]
    assert engine.disposed is True


def test_connection_returns_false_for_sqlalchemy_error(monkeypatch) -> None:
    class FailingEngine(_FakeEngine):
        def connect(self):
            raise SQLAlchemyError("database unavailable")

    engine = FailingEngine(connection=None)
    monkeypatch.setattr(database, "get_engine", lambda: engine)

    assert database.test_connection() is False
    assert engine.disposed is True
