from __future__ import annotations

from contextlib import contextmanager

import psycopg2

from config.settings import DatabaseConfig


class PostgresDatabase:
    """Connection handling shared by the Postgres-backed adapters."""

    def __init__(self, db_config: DatabaseConfig) -> None:
        self.db_config = db_config

    def connect(self):
        return psycopg2.connect(**self.db_config.to_dict())

    @contextmanager
    def cursor(self, commit: bool = False, **kwargs):
        conn = self.connect()
        try:
            yield conn.cursor(**kwargs)
            if commit:
                conn.commit()
        finally:
            conn.close()
