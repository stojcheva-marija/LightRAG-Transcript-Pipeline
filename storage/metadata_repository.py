from __future__ import annotations

import json
import logging
from contextlib import contextmanager

import psycopg2

from config.settings import DatabaseConfig

logger = logging.getLogger(__name__)


class MetadataRepository:
    def __init__(self, db_config: DatabaseConfig) -> None:
        self.db_config = db_config

    def _connect(self):
        return psycopg2.connect(**self.db_config.to_dict())

    @contextmanager
    def _cursor(self, commit: bool = False, **kwargs):
        conn = self._connect()
        try:
            yield conn.cursor(**kwargs)
            if commit:
                conn.commit()
        finally:
            conn.close()

    def setup(self) -> None:
        try:
            with self._cursor(commit=True) as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_metadata (
                        doc_id      VARCHAR PRIMARY KEY,
                        file_path   TEXT,
                        metadata    JSONB,
                        created_at  TIMESTAMP DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_metadata_date
                        ON document_metadata ((metadata->>'date'));
                    CREATE INDEX IF NOT EXISTS idx_metadata_speakers
                        ON document_metadata USING GIN ((metadata->'speakers'));
                    CREATE INDEX IF NOT EXISTS idx_metadata_show
                        ON document_metadata ((metadata->>'show'));
                """)
            logger.info("document_metadata table is ready.")
        except Exception:
            logger.exception("Failed to setup document_metadata table.")

    def save(self, doc_id: str, file_path: str, metadata: dict) -> None:
        try:
            with self._cursor(commit=True) as cur:
                cur.execute(
                    """
                    INSERT INTO document_metadata (doc_id, file_path, metadata)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                        SET metadata  = EXCLUDED.metadata,
                            file_path = EXCLUDED.file_path
                    """,
                    (doc_id, file_path, json.dumps(metadata)),
                )
            logger.info("Saved metadata for doc_id=%s", doc_id)
        except Exception:
            logger.exception("Failed to save metadata for doc_id=%s", doc_id)

    def get_file_path(self, doc_id: str) -> str | None:
        try:
            with self._cursor() as cur:
                cur.execute(
                    "SELECT file_path FROM document_metadata WHERE doc_id = %s", (doc_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            logger.exception("Failed to fetch file_path for doc_id=%s", doc_id)
            return None

    def get_metadata(self, doc_id: str) -> dict | None:
        try:
            with self._cursor() as cur:
                cur.execute(
                    "SELECT metadata FROM document_metadata WHERE doc_id = %s", (doc_id,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception:
            logger.exception("Failed to fetch metadata for doc_id=%s", doc_id)
            return None

    def get_all_speakers(self) -> list[str]:
        try:
            with self._cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT jsonb_array_elements_text(metadata->'speakers')
                    FROM document_metadata
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch speakers.")
            return []
