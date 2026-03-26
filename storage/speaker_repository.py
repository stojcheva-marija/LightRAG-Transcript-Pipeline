from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from config.settings import DatabaseConfig

logger = logging.getLogger(__name__)


class SpeakerRepository:
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

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        return embedding / (np.linalg.norm(embedding) + 1e-9)

    def setup(self) -> None:
        try:
            with self._cursor(commit=True) as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS speakers (
                        id         SERIAL PRIMARY KEY,
                        name       TEXT NOT NULL UNIQUE,
                        notes      TEXT,
                        embedding  vector(192),
                        no_files   INTEGER DEFAULT 0,
                        no_turns   INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_speakers_embedding
                        ON speakers USING hnsw (embedding vector_cosine_ops);
                """)
            logger.info("Speakers table is ready.")
        except Exception:
            logger.exception("Failed to setup speakers table.")

    def upsert_speaker(
        self,
        name: str,
        embedding: np.ndarray,
        notes: str = "",
        no_files: int = 0,
        no_turns: int = 0,
    ) -> None:
        norm_emb = self._normalize(embedding)
        try:
            with self._cursor(commit=True) as cur:
                cur.execute("SELECT id, embedding FROM speakers WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    avg_emb = self._normalize((np.array(row[1]) + norm_emb) / 2)
                    cur.execute(
                        """UPDATE speakers
                           SET embedding  = %s,
                               no_files   = no_files + %s,
                               no_turns   = no_turns + %s,
                               updated_at = NOW()
                           WHERE id = %s""",
                        (avg_emb.tolist(), no_files, no_turns, row[0]),
                    )
                else:
                    cur.execute(
                        """INSERT INTO speakers (name, notes, embedding, no_files, no_turns)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (name, notes, norm_emb.tolist(), no_files, no_turns),
                    )
            logger.info("Upserted speaker: %s", name)
        except Exception:
            logger.exception("Failed to upsert speaker: %s", name)

    def match_speaker(self, query_embedding: np.ndarray, threshold: float) -> dict | None:
        norm_emb = self._normalize(query_embedding)
        try:
            with self._cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT id, name,
                              1 - (embedding <=> %s::vector) AS similarity
                       FROM speakers
                       ORDER BY embedding <=> %s::vector
                       LIMIT 1""",
                    (norm_emb.tolist(), norm_emb.tolist()),
                )
                row = cur.fetchone()
                if row and row["similarity"] >= threshold:
                    return {"id": row["id"], "name": row["name"], "similarity": row["similarity"]}
                return None
        except Exception:
            logger.exception("Failed to match speaker.")
            return None

    def get_all_speakers(self) -> list[dict]:
        try:
            with self._cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, name, notes, no_files, no_turns FROM speakers ORDER BY name"
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception:
            logger.exception("Failed to fetch all speakers.")
            return []
