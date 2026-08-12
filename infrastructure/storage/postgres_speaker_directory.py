from __future__ import annotations

import logging

import numpy as np
from psycopg2.extras import RealDictCursor

from config.settings import DatabaseConfig
from domain.speaker import SpeakerMatch
from infrastructure.storage.postgres import PostgresDatabase

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 192


class PostgresSpeakerDirectory:
    """``SpeakerDirectory`` backed by pgvector.

    Re-hearing a known speaker averages the stored embedding with the new one,
    so a voice profile drifts towards the speaker's typical sound over time.
    """

    def __init__(self, db_config: DatabaseConfig) -> None:
        self._db = PostgresDatabase(db_config)

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        return embedding / (np.linalg.norm(embedding) + 1e-9)

    def setup(self) -> None:
        try:
            with self._db.cursor(commit=True) as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS speakers (
                        id         SERIAL PRIMARY KEY,
                        name       TEXT NOT NULL UNIQUE,
                        notes      TEXT,
                        embedding  vector({EMBEDDING_DIM}),
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

    def remember(
        self,
        name: str,
        embedding: np.ndarray,
        notes: str = "",
        files: int = 0,
        turns: int = 0,
    ) -> None:
        norm_emb = self._normalize(embedding)
        try:
            with self._db.cursor(commit=True) as cur:
                cur.execute("SELECT id, embedding FROM speakers WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    avg_emb = self._normalize((row[1].to_numpy() + norm_emb) / 2)
                    cur.execute(
                        """UPDATE speakers
                           SET embedding  = %s,
                               no_files   = no_files + %s,
                               no_turns   = no_turns + %s,
                               updated_at = NOW()
                           WHERE id = %s""",
                        (avg_emb.tolist(), files, turns, row[0]),
                    )
                else:
                    cur.execute(
                        """INSERT INTO speakers (name, notes, embedding, no_files, no_turns)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (name, notes, norm_emb.tolist(), files, turns),
                    )
            logger.info("Upserted speaker: %s", name)
        except Exception:
            logger.exception("Failed to upsert speaker: %s", name)

    def match(self, embedding: np.ndarray, threshold: float) -> SpeakerMatch | None:
        norm_emb = self._normalize(embedding)
        try:
            with self._db.cursor(cursor_factory=RealDictCursor) as cur:
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
                    return SpeakerMatch(name=row["name"], similarity=row["similarity"])
                return None
        except Exception:
            logger.exception("Failed to match speaker.")
            return None
