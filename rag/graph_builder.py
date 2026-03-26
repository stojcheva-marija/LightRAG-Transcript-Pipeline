from __future__ import annotations

import json
import logging

from config.settings import Config
from rag.conversational_rag import ConversationalRAG
from storage.minio_client import MinIOClient
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, config: Config, rag: ConversationalRAG, minio: MinIOClient) -> None:
        self.rag = rag
        self.minio = minio
        self.metadata_repo = MetadataRepository(config.database)

    @staticmethod
    def make_doc_id(stem: str) -> str:
        return f"transcript_{stem}_{hash(stem) % 10000}"

    async def build(self) -> bool:
        self.metadata_repo.setup()

        stems = self.minio.list_transcripts()
        if not stems:
            logger.warning("No transcripts found in MinIO bucket.")
            return False

        if not await self.rag.initialize():
            return False

        successful, failed = 0, 0
        for i, stem in enumerate(stems, 1):
            logger.info("[%d/%d] Processing: %s", i, len(stems), stem)
            try:
                transcript, metadata_str = self.minio.load_transcript_pair(stem)
                doc_id = self.make_doc_id(stem)
                txt_key = f"transcripts/{stem}/{stem}.txt"
                inserted = await self.rag.insert_document(transcript, doc_id, txt_key, metadata_str)
                if inserted:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    self.metadata_repo.save(doc_id, txt_key, metadata)
                    successful += 1
                    logger.info("Inserted doc_id=%s", doc_id)
                else:
                    failed += 1
                    logger.warning("Failed to insert %s", stem)
            except Exception:
                failed += 1
                logger.exception("Error processing %s", stem)

        logger.info("Done. Successful: %d, Failed: %d", successful, failed)
        return failed == 0

    async def cleanup(self) -> None:
        await self.rag.finalize()
