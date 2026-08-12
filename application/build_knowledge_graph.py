"""Re-indexing every transcript already in the archive into the knowledge graph.
LightRAG skips any chunk it has already indexed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from application.progress import LoggingProgress, ProgressReporter
from domain.document import transcript_key
from domain.ports import KnowledgeBase, TranscriptArchive, TranscriptMetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildReport:
    total: int
    successful: int
    failed: int

    @property
    def succeeded(self) -> bool:
        return self.total > 0 and self.failed == 0


class BuildKnowledgeGraphUseCase:

    def __init__(
        self,
        archive: TranscriptArchive,
        metadata_store: TranscriptMetadataStore,
        knowledge_base: KnowledgeBase,
    ) -> None:
        self.archive = archive
        self.metadata_store = metadata_store
        self.knowledge_base = knowledge_base

    async def build(self, progress: ProgressReporter | None = None) -> BuildReport:
        progress = progress or LoggingProgress()

        stems = self.archive.list_stems()
        if not stems:
            logger.warning("No transcripts found in the archive.")
            return BuildReport(total=0, successful=0, failed=0)

        successful = failed = 0
        for i, stem in enumerate(stems, 1):
            progress.report(f"[{i}/{len(stems)}] Processing: {stem}")
            if await self._ingest_existing(stem):
                successful += 1
            else:
                failed += 1

        logger.info("Done. Successful: %d, Failed: %d", successful, failed)
        return BuildReport(total=len(stems), successful=successful, failed=failed)

    async def _ingest_existing(self, stem: str) -> bool:
        try:
            transcript, metadata_json = self.archive.load_transcript(stem)
            file_path = transcript_key(stem)
            if not await self.knowledge_base.insert_document(
                transcript, stem, file_path, metadata_json
            ):
                logger.warning("Failed to insert %s", stem)
                return False

            metadata = json.loads(metadata_json) if metadata_json else {}
            self.metadata_store.save(stem, file_path, metadata)
            logger.info("Inserted doc_id=%s", stem)
            return True
        except Exception:
            logger.exception("Error processing %s", stem)
            return False
