"""The composition root: the sole module that reads configuration, constructs
adapters, and wires concrete implementations together. Both entry points —
the API and the CLI — assemble their object graph here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from application.build_knowledge_graph import BuildKnowledgeGraphUseCase
from application.ingest_transcript import IngestTranscriptUseCase
from application.query_transcripts import QueryTranscriptsUseCase
from application.transcript_catalog import TranscriptCatalog
from config.settings import Config
from domain.chunking.factory import make_chunker
from domain.ports import KnowledgeBase
from infrastructure.audio.nemo_diarizer import NemoDiarizer
from infrastructure.audio.whisper_transcriber import WhisperTranscriber
from infrastructure.llm.lightrag_functions import make_embed_func, make_llm_func, make_rerank_func
from infrastructure.llm.openai_completer import OpenAITextCompleter
from infrastructure.rag.lightrag_knowledge_base import LightRAGKnowledgeBase
from infrastructure.storage.minio_archive import MinIOTranscriptArchive
from infrastructure.storage.postgres_metadata_store import PostgresMetadataStore
from infrastructure.storage.postgres_speaker_directory import PostgresSpeakerDirectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Container:
    """The wired application: use cases plus the resources they share."""

    config: Config
    knowledge_base: KnowledgeBase
    ingest: IngestTranscriptUseCase
    query: QueryTranscriptsUseCase
    build_graph: BuildKnowledgeGraphUseCase
    catalog: TranscriptCatalog
    _schema_owners: tuple

    def ensure_schema(self) -> None:
        """Create the tables the adapters need. Safe to call repeatedly."""
        for owner in self._schema_owners:
            owner.setup()

    async def startup(self) -> None:
        self.ensure_schema()
        await self.knowledge_base.initialize()
        logger.info("Application ready (chunker=%s).", self.config.chunker.chunker_type)

    async def shutdown(self) -> None:
        await self.knowledge_base.finalize()


def build_container(config: Config | None = None) -> Container:
    config = config or Config.from_env()

    archive = MinIOTranscriptArchive(config.minio)
    metadata_store = PostgresMetadataStore(config.database)
    speakers = PostgresSpeakerDirectory(config.database)

    completer = OpenAITextCompleter(config.openai_api_key, config.model)
    chunker = make_chunker(
        config.chunker.chunker_type,
        completer=completer,
        block_window_size=config.chunker.block_window_size,
        contextual_window_size=config.chunker.contextual_window_size,
    )

    knowledge_base = LightRAGKnowledgeBase(
        config=config,
        chunker=chunker,
        llm_func=make_llm_func(config.model),
        embed_func=make_embed_func(config.model),
        rerank_func=make_rerank_func(config.reranker),
    )

    diarizer = NemoDiarizer(
        config.diarization.nemo_config_url, config.diarization.segment_minutes
    )
    transcriber = WhisperTranscriber(
        config.transcription.whisper_model,
        config.language.code,
        config.diarization.segment_minutes,
    )

    return Container(
        config=config,
        knowledge_base=knowledge_base,
        ingest=IngestTranscriptUseCase(
            archive=archive,
            speakers=speakers,
            metadata_store=metadata_store,
            knowledge_base=knowledge_base,
            diarizer=diarizer,
            transcriber=transcriber,
            completer=completer,
            speaker_similarity_threshold=config.diarization.speaker_similarity_threshold,
            language=config.language.code,
        ),
        query=QueryTranscriptsUseCase(
            knowledge_base=knowledge_base,
            archive=archive,
            metadata_store=metadata_store,
            completer=completer,
        ),
        build_graph=BuildKnowledgeGraphUseCase(
            archive=archive,
            metadata_store=metadata_store,
            knowledge_base=knowledge_base,
        ),
        catalog=TranscriptCatalog(archive),
        _schema_owners=(metadata_store, speakers),
    )
