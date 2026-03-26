from __future__ import annotations

import asyncio
import logging

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from chunkers.base import BaseChunker
from config.settings import Config
from ingestion.metadata_builder import build_metadata_header
from prompts.mk.entity_extraction import build_entity_extraction_prompts, ENTITY_TYPES
from prompts.mk.keywords_extraction import extract_keywords
from prompts.mk.rag_response import build_rag_response_prompts
from rag.context_parser import parse_source_stems, parse_context_chunks
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


class ConversationalRAG:

    def __init__(
        self,
        config: Config,
        chunker: BaseChunker,
        llm_func,
        embed_func,
        rerank_func=None,
        metadata_repo: MetadataRepository | None = None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.llm_func = llm_func
        self.embed_func = embed_func
        self.rerank_func = rerank_func
        self.metadata_repo = metadata_repo
        self._lightrag: LightRAG | None = None

    async def initialize(self) -> bool:
        postgres_config = self.config.database.to_dict()
        logger.info(
            "Configuring PostgreSQL: %s:%s/%s",
            postgres_config["host"],
            postgres_config["port"],
            postgres_config["database"],
        )

        build_entity_extraction_prompts()
        build_rag_response_prompts()

        self._lightrag = LightRAG(
            working_dir=self.config.working_dir,
            workspace=self.config.workspace,
            llm_model_func=self.llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=self.config.model.embedding_dim,
                max_token_size=self.config.model.max_token_size,
                func=self.embed_func,
            ),
            rerank_model_func=self.rerank_func,
            kv_storage=self.config.storage.kv_storage,
            vector_storage=self.config.storage.vector_storage,
            graph_storage=self.config.storage.graph_storage,
            doc_status_storage=self.config.storage.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "namespace": self.config.vector_namespace,
                "global_config": postgres_config,
            },
            cosine_threshold=0.25,
            cosine_better_than_threshold=0.25,
            addon_params={
                "postgres_config": postgres_config,
                "language": "Macedonian",
                "entity_types": ENTITY_TYPES,
            },
        )

        await self._lightrag.initialize_storages()
        await initialize_pipeline_status()
        logger.info("LightRAG with PostgreSQL initialized.")
        return True

    async def insert_document(
        self, transcript: str, doc_id: str, file_path: str, metadata: str = ""
    ) -> bool:
        if not self._lightrag:
            raise RuntimeError("RAG system not initialized.")
        try:
            chunks = await self.chunker.chunk_transcript(transcript=transcript, doc_id=doc_id)
            if not chunks:
                logger.error("No chunks produced for doc_id=%s", doc_id)
                return False

            header = build_metadata_header(metadata)
            for chunk in chunks:
                await self._lightrag.ainsert(
                    input=header + chunk.formatted_text,
                    ids=chunk.chunk_id,
                    file_paths=file_path,
                    track_id=doc_id,
                )
            return True
        except Exception:
            logger.exception("Failed to insert transcript doc_id=%s", doc_id)
            return False

    async def _fetch_context_str(
        self, question: str, hl_keywords: list[str], ll_keywords: list[str]
    ) -> str:
        try:
            context = await self._lightrag.aquery(
                question,
                param=QueryParam(
                    mode="mix",
                    hl_keywords=hl_keywords,
                    ll_keywords=ll_keywords,
                    only_need_context=True,
                ),
            )
            return str(context)
        except Exception:
            logger.exception("Failed to fetch context")
            return ""

    async def query(
        self,
        question: str,
        mode: str = "mix",
        history: list[dict[str, str]] | None = None,
        return_contexts: bool = False,
    ) -> tuple[str, list[tuple[str, float]]]:
        if not self._lightrag:
            raise RuntimeError("RAG system not initialized.")

        known_speakers = self.metadata_repo.get_all_speakers() if self.metadata_repo else []
        hl_keywords, ll_keywords = await extract_keywords(question, self.llm_func, known_speakers)
        logger.info("hl_keywords=%s ll_keywords=%s", hl_keywords, ll_keywords)

        answer_task = self._lightrag.aquery(
            question,
            param=QueryParam(
                mode=mode,
                hl_keywords=hl_keywords,
                ll_keywords=ll_keywords,
                conversation_history=history or [],
            ),
        )
        context_task = self._fetch_context_str(question, hl_keywords, ll_keywords)

        answer, context_str = await asyncio.gather(answer_task, context_task)
        source_stems = parse_source_stems(context_str)

        if return_contexts:
            return answer, source_stems, parse_context_chunks(context_str)

        return answer, source_stems

    async def finalize(self) -> None:
        if self._lightrag:
            await self._lightrag.finalize_storages()
            logger.info("PostgreSQL storage finalized.")
