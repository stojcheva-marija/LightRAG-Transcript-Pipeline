from __future__ import annotations

import asyncio
import logging

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from config.settings import Config
from domain.chunking.base import BaseChunker
from domain.metadata import metadata_header
from domain.retrieval import QueryKeywords, RetrievedAnswer
from infrastructure.rag.context_parser import parse_context_chunks, parse_source_references
from infrastructure.rag.prompts.entity_extraction import (
    ENTITY_TYPES_GUIDANCE,
    build_entity_extraction_prompts,
)
from infrastructure.rag.prompts.rag_response import build_rag_response_prompts

logger = logging.getLogger(__name__)

QUERY_MODE = "mix"
COSINE_THRESHOLD = 0.25


class LightRAGKnowledgeBase:
    """``KnowledgeBase`` backed by LightRAG over PostgreSQL.

    Chunking is done by the domain before insertion, so LightRAG's own chunker
    is replaced with a pass-through.
    """

    def __init__(
        self,
        config: Config,
        chunker: BaseChunker,
        llm_func,
        embed_func,
        rerank_func=None,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.llm_func = llm_func
        self.embed_func = embed_func
        self.rerank_func = rerank_func
        self._lightrag: LightRAG | None = None

    async def initialize(self) -> None:
        postgres_config = self.config.database.to_dict()
        logger.info(
            "Configuring PostgreSQL: %s:%s/%s",
            postgres_config["host"],
            postgres_config["port"],
            postgres_config["database"],
        )

        # LightRAG reads its prompts from a module-level table; install ours.
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
            chunking_func=_passthrough_chunk,
            kv_storage=self.config.storage.kv_storage,
            vector_storage=self.config.storage.vector_storage,
            graph_storage=self.config.storage.graph_storage,
            doc_status_storage=self.config.storage.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "namespace": self.config.vector_namespace,
                "global_config": postgres_config,
            },
            cosine_better_than_threshold=COSINE_THRESHOLD,
            addon_params={
                "postgres_config": postgres_config,
                "language": self.config.language.name,
                "entity_types_guidance": ENTITY_TYPES_GUIDANCE,
            },
        )

        await self._lightrag.initialize_storages()
        await initialize_pipeline_status()
        logger.info("LightRAG with PostgreSQL initialized.")

    async def insert_document(
        self, transcript: str, doc_id: str, file_path: str, metadata: str = ""
    ) -> bool:
        self._require_ready()
        try:
            chunks = await self.chunker.chunk_transcript(transcript=transcript, doc_id=doc_id)
            if not chunks:
                logger.error("No chunks produced for doc_id=%s", doc_id)
                return False

            header = metadata_header(metadata)
            await self._lightrag.ainsert(
                input=[header + chunk.formatted_text for chunk in chunks],
                ids=[chunk.chunk_id for chunk in chunks],
                # LightRAG treats any two documents with the same file_path as
                # duplicates of each other, even within one batch — so each
                # block needs its own distinct path. The suffix comes after
                # ".txt" so citation parsing (which matches on the
                # "transcripts/{stem}/{stem}.txt" substring) still finds it.
                file_paths=[f"{file_path}#{chunk.chunk_id}" for chunk in chunks],
                track_id=doc_id,
            )
            return True
        except Exception:
            logger.exception("Failed to insert transcript doc_id=%s", doc_id)
            return False

    async def query(
        self,
        question: str,
        keywords: QueryKeywords,
        history: list[dict[str, str]] | None = None,
    ) -> RetrievedAnswer:
        self._require_ready()

        answer, context_str = await asyncio.gather(
            self._lightrag.aquery(
                question,
                param=QueryParam(
                    mode=QUERY_MODE,
                    hl_keywords=keywords.speakers,
                    ll_keywords=keywords.dates,
                    conversation_history=history or [],
                ),
            ),
            self._fetch_context(question, keywords),
        )

        logger.debug("Context str: %s", context_str[:500] if context_str else "EMPTY")
        return RetrievedAnswer(
            answer=answer,
            sources=parse_source_references(context_str),
            context_chunks=parse_context_chunks(context_str),
        )

    async def finalize(self) -> None:
        if self._lightrag:
            await self._lightrag.finalize_storages()
            logger.info("PostgreSQL storage finalized.")

    async def _fetch_context(self, question: str, keywords: QueryKeywords) -> str:
        """Retrieve the raw context alongside the answer, to cite sources from."""
        try:
            context = await self._lightrag.aquery(
                question,
                param=QueryParam(
                    mode=QUERY_MODE,
                    hl_keywords=keywords.speakers,
                    ll_keywords=keywords.dates,
                    only_need_context=True,
                ),
            )
            return str(context)
        except Exception:
            logger.exception("Failed to fetch context")
            return ""

    def _require_ready(self) -> None:
        if not self._lightrag:
            raise RuntimeError("Knowledge base not initialized.")


def _passthrough_chunk(tokenizer, content: str, *_) -> list[dict]:
    return [{"content": content, "tokens": len(tokenizer.encode(content)), "chunk_order_index": 0}]
