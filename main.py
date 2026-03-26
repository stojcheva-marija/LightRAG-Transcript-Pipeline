from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from chunkers.block import BlockChunker
from config.settings import Config
from rag.conversational_rag import ConversationalRAG
from rag.graph_builder import GraphBuilder
from llm.adapters import make_llm_func, make_embed_func, make_rerank_func
from parsers.transcript import TranscriptParser
from storage.minio_client import MinIOClient
from storage.metadata_repository import MetadataRepository

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/rag_system.log"),
        ],
    )


def build_rag_and_builder(config: Config) -> tuple[ConversationalRAG, GraphBuilder]:
    minio = MinIOClient(config)
    metadata_repo = MetadataRepository(config.database)

    chunker = BlockChunker(parser=TranscriptParser(), window_size=config.chunker.block_window_size)
    rag = ConversationalRAG(
        config=config,
        chunker=chunker,
        llm_func=make_llm_func(config),
        embed_func=make_embed_func(config),
        rerank_func=make_rerank_func(config),
        metadata_repo=metadata_repo,
    )
    builder = GraphBuilder(config=config, rag=rag, minio=minio)
    return rag, builder


async def build_mode(config: Config) -> bool:
    _, builder = build_rag_and_builder(config)
    try:
        return await builder.build()
    finally:
        await builder.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="LightRAG System with PostgreSQL Backend")
    parser.add_argument("--build",   action="store_true", help="Build knowledge graph from transcripts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    config = Config.from_env()

    print(f"LLM: {config.model.llm_model} | Embeddings: {config.model.embedding_model}")

    try:
        if args.build:
            sys.exit(0 if asyncio.run(build_mode(config)) else 1)
        else:
            print("Use --build to build the knowledge graph from transcripts.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        logging.exception("Unexpected error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
