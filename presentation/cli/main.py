from __future__ import annotations

"""The ``--build`` CLI: rebuilds the knowledge graph from transcripts already in MinIO."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from composition import Container, build_container


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_system.log"


def setup_logging(verbose: bool = False) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE),
        ],
    )


async def build_knowledge_graph(container: Container) -> bool:
    await container.startup()
    try:
        report = await container.build_graph.build()
        print(
            f"Transcripts: {report.total} | "
            f"Inserted: {report.successful} | Failed: {report.failed}"
        )
        return report.succeeded
    finally:
        await container.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="LightRAG System with PostgreSQL Backend")
    parser.add_argument("--build", action="store_true", help="Build knowledge graph from transcripts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if not args.build:
        print("Use --build to build the knowledge graph from transcripts.")
        return

    try:
        container = build_container()
        print(
            f"LLM: {container.config.model.llm_model} | "
            f"Embeddings: {container.config.model.embedding_model} | "
            f"Chunker: {container.config.chunker.chunker_type}"
        )
        sys.exit(0 if asyncio.run(build_knowledge_graph(container)) else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        logging.exception("Unexpected error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
