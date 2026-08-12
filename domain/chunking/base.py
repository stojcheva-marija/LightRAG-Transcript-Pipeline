from __future__ import annotations

from abc import ABC, abstractmethod

from domain.parsing.base import BaseParser
from domain.transcript import Chunk, TranscriptLine


class BaseChunker(ABC):
    """Splits a stored transcript into the units that get indexed."""

    def __init__(self, *, parser: BaseParser) -> None:
        self.parser = parser

    @property
    @abstractmethod
    def prefix(self) -> str: ...

    async def chunk_transcript(self, transcript: str, doc_id: str) -> list[Chunk]:
        lines = self.parser.parse(transcript)
        return await self._build_chunks(lines, doc_id)

    @abstractmethod
    async def _build_chunks(self, lines: list[TranscriptLine], doc_id: str) -> list[Chunk]: ...

    def _build_chunk(self, *, doc_id: str, idx: int, speaker: str,
                     timestamp_start: str, timestamp_end: str, text: str) -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}::{self.prefix}::{idx:05d}",
            doc_id=doc_id,
            speaker=speaker,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            text=text,
        )
