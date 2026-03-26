from __future__ import annotations

from abc import ABC, abstractmethod
from models.models import SpeakerTurn, Chunk
from parsers.base import BaseParser


class BaseChunker(ABC):

    def __init__(self, *, parser: BaseParser) -> None:
        self.parser = parser

    @property
    @abstractmethod
    def prefix(self) -> str: ...

    async def chunk_transcript(self, transcript: str, doc_id: str) -> list[Chunk]:
        turns = self.parser.parse(transcript)
        return await self._build_chunks(turns, doc_id)

    @abstractmethod
    async def _build_chunks(self, turns: list[SpeakerTurn], doc_id: str) -> list[Chunk]: ...

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