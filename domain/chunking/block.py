from __future__ import annotations

from domain.chunking.base import BaseChunker
from domain.parsing.base import BaseParser
from domain.transcript import Chunk, TranscriptLine


class BlockChunker(BaseChunker):
    """Groups a fixed number of consecutive lines into one chunk."""

    prefix = "block"

    def __init__(self, *, parser: BaseParser, window_size: int) -> None:
        super().__init__(parser=parser)
        self.window_size = window_size

    async def _build_chunks(self, lines: list[TranscriptLine], doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk_idx, offset in enumerate(range(0, len(lines), self.window_size), start=1):
            block = lines[offset : offset + self.window_size]
            speakers = list(dict.fromkeys(line.speaker for line in block))
            chunks.append(self._build_chunk(
                doc_id=doc_id,
                idx=chunk_idx,
                speaker=", ".join(speakers),
                timestamp_start=block[0].timestamp_start,
                timestamp_end=block[-1].timestamp_end,
                text="\n".join(self.parser.format_line(line) for line in block),
            ))
        return chunks
