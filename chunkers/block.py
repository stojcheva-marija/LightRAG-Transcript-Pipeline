from __future__ import annotations

from chunkers.base import BaseChunker
from models.models import SpeakerTurn, Chunk
from parsers.base import BaseParser


class BlockChunker(BaseChunker):
    prefix = "block"

    def __init__(self, *, parser: BaseParser, window_size: int) -> None:
        super().__init__(parser=parser)
        self.window_size = window_size

    async def _build_chunks(self, turns: list[SpeakerTurn], doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chunk_idx, turn_offset in enumerate(range(0, len(turns), self.window_size), start=1):
            turn_block = turns[turn_offset : turn_offset + self.window_size]
            speakers = list(dict.fromkeys(turn.speaker for turn in turn_block))
            speaker_label = ", ".join(speakers)
            text = "\n".join(self.parser.format_speaker_turn(turn) for turn in turn_block)
            chunks.append(self._build_chunk(
                doc_id=doc_id,
                idx=chunk_idx,
                speaker=speaker_label,
                timestamp_start=turn_block[0].timestamp_start,
                timestamp_end=turn_block[-1].timestamp_end,
                text=text,
            ))
        return chunks