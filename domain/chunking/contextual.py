from __future__ import annotations

import asyncio

from domain.chunking.base import BaseChunker
from domain.parsing.base import BaseParser
from domain.ports import TextCompleter
from domain.prompts.enrichment import build_transcript_enrichment_prompt
from domain.transcript import Chunk, TranscriptLine


class ContextualChunker(BaseChunker):
    """One chunk per line, each rewritten with context from its neighbours."""

    prefix = "turn"

    def __init__(self, completer: TextCompleter, *, context_window: int, parser: BaseParser) -> None:
        super().__init__(parser=parser)
        self.completer = completer
        self.context_window = context_window

    async def _build_chunks(self, lines: list[TranscriptLine], doc_id: str) -> list[Chunk]:
        enriched = await asyncio.gather(
            *[self._enrich(lines, idx, line) for idx, line in enumerate(lines)]
        )

        return [
            self._build_chunk(
                doc_id=doc_id,
                idx=idx + 1,
                speaker=line.speaker,
                timestamp_start=line.timestamp_start,
                timestamp_end=line.timestamp_end,
                text=text,
            )
            for idx, (line, text) in enumerate(zip(lines, enriched))
        ]

    async def _enrich(self, lines: list[TranscriptLine], idx: int, line: TranscriptLine) -> str:
        preceding, following = self.parser.context_window(lines, idx, self.context_window)
        prompt = build_transcript_enrichment_prompt(
            prev_block=preceding,
            current_line=self.parser.format_line(line),
            next_block=following,
        )
        response = await self.completer.complete(prompt)
        return response.strip() if isinstance(response, str) and response.strip() else line.text
