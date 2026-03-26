from __future__ import annotations

import asyncio
from typing import Awaitable, Callable
from chunkers.base import BaseChunker
from models.models import SpeakerTurn, Chunk
from parsers.base import BaseParser
from prompts.mk.transcript_enrichment import build_transcript_enrichment_prompt


class ContextualChunker(BaseChunker):
    prefix = "turn"

    def __init__(self, llm_func: Callable[..., Awaitable[str]], *, context_window: int, parser: BaseParser) -> None:
        super().__init__(parser=parser)
        self.llm_func = llm_func
        self.context_window = context_window

    async def _build_chunks(self, turns: list[SpeakerTurn], doc_id: str) -> list[Chunk]:
        enriched = await asyncio.gather(
            *[self._enrich_turn(turns, idx, turn) for idx, turn in enumerate(turns)]
        )

        return [
            self._build_chunk(
                doc_id=doc_id,
                idx=idx + 1,
                speaker=turn.speaker,
                timestamp_start=turn.timestamp_start,
                timestamp_end=turn.timestamp_end,
                text=text,
            )
            for idx, (turn, text) in enumerate(zip(turns, enriched))
        ]

    async def _enrich_turn(self, turns: list[SpeakerTurn], idx: int,
                           turn: SpeakerTurn) -> str:
        preceding_turns, following_turns = self.parser.context_window(
            turns, idx, self.context_window
        )
        prompt = build_transcript_enrichment_prompt(
            prev_block=preceding_turns,
            current_line=self.parser.format_speaker_turn(turn),
            next_block=following_turns,
        )
        llm_response = await self.llm_func(prompt=prompt)
        return llm_response.strip() if isinstance(llm_response, str) and llm_response.strip() else turn.text
