from __future__ import annotations

import pytest

from domain.chunking.block import BlockChunker
from domain.chunking.contextual import ContextualChunker
from domain.chunking.factory import make_chunker
from domain.parsing.transcript import TranscriptParser
from domain.transcript import TranscriptLine

from tests.fakes import FakeCompleter


def make_line(speaker: str, start: str, end: str, text: str) -> TranscriptLine:
    return TranscriptLine(speaker=speaker, timestamp_start=start, timestamp_end=end, text=text)


LINES = [
    make_line("Alice", "00:00", "00:05", "Hello everyone."),
    make_line("Bob",   "00:05", "00:10", "Hi Alice."),
    make_line("Alice", "00:10", "00:15", "How are you?"),
    make_line("Bob",   "00:15", "00:20", "I am fine."),
    make_line("Alice", "00:20", "00:25", "Great to hear."),
    make_line("Bob",   "00:25", "00:30", "Thanks."),
]

PARSER = TranscriptParser()


def repeating(response: str) -> FakeCompleter:
    completer = FakeCompleter([response] * len(LINES))
    return completer


# --- BlockChunker ---

class TestBlockChunker:

    @pytest.mark.asyncio
    async def test_chunk_count(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_chunk_id_format(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        assert chunks[0].chunk_id == "doc1::block::00001"

    @pytest.mark.asyncio
    async def test_timestamps_span_block(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        assert chunks[0].timestamp_start == "00:00"
        assert chunks[0].timestamp_end == "00:10"

    @pytest.mark.asyncio
    async def test_multiple_speakers_joined(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        assert chunks[0].speaker == "Alice, Bob"

    @pytest.mark.asyncio
    async def test_single_speaker_no_duplicate(self):
        lines = [make_line("Alice", "00:00", "00:05", "Hi."),
                 make_line("Alice", "00:05", "00:10", "Bye.")]
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(lines, "doc1")
        assert chunks[0].speaker == "Alice"

    @pytest.mark.asyncio
    async def test_window_larger_than_lines(self):
        chunks = await BlockChunker(parser=PARSER, window_size=100)._build_chunks(LINES, "doc1")
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_empty_lines(self):
        chunks = await BlockChunker(parser=PARSER, window_size=5)._build_chunks([], "doc1")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_text_contains_all_lines_in_block(self):
        chunks = await BlockChunker(parser=PARSER, window_size=2)._build_chunks(LINES, "doc1")
        assert "Hello everyone." in chunks[0].text
        assert "Hi Alice." in chunks[0].text

    @pytest.mark.asyncio
    async def test_chunk_transcript_parses_and_chunks(self):
        transcript = (
            "Alice (00:00-00:05): Hello.\n"
            "Bob (00:05-00:10): Hi.\n"
            "Alice (00:10-00:15): Bye.\n"
        )
        chunks = await BlockChunker(parser=PARSER, window_size=2).chunk_transcript(transcript, "doc1")
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_formatted_text_contains_speaker_and_timestamp(self):
        chunks = await BlockChunker(parser=PARSER, window_size=6)._build_chunks(LINES, "doc1")
        assert "[SOURCE=doc1]" in chunks[0].formatted_text
        assert "[SPEAKER=Alice, Bob]" in chunks[0].formatted_text
        assert "[TIMESTAMP=00:00-00:30]" in chunks[0].formatted_text


# --- ContextualChunker ---

class TestContextualChunker:

    @pytest.mark.asyncio
    async def test_one_chunk_per_line(self):
        chunker = ContextualChunker(repeating("Enriched text."), parser=PARSER, context_window=2)
        assert len(await chunker._build_chunks(LINES, "doc1")) == len(LINES)

    @pytest.mark.asyncio
    async def test_chunk_id_format(self):
        chunker = ContextualChunker(repeating("Enriched."), parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].chunk_id == "doc1::turn::00001"

    @pytest.mark.asyncio
    async def test_llm_response_used_as_text(self):
        chunker = ContextualChunker(
            repeating("  LLM enriched response.  "), parser=PARSER, context_window=2
        )
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].text == "LLM enriched response."

    @pytest.mark.asyncio
    async def test_falls_back_to_line_text_on_empty_response(self):
        chunker = ContextualChunker(repeating(""), parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].text == LINES[0].text

    @pytest.mark.asyncio
    async def test_falls_back_to_line_text_on_whitespace_response(self):
        chunker = ContextualChunker(repeating("   "), parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].text == LINES[0].text

    @pytest.mark.asyncio
    async def test_speaker_preserved_per_line(self):
        chunker = ContextualChunker(repeating("Enriched."), parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].speaker == "Alice"
        assert chunks[1].speaker == "Bob"

    @pytest.mark.asyncio
    async def test_timestamps_preserved_per_line(self):
        chunker = ContextualChunker(repeating("Enriched."), parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(LINES, "doc1")
        assert chunks[0].timestamp_start == "00:00"
        assert chunks[0].timestamp_end == "00:05"

    @pytest.mark.asyncio
    async def test_llm_called_once_per_line(self):
        completer = repeating("Enriched.")
        chunker = ContextualChunker(completer, parser=PARSER, context_window=2)
        await chunker._build_chunks(LINES, "doc1")
        assert len(completer.prompts) == len(LINES)

    @pytest.mark.asyncio
    async def test_prompt_includes_neighbouring_lines(self):
        completer = repeating("Enriched.")
        chunker = ContextualChunker(completer, parser=PARSER, context_window=1)
        await chunker._build_chunks(LINES, "doc1")
        assert "Hello everyone." in completer.prompts[1]

    @pytest.mark.asyncio
    async def test_empty_lines(self):
        chunker = ContextualChunker(repeating("Enriched."), parser=PARSER, context_window=2)
        assert await chunker._build_chunks([], "doc1") == []


# --- factory ---

class TestMakeChunker:

    def _make(self, chunker_type: str):
        return make_chunker(
            chunker_type,
            completer=FakeCompleter(),
            block_window_size=5,
            contextual_window_size=3,
        )

    def test_builds_block_chunker(self):
        chunker = self._make("block")
        assert isinstance(chunker, BlockChunker)
        assert chunker.window_size == 5

    def test_builds_contextual_chunker(self):
        chunker = self._make("contextual")
        assert isinstance(chunker, ContextualChunker)
        assert chunker.context_window == 3

    def test_type_is_case_and_whitespace_insensitive(self):
        assert isinstance(self._make("  BLOCK "), BlockChunker)

    def test_unknown_type_raises_with_the_valid_options(self):
        with pytest.raises(ValueError, match="block, contextual"):
            self._make("nonsense")
