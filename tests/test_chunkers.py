import pytest
from unittest.mock import AsyncMock
from parsers.transcript import TranscriptParser
from chunkers.block import BlockChunker
from chunkers.contextual import ContextualChunker
from models.models import SpeakerTurn


def make_turn(speaker: str, start: str, end: str, text: str) -> SpeakerTurn:
    return SpeakerTurn(speaker=speaker, timestamp_start=start, timestamp_end=end, text=text)


TURNS = [
    make_turn("Alice", "00:00", "00:05", "Hello everyone."),
    make_turn("Bob",   "00:05", "00:10", "Hi Alice."),
    make_turn("Alice", "00:10", "00:15", "How are you?"),
    make_turn("Bob",   "00:15", "00:20", "I am fine."),
    make_turn("Alice", "00:20", "00:25", "Great to hear."),
    make_turn("Bob",   "00:25", "00:30", "Thanks."),
]

PARSER = TranscriptParser()


# --- BlockChunker ---

class TestBlockChunker:

    def test_chunk_count(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = pytest.run_async(chunker._build_chunks(TURNS, "doc1"))
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_chunk_count(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_chunk_id_format(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].chunk_id == "doc1::block::00001"

    @pytest.mark.asyncio
    async def test_timestamps_span_block(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].timestamp_start == "00:00"
        assert chunks[0].timestamp_end == "00:10"

    @pytest.mark.asyncio
    async def test_multiple_speakers_joined(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].speaker == "Alice, Bob"

    @pytest.mark.asyncio
    async def test_single_speaker_no_duplicate(self):
        turns = [make_turn("Alice", "00:00", "00:05", "Hi."),
                 make_turn("Alice", "00:05", "00:10", "Bye.")]
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(turns, "doc1")
        assert chunks[0].speaker == "Alice"

    @pytest.mark.asyncio
    async def test_window_larger_than_turns(self):
        chunker = BlockChunker(parser=PARSER, window_size=100)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_empty_turns(self):
        chunker = BlockChunker(parser=PARSER, window_size=5)
        chunks = await chunker._build_chunks([], "doc1")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_text_contains_all_turns(self):
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert "Hello everyone." in chunks[0].text
        assert "Hi Alice." in chunks[0].text

    @pytest.mark.asyncio
    async def test_chunk_transcript_parses_and_chunks(self):
        transcript = (
            "Alice (00:00-00:05): Hello.\n"
            "Bob (00:05-00:10): Hi.\n"
            "Alice (00:10-00:15): Bye.\n"
        )
        chunker = BlockChunker(parser=PARSER, window_size=2)
        chunks = await chunker.chunk_transcript(transcript, "doc1")
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_formatted_text_contains_speaker_and_timestamp(self):
        chunker = BlockChunker(parser=PARSER, window_size=6)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert "[SPEAKER=Alice, Bob]" in chunks[0].formatted_text
        assert "[TIMESTAMP=00:00-00:30]" in chunks[0].formatted_text


# --- ContextualChunker ---

class TestContextualChunker:

    @pytest.mark.asyncio
    async def test_one_chunk_per_turn(self):
        llm = AsyncMock(return_value="Enriched text.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert len(chunks) == len(TURNS)

    @pytest.mark.asyncio
    async def test_chunk_id_format(self):
        llm = AsyncMock(return_value="Enriched.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].chunk_id == "doc1::turn::00001"

    @pytest.mark.asyncio
    async def test_llm_response_used_as_text(self):
        llm = AsyncMock(return_value="  LLM enriched response.  ")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].text == "LLM enriched response."

    @pytest.mark.asyncio
    async def test_falls_back_to_turn_text_on_empty_response(self):
        llm = AsyncMock(return_value="")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].text == TURNS[0].text

    @pytest.mark.asyncio
    async def test_falls_back_to_turn_text_on_whitespace_response(self):
        llm = AsyncMock(return_value="   ")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].text == TURNS[0].text

    @pytest.mark.asyncio
    async def test_speaker_preserved_per_turn(self):
        llm = AsyncMock(return_value="Enriched.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].speaker == "Alice"
        assert chunks[1].speaker == "Bob"

    @pytest.mark.asyncio
    async def test_timestamps_preserved_per_turn(self):
        llm = AsyncMock(return_value="Enriched.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks(TURNS, "doc1")
        assert chunks[0].timestamp_start == "00:00"
        assert chunks[0].timestamp_end == "00:05"

    @pytest.mark.asyncio
    async def test_llm_called_once_per_turn(self):
        llm = AsyncMock(return_value="Enriched.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        await chunker._build_chunks(TURNS, "doc1")
        assert llm.call_count == len(TURNS)

    @pytest.mark.asyncio
    async def test_empty_turns(self):
        llm = AsyncMock(return_value="Enriched.")
        chunker = ContextualChunker(llm, parser=PARSER, context_window=2)
        chunks = await chunker._build_chunks([], "doc1")
        assert chunks == []
