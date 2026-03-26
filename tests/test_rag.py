from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.context_parser import (
    _timestamp_to_seconds,
    parse_context_chunks,
    parse_source_stems,
)


# ---------------------------------------------------------------------------
# context_parser — _timestamp_to_seconds
# ---------------------------------------------------------------------------

def test_timestamp_to_seconds_basic():
    assert _timestamp_to_seconds("1:30") == 90.0

def test_timestamp_to_seconds_zero():
    assert _timestamp_to_seconds("0:00") == 0.0

def test_timestamp_to_seconds_hours_expressed_as_minutes():
    assert _timestamp_to_seconds("90:00") == 5400.0

def test_timestamp_to_seconds_invalid_returns_zero():
    assert _timestamp_to_seconds("bad") == 0.0

def test_timestamp_to_seconds_empty_returns_zero():
    assert _timestamp_to_seconds("") == 0.0


# ---------------------------------------------------------------------------
# context_parser — parse_source_stems
# ---------------------------------------------------------------------------

CONTEXT_WITH_STEMS = (
    "Some text\n"
    "[TIMESTAMP=02:10-03:00]\n"
    "transcripts/show_a/show_a.txt\n"
    "transcripts/show_b/show_b.txt\n"
)

def test_parse_source_stems_extracts_stems():
    result = parse_source_stems(CONTEXT_WITH_STEMS)
    stems = [s for s, _ in result]
    assert stems == ["show_a", "show_b"]

def test_parse_source_stems_uses_first_timestamp():
    result = parse_source_stems(CONTEXT_WITH_STEMS)
    assert result[0][1] == pytest.approx(130.0)  # 2*60 + 10

def test_parse_source_stems_no_timestamp_defaults_zero():
    context = "transcripts/show_a/show_a.txt\n"
    result = parse_source_stems(context)
    assert result[0][1] == 0.0

def test_parse_source_stems_deduplicates():
    context = (
        "transcripts/show_a/show_a.txt\n"
        "transcripts/show_a/show_a.txt\n"
    )
    result = parse_source_stems(context)
    assert len(result) == 1

def test_parse_source_stems_empty_context():
    assert parse_source_stems("") == []

def test_parse_source_stems_no_match():
    assert parse_source_stems("nothing relevant here") == []

def test_parse_source_stems_preserves_order():
    context = (
        "transcripts/zebra/zebra.txt\n"
        "transcripts/alpha/alpha.txt\n"
    )
    stems = [s for s, _ in parse_source_stems(context)]
    assert stems == ["zebra", "alpha"]

def test_parse_source_stems_mismatched_path_ignored():
    # path where folder name != file name stem — should not match
    context = "transcripts/show_a/show_b.txt\n"
    assert parse_source_stems(context) == []


# ---------------------------------------------------------------------------
# context_parser — parse_context_chunks
# ---------------------------------------------------------------------------

def test_parse_context_chunks_splits_on_double_newline():
    context = "chunk one\n\nchunk two\n\nchunk three"
    assert parse_context_chunks(context) == ["chunk one", "chunk two", "chunk three"]

def test_parse_context_chunks_strips_whitespace():
    context = "  chunk one  \n\n  chunk two  "
    assert parse_context_chunks(context) == ["chunk one", "chunk two"]

def test_parse_context_chunks_filters_blank_sections():
    context = "chunk one\n\n\n\nchunk two"
    assert parse_context_chunks(context) == ["chunk one", "chunk two"]

def test_parse_context_chunks_empty_returns_list_with_empty_string():
    assert parse_context_chunks("") == [""]

def test_parse_context_chunks_no_double_newline_returns_single():
    assert parse_context_chunks("just one chunk") == ["just one chunk"]


# ---------------------------------------------------------------------------
# ConversationalRAG — insert_document
# ---------------------------------------------------------------------------

def make_rag():
    from rag.conversational_rag import ConversationalRAG
    config = MagicMock()
    config.database.to_dict.return_value = {
        "host": "localhost", "port": 5432, "database": "db", "user": "u", "password": "p"
    }
    config.working_dir = "/tmp"
    config.workspace = "test"
    config.model.embedding_dim = 768
    config.model.max_token_size = 512
    config.storage.kv_storage = "PGKVStorage"
    config.storage.vector_storage = "PGVectorStorage"
    config.storage.graph_storage = "AGEStorage"
    config.storage.doc_status_storage = "PGDocStatusStorage"
    config.vector_namespace = "test_ns"

    rag = ConversationalRAG(
        config=config,
        chunker=MagicMock(),
        llm_func=AsyncMock(),
        embed_func=AsyncMock(),
    )
    rag._lightrag = MagicMock()
    rag._lightrag.ainsert = AsyncMock()
    rag._lightrag.aquery = AsyncMock()
    return rag


@pytest.mark.asyncio
async def test_insert_document_returns_false_when_no_chunks():
    rag = make_rag()
    rag.chunker.chunk_transcript = AsyncMock(return_value=[])
    result = await rag.insert_document("text", "doc1", "path/doc1.txt")
    assert result is False


@pytest.mark.asyncio
async def test_insert_document_calls_ainsert_per_chunk():
    rag = make_rag()
    chunk = MagicMock()
    chunk.formatted_text = "chunk text"
    chunk.chunk_id = "c1"
    rag.chunker.chunk_transcript = AsyncMock(return_value=[chunk, chunk])
    result = await rag.insert_document("text", "doc1", "path/doc1.txt")
    assert result is True
    assert rag._lightrag.ainsert.call_count == 2


@pytest.mark.asyncio
async def test_insert_document_prepends_metadata_header():
    rag = make_rag()
    chunk = MagicMock()
    chunk.formatted_text = "chunk text"
    chunk.chunk_id = "c1"
    rag.chunker.chunk_transcript = AsyncMock(return_value=[chunk])
    import json
    metadata = json.dumps({"date": "2024-01-01", "show": "My Show"})
    await rag.insert_document("text", "doc1", "path/doc1.txt", metadata)
    call_input = rag._lightrag.ainsert.call_args.kwargs["input"]
    assert "[DATE=2024-01-01]" in call_input
    assert "[SHOW=My Show]" in call_input


@pytest.mark.asyncio
async def test_insert_document_returns_false_on_exception():
    rag = make_rag()
    rag.chunker.chunk_transcript = AsyncMock(side_effect=RuntimeError("boom"))
    result = await rag.insert_document("text", "doc1", "path/doc1.txt")
    assert result is False


@pytest.mark.asyncio
async def test_insert_document_raises_if_not_initialized():
    from rag.conversational_rag import ConversationalRAG
    rag = ConversationalRAG(
        config=MagicMock(), chunker=MagicMock(),
        llm_func=AsyncMock(), embed_func=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        await rag.insert_document("text", "doc1", "path/doc1.txt")


# ---------------------------------------------------------------------------
# ConversationalRAG — query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_raises_if_not_initialized():
    from rag.conversational_rag import ConversationalRAG
    rag = ConversationalRAG(
        config=MagicMock(), chunker=MagicMock(),
        llm_func=AsyncMock(), embed_func=AsyncMock(),
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        await rag.query("what?")


@pytest.mark.asyncio
async def test_query_returns_answer_and_source_stems():
    rag = make_rag()
    rag._lightrag.aquery = AsyncMock(side_effect=[
        "The answer",
        "transcripts/show_a/show_a.txt [TIMESTAMP=01:00-02:00]",
    ])
    with patch("rag.conversational_rag.extract_keywords", new=AsyncMock(return_value=([], []))):
        answer, source_stems = await rag.query("what happened?")
    assert answer == "The answer"
    assert source_stems[0][0] == "show_a"


@pytest.mark.asyncio
async def test_query_return_contexts_true():
    rag = make_rag()
    rag._lightrag.aquery = AsyncMock(side_effect=[
        "The answer",
        "chunk one\n\nchunk two",
    ])
    with patch("rag.conversational_rag.extract_keywords", new=AsyncMock(return_value=([], []))):
        answer, source_stems, contexts = await rag.query("what?", return_contexts=True)
    assert answer == "The answer"
    assert contexts == ["chunk one", "chunk two"]


# ---------------------------------------------------------------------------
# GraphBuilder — build
# ---------------------------------------------------------------------------

def make_builder():
    from rag.graph_builder import GraphBuilder
    rag = MagicMock()
    rag.initialize = AsyncMock(return_value=True)
    rag.insert_document = AsyncMock(return_value=True)
    rag.finalize = AsyncMock()

    minio = MagicMock()
    minio.list_transcripts.return_value = ["show_a", "show_b"]
    minio.load_transcript_pair.return_value = ("transcript text", '{"date": "2024-01-01"}')

    config = MagicMock()
    config.database = MagicMock()

    with patch("rag.graph_builder.MetadataRepository"):
        builder = GraphBuilder(config=config, rag=rag, minio=minio)
    builder.metadata_repo = MagicMock()
    return builder, rag, minio


@pytest.mark.asyncio
async def test_build_returns_true_when_all_inserted():
    builder, rag, _ = make_builder()
    result = await builder.build()
    assert result is True


@pytest.mark.asyncio
async def test_build_returns_false_when_no_transcripts():
    builder, rag, minio = make_builder()
    minio.list_transcripts.return_value = []
    result = await builder.build()
    assert result is False
    rag.initialize.assert_not_called()


@pytest.mark.asyncio
async def test_build_returns_false_when_any_insert_fails():
    builder, rag, _ = make_builder()
    rag.insert_document = AsyncMock(side_effect=[True, False])
    result = await builder.build()
    assert result is False


@pytest.mark.asyncio
async def test_build_saves_metadata_for_successful_inserts():
    builder, rag, _ = make_builder()
    await builder.build()
    assert builder.metadata_repo.save.call_count == 2


@pytest.mark.asyncio
async def test_build_continues_after_exception():
    builder, rag, minio = make_builder()
    minio.load_transcript_pair.side_effect = [
        Exception("read error"),
        ("transcript text", "{}"),
    ]
    result = await builder.build()
    assert result is False
    # second stem still attempted
    assert rag.insert_document.call_count == 1


@pytest.mark.asyncio
async def test_cleanup_calls_finalize():
    builder, rag, _ = make_builder()
    await builder.cleanup()
    rag.finalize.assert_called_once()
