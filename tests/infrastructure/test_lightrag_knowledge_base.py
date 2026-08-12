from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from domain.retrieval import QueryKeywords
from infrastructure.rag.lightrag_knowledge_base import LightRAGKnowledgeBase


def make_knowledge_base(initialized: bool = True) -> LightRAGKnowledgeBase:
    knowledge_base = LightRAGKnowledgeBase(
        config=MagicMock(),
        chunker=MagicMock(),
        llm_func=AsyncMock(),
        embed_func=AsyncMock(),
    )
    if initialized:
        knowledge_base._lightrag = MagicMock()
        knowledge_base._lightrag.ainsert = AsyncMock()
        knowledge_base._lightrag.aquery = AsyncMock()
    return knowledge_base


def make_chunk(chunk_id: str = "c1", text: str = "chunk text"):
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.formatted_text = text
    return chunk


# --- insert_document ---

@pytest.mark.asyncio
async def test_insert_returns_false_when_no_chunks():
    kb = make_knowledge_base()
    kb.chunker.chunk_transcript = AsyncMock(return_value=[])
    assert await kb.insert_document("text", "doc1", "path/doc1.txt") is False


@pytest.mark.asyncio
async def test_insert_calls_ainsert_per_chunk():
    kb = make_knowledge_base()
    kb.chunker.chunk_transcript = AsyncMock(return_value=[make_chunk(), make_chunk()])
    assert await kb.insert_document("text", "doc1", "path/doc1.txt") is True
    assert kb._lightrag.ainsert.call_count == 2


@pytest.mark.asyncio
async def test_insert_prepends_metadata_header():
    kb = make_knowledge_base()
    kb.chunker.chunk_transcript = AsyncMock(return_value=[make_chunk()])
    metadata = json.dumps({"date": "2024-01-01", "show": "My Show"})
    await kb.insert_document("text", "doc1", "path/doc1.txt", metadata)
    call_input = kb._lightrag.ainsert.call_args.kwargs["input"]
    assert "[DATE=2024-01-01]" in call_input
    assert "[SHOW=My Show]" in call_input


@pytest.mark.asyncio
async def test_insert_tracks_the_document_id():
    kb = make_knowledge_base()
    kb.chunker.chunk_transcript = AsyncMock(return_value=[make_chunk()])
    await kb.insert_document("text", "doc1", "path/doc1.txt")
    assert kb._lightrag.ainsert.call_args.kwargs["track_id"] == "doc1"


@pytest.mark.asyncio
async def test_insert_returns_false_on_exception():
    kb = make_knowledge_base()
    kb.chunker.chunk_transcript = AsyncMock(side_effect=RuntimeError("boom"))
    assert await kb.insert_document("text", "doc1", "path/doc1.txt") is False


@pytest.mark.asyncio
async def test_insert_raises_if_not_initialized():
    kb = make_knowledge_base(initialized=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        await kb.insert_document("text", "doc1", "path/doc1.txt")


# --- query ---

@pytest.mark.asyncio
async def test_query_raises_if_not_initialized():
    kb = make_knowledge_base(initialized=False)
    with pytest.raises(RuntimeError, match="not initialized"):
        await kb.query("what?", QueryKeywords.none())


@pytest.mark.asyncio
async def test_query_returns_answer_sources_and_context():
    kb = make_knowledge_base()
    context = (
        '{"reference_id": "1", "content": "text [TIMESTAMP=1:30-1:45]"}\n\n'
        "[1] transcripts/show_a/show_a.txt"
    )
    kb._lightrag.aquery = AsyncMock(side_effect=["The answer", context])

    result = await kb.query("what?", QueryKeywords.none())

    assert result.answer == "The answer"
    assert result.sources[0].stem == "show_a"
    assert len(result.context_chunks) == 2


@pytest.mark.asyncio
async def test_query_passes_keywords_to_retrieval():
    kb = make_knowledge_base()
    kb._lightrag.aquery = AsyncMock(side_effect=["answer", ""])
    keywords = QueryKeywords(speakers=["Ана"], dates=["2024-01-01"])

    await kb.query("what?", keywords)

    param = kb._lightrag.aquery.call_args_list[0].kwargs["param"]
    assert param.hl_keywords == ["Ана"]
    assert param.ll_keywords == ["2024-01-01"]


@pytest.mark.asyncio
async def test_query_survives_a_failing_context_fetch():
    kb = make_knowledge_base()
    kb._lightrag.aquery = AsyncMock(side_effect=["The answer", RuntimeError("no context")])

    result = await kb.query("what?", QueryKeywords.none())

    assert result.answer == "The answer"
    assert result.sources == []
