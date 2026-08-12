from __future__ import annotations

import json

import pytest

from application.query_transcripts import QueryTranscriptsUseCase
from domain.retrieval import RetrievedAnswer, SourceReference
from tests.fakes import FakeArchive, FakeCompleter, FakeKnowledgeBase, FakeMetadataStore

NO_KEYWORDS = json.dumps({"speakers": [], "dates": []})


def build(answer: RetrievedAnswer, stems=("sednica_77",), known_speakers=("Ана",), archive=None):
    archive = archive or FakeArchive(list(stems))
    knowledge_base = FakeKnowledgeBase(answer)
    use_case = QueryTranscriptsUseCase(
        knowledge_base=knowledge_base,
        archive=archive,
        metadata_store=FakeMetadataStore(list(known_speakers)),
        completer=FakeCompleter([NO_KEYWORDS]),
    )
    return use_case, knowledge_base, archive


@pytest.mark.asyncio
async def test_answer_is_returned():
    use_case, _, _ = build(RetrievedAnswer(answer="Одговорот."))
    assert (await use_case.ask("Прашање?")).text == "Одговорот."


@pytest.mark.asyncio
async def test_trailing_reference_list_is_stripped():
    answer = RetrievedAnswer(answer="Одговорот.\n\n### References\n[1] transcripts/x/x.txt")
    use_case, _, _ = build(answer)
    assert (await use_case.ask("Прашање?")).text == "Одговорот."


@pytest.mark.asyncio
async def test_sources_become_playable_urls():
    answer = RetrievedAnswer(answer="a", sources=[SourceReference("sednica_77", 90.0)])
    use_case, _, _ = build(answer)
    sources = (await use_case.ask("Прашање?")).sources
    assert len(sources) == 1
    assert sources[0].stem == "sednica_77"
    assert sources[0].start_seconds == 90.0
    assert sources[0].audio_url == "https://audio.test/sednica_77.wav"


@pytest.mark.asyncio
async def test_cited_stem_is_resolved_against_the_archive():
    answer = RetrievedAnswer(answer="a", sources=[SourceReference("Sednica-77-part2", 5.0)])
    use_case, _, _ = build(answer)
    assert (await use_case.ask("Прашање?")).sources[0].stem == "sednica_77"


@pytest.mark.asyncio
async def test_sources_without_playable_audio_are_dropped():
    answer = RetrievedAnswer(answer="a", sources=[SourceReference("missing_show", 5.0)])
    use_case, _, _ = build(answer)
    assert (await use_case.ask("Прашање?")).sources == []


@pytest.mark.asyncio
async def test_at_most_three_sources_are_returned():
    references = [SourceReference("sednica_77", float(i)) for i in range(10)]
    use_case, _, _ = build(RetrievedAnswer(answer="a", sources=references))
    assert len((await use_case.ask("Прашање?")).sources) == 3


@pytest.mark.asyncio
async def test_audio_urls_are_looked_up_once_per_stem():
    calls = []
    archive = FakeArchive(["sednica_77"])
    original = archive.audio_url

    def counting(stem, expires_in=3600):
        calls.append(stem)
        return original(stem, expires_in)

    archive.audio_url = counting
    references = [SourceReference("sednica_77", float(i)) for i in range(3)]
    use_case, _, _ = build(RetrievedAnswer(answer="a", sources=references), archive=archive)

    await use_case.ask("Прашање?")
    assert calls == ["sednica_77"]


@pytest.mark.asyncio
async def test_extracted_keywords_are_passed_to_retrieval():
    completer = FakeCompleter([json.dumps({"speakers": ["Ана"], "dates": ["2024-01-01"]})])
    knowledge_base = FakeKnowledgeBase(RetrievedAnswer(answer="a"))
    use_case = QueryTranscriptsUseCase(
        knowledge_base=knowledge_base,
        archive=FakeArchive(),
        metadata_store=FakeMetadataStore(["Ана"]),
        completer=completer,
    )

    await use_case.ask("Кога зборуваше Ана?")

    _, keywords = knowledge_base.queries[0]
    assert keywords.speakers == ["Ана"]
    assert keywords.dates == ["2024-01-01"]


@pytest.mark.asyncio
async def test_conversation_history_is_forwarded():
    knowledge_base = FakeKnowledgeBase(RetrievedAnswer(answer="a"))
    use_case = QueryTranscriptsUseCase(
        knowledge_base=knowledge_base,
        archive=FakeArchive(),
        metadata_store=FakeMetadataStore(),
        completer=FakeCompleter([NO_KEYWORDS]),
    )
    history = [{"role": "user", "content": "претходно"}]

    captured = {}
    original = knowledge_base.query

    async def capture(question, keywords, history=None):
        captured["history"] = history
        return await original(question, keywords, history)

    knowledge_base.query = capture
    await use_case.ask("Прашање?", history)
    assert captured["history"] == history
