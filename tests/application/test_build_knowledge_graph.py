from __future__ import annotations

import pytest

from application.build_knowledge_graph import BuildKnowledgeGraphUseCase
from tests.fakes import FakeArchive, FakeKnowledgeBase, FakeMetadataStore

METADATA = '{"date": "2024-01-01", "speakers": ["Ана"]}'


def build(stems=("show_a", "show_b"), insert_results=None, transcripts=None):
    archive = FakeArchive(list(stems))
    archive.transcripts = transcripts if transcripts is not None else {
        stem: ("transcript text", METADATA) for stem in stems
    }
    metadata_store = FakeMetadataStore()
    knowledge_base = FakeKnowledgeBase(insert_results=insert_results)
    use_case = BuildKnowledgeGraphUseCase(
        archive=archive, metadata_store=metadata_store, knowledge_base=knowledge_base
    )
    return use_case, archive, metadata_store, knowledge_base


@pytest.mark.asyncio
async def test_reports_success_when_everything_is_inserted(progress):
    use_case, *_ = build()
    report = await use_case.build(progress)
    assert report.succeeded
    assert (report.total, report.successful, report.failed) == (2, 2, 0)


@pytest.mark.asyncio
async def test_empty_archive_is_not_a_success(progress):
    use_case, _, _, knowledge_base = build(stems=())
    report = await use_case.build(progress)
    assert not report.succeeded
    assert report.total == 0
    assert knowledge_base.inserted == []


@pytest.mark.asyncio
async def test_a_failed_insert_is_reported(progress):
    use_case, *_ = build(insert_results=[True, False])
    report = await use_case.build(progress)
    assert not report.succeeded
    assert (report.successful, report.failed) == (1, 1)


@pytest.mark.asyncio
async def test_metadata_is_saved_for_successful_inserts_only(progress):
    use_case, _, metadata_store, _ = build(insert_results=[True, False])
    await use_case.build(progress)
    assert list(metadata_store.saved) == ["show_a"]


@pytest.mark.asyncio
async def test_documents_are_indexed_under_a_stable_stem_id(progress):
    use_case, _, _, knowledge_base = build(stems=("show_a",))
    await use_case.build(progress)
    _, doc_id, file_path, _ = knowledge_base.inserted[0]
    assert doc_id == "show_a"
    assert file_path == "transcripts/show_a/show_a.txt"


@pytest.mark.asyncio
async def test_rebuilding_reuses_the_same_document_id(progress):
    use_case, _, _, knowledge_base = build(stems=("show_a",))
    await use_case.build(progress)
    await use_case.build(progress)
    assert {entry[1] for entry in knowledge_base.inserted} == {"show_a"}


@pytest.mark.asyncio
async def test_a_broken_transcript_does_not_stop_the_run(progress):
    use_case, archive, _, knowledge_base = build()
    del archive.transcripts["show_a"]

    report = await use_case.build(progress)

    assert (report.successful, report.failed) == (1, 1)
    assert len(knowledge_base.inserted) == 1


@pytest.mark.asyncio
async def test_transcripts_without_metadata_are_still_indexed(progress):
    use_case, _, metadata_store, _ = build(
        stems=("show_a",), transcripts={"show_a": ("transcript text", "")}
    )
    report = await use_case.build(progress)
    assert report.succeeded
    assert metadata_store.saved["show_a"][1] == {}
