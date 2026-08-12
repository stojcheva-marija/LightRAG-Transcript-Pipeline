from __future__ import annotations

import json

import pytest

from application.ingest_transcript import IngestTranscriptUseCase, IngestionRequest
from domain.errors import AudioNotFound, DiarizationNotCached
from domain.metadata import SessionDetails
from domain.speaker import SpeakerMatch
from domain.transcript import SpeakerTurn
from tests.fakes import (
    FakeArchive,
    FakeCompleter,
    FakeDiarizer,
    FakeKnowledgeBase,
    FakeMetadataStore,
    FakeSpeakerDirectory,
    FakeTranscriber,
    make_clustering,
)

STEM = "sednica_77"
TURNS = [
    SpeakerTurn(speaker="speaker_1", start=0.0, end=10.0, text="Добар ден."),
    SpeakerTurn(speaker="speaker_2", start=10.0, end=25.0, text="Добар ден и вам."),
    SpeakerTurn(speaker="speaker_1", start=25.0, end=40.0, text="Да почнеме."),
]
SUMMARY_JSON = json.dumps({"topics": ["поздрав"], "summary": "Кратка размена.", "language": "mk"})


def build(
    matches=None,
    completer_responses=None,
    turns=TURNS,
    archive=None,
    speakers=None,
    metadata_store=None,
    knowledge_base=None,
):
    parts = {
        "archive": archive or FakeArchive(),
        "speakers": speakers or FakeSpeakerDirectory(matches or {}),
        "metadata_store": metadata_store or FakeMetadataStore(),
        "knowledge_base": knowledge_base or FakeKnowledgeBase(),
        "diarizer": FakeDiarizer(make_clustering("speaker_1", "speaker_2"), num_segments=2),
        "transcriber": FakeTranscriber(turns),
        "completer": FakeCompleter(completer_responses if completer_responses is not None else [SUMMARY_JSON]),
    }
    use_case = IngestTranscriptUseCase(
        speaker_similarity_threshold=0.75, language="mk", **parts
    )
    return use_case, parts


def request(**kwargs) -> IngestionRequest:
    return IngestionRequest(
        stem=STEM,
        session=SessionDetails(date="2024-01-01", time="10:00", show="Шоу", location="Скопје"),
        **kwargs,
    )


# --- the happy path, end to end ---

@pytest.mark.asyncio
async def test_ingest_produces_transcript_metadata_and_report(progress):
    use_case, _ = build(
        matches={1: SpeakerMatch(name="Ана", similarity=0.9), 2: SpeakerMatch(name="Бојан", similarity=0.8)},
    )
    result = await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    assert result.stem == STEM
    assert "Ана (0:00 - 0:10): Добар ден." in result.transcript_txt
    assert json.loads(result.metadata_json)["speakers"] == ["Ана", "Бојан"]
    assert "matched from DB" in result.resolution_report


@pytest.mark.asyncio
async def test_ingest_stores_audio_transcript_and_diarization(progress):
    use_case, parts = build(matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)})
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    archive = parts["archive"]
    assert archive.stored_audio == ["/tmp/audio.wav"]
    assert STEM in archive.stored_outputs
    assert archive.has_diarization(STEM)


@pytest.mark.asyncio
async def test_ingest_indexes_the_transcript_under_its_stem(progress):
    knowledge_base = FakeKnowledgeBase()
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)},
        knowledge_base=knowledge_base,
    )
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    transcript, doc_id, file_path, metadata = knowledge_base.inserted[0]
    assert doc_id == STEM
    assert file_path == f"transcripts/{STEM}/{STEM}.txt"
    assert "Ана" in transcript
    assert json.loads(metadata)["show"] == "Шоу"


@pytest.mark.asyncio
async def test_ingest_saves_metadata_under_the_same_id_as_the_index(progress):
    metadata_store = FakeMetadataStore()
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)},
        metadata_store=metadata_store,
    )
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    file_path, metadata = metadata_store.saved[STEM]
    assert file_path == f"transcripts/{STEM}/{STEM}.txt"
    assert metadata["speaker_count"] == 2


# --- speaker resolution ---

@pytest.mark.asyncio
async def test_unknown_speakers_are_named_by_the_llm(progress):
    identities = json.dumps({"speaker_2": "Бојан"})
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.9)},
        completer_responses=[identities, SUMMARY_JSON],
    )
    result = await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    assert json.loads(result.metadata_json)["speakers"] == ["Ана", "Бојан"]
    assert "identified by LLM" in result.resolution_report


@pytest.mark.asyncio
async def test_llm_step_is_skipped_when_every_speaker_matched(progress):
    use_case, _ = build(matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)})
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    assert "LLM step skipped" in progress


@pytest.mark.asyncio
async def test_matches_below_the_threshold_are_treated_as_unknown(progress):
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.5), 2: SpeakerMatch("Бојан", 0.6)},
        completer_responses=["{}", SUMMARY_JSON],
    )
    result = await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    assert json.loads(result.metadata_json)["speakers"] == ["speaker_1", "speaker_2"]


@pytest.mark.asyncio
async def test_named_speakers_are_fed_back_into_the_directory(progress):
    speakers = FakeSpeakerDirectory({1: SpeakerMatch("Ана", 0.9)})
    use_case, _ = build(
        completer_responses=[json.dumps({"speaker_2": "Бојан"}), SUMMARY_JSON],
        speakers=speakers,
    )
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    remembered = {entry["name"]: entry for entry in speakers.remembered}
    assert set(remembered) == {"Ана", "Бојан"}
    assert remembered["Ана"]["turns"] == 2
    assert remembered["Бојан"]["turns"] == 1


@pytest.mark.asyncio
async def test_unidentified_speakers_are_not_remembered(progress):
    speakers = FakeSpeakerDirectory()
    use_case, _ = build(completer_responses=["{}", SUMMARY_JSON], speakers=speakers)
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    assert speakers.remembered == []


@pytest.mark.asyncio
async def test_known_speaker_hints_reach_the_model(progress):
    use_case, parts = build(
        completer_responses=[json.dumps({"speaker_1": "Ана"}), SUMMARY_JSON]
    )
    await use_case.ingest_audio("/tmp/audio.wav", request(known_speakers=["Ана", "Бојан"]), progress)
    assert "Ана, Бојан" in parts["completer"].prompts[0]


# --- degradation ---

@pytest.mark.asyncio
async def test_a_failing_summary_does_not_fail_the_ingestion(progress):
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)},
        completer_responses=["not json"],
    )
    result = await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    metadata = json.loads(result.metadata_json)
    assert metadata["summary"] == ""
    assert metadata["topics"] == []


@pytest.mark.asyncio
async def test_a_failing_diarization_cache_upload_does_not_fail_the_ingestion(progress):
    archive = FakeArchive()

    def explode(stem, source_dir):
        raise RuntimeError("bucket unavailable")

    archive.store_diarization = explode
    use_case, _ = build(
        matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)}, archive=archive
    )
    result = await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    assert result.transcript_txt
    assert "NeMo caching failed (non-fatal)" in progress


# --- resume ---

@pytest.mark.asyncio
async def test_resume_reuses_cached_diarization_and_skips_re_diarizing(progress):
    archive = FakeArchive()
    archive.diarization.add(STEM)
    use_case, parts = build(
        matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)}, archive=archive
    )

    result = await use_case.resume(request(), progress)

    assert result.transcript_txt
    assert parts["diarizer"].diarized is False
    assert archive.stored_audio == []


@pytest.mark.asyncio
async def test_resume_without_cached_diarization_is_rejected(progress):
    use_case, _ = build()
    with pytest.raises(DiarizationNotCached):
        await use_case.resume(request(), progress)


@pytest.mark.asyncio
async def test_resume_without_archived_audio_is_rejected(progress):
    archive = FakeArchive()
    archive.diarization.add(STEM)
    archive.missing_audio.add(STEM)
    use_case, _ = build(archive=archive)
    with pytest.raises(AudioNotFound):
        await use_case.resume(request(), progress)


# --- progress reporting ---

@pytest.mark.asyncio
async def test_progress_walks_through_every_stage(progress):
    use_case, _ = build(matches={1: SpeakerMatch("Ана", 0.9), 2: SpeakerMatch("Бојан", 0.8)})
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)

    for stage in [
        "Step 1/3", "Step 2/3", "Step 3/3",
        "Clustering speakers", "Matching speakers", "Transcribing audio",
        "Generating metadata", "Uploading outputs", "Knowledge graph updated",
    ]:
        assert stage in progress


@pytest.mark.asyncio
async def test_progress_reports_speaker_counts(progress):
    use_case, _ = build(matches={1: SpeakerMatch("Ана", 0.9)}, completer_responses=["{}", SUMMARY_JSON])
    await use_case.ingest_audio("/tmp/audio.wav", request(), progress)
    assert "Found 2 speaker(s)" in progress
    assert "DB matching: 1/2 matched" in progress
