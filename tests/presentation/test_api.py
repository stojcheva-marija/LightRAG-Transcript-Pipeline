"""The HTTP and SSE contract the frontend depends on."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from application.ingest_transcript import IngestionResult
from application.query_transcripts import Answer, PlayableSource
from domain.errors import AudioNotFound, DiarizationNotCached
from presentation.api.app import create_app
from presentation.api.dependencies import get_container


@pytest.fixture
def container() -> MagicMock:
    container = MagicMock()
    container.catalog.list_stems.return_value = ["show_a"]
    container.catalog.audio_url.return_value = "https://audio.test/show_a.wav"
    return container


@pytest.fixture
def client(container) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    # No lifespan: the container is injected, so nothing needs to be started.
    return TestClient(app)


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.strip().split("\n")
        name = next(l[len("event: "):] for l in lines if l.startswith("event: "))
        data = next(l[len("data: "):] for l in lines if l.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


# --- query ---

def test_query_returns_answer_and_sources(client, container):
    async def ask(question, history=None):
        return Answer(
            text="Одговорот.",
            sources=[PlayableSource("show_a", 90.0, "https://audio.test/show_a.wav")],
        )

    container.query.ask = ask
    response = client.post("/api/query", json={"question": "Прашање?", "history": []})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Одговорот."
    assert body["source_stems"] == [
        {"stem": "show_a", "start_seconds": 90.0, "audio_url": "https://audio.test/show_a.wav"}
    ]


def test_empty_question_is_rejected(client):
    response = client.post("/api/query", json={"question": "   "})
    assert response.status_code == 400


# --- transcripts ---

def test_list_transcripts(client):
    response = client.get("/api/transcripts")
    assert response.status_code == 200
    assert response.json() == {"stems": ["show_a"]}


def test_audio_url(client):
    response = client.get("/api/transcripts/show_a/audio")
    assert response.status_code == 200
    assert response.json() == {"url": "https://audio.test/show_a.wav"}


def test_audio_url_missing_returns_404(client, container):
    container.catalog.audio_url.side_effect = AudioNotFound("ghost")
    response = client.get("/api/transcripts/ghost/audio")
    assert response.status_code == 404


# --- pipeline streaming ---

def test_pipeline_streams_progress_then_the_result(client, container):
    async def ingest_audio(audio_path, request, progress):
        progress.report("Step 1/3: Saving audio to MinIO...")
        progress.report("Knowledge graph updated")
        return IngestionResult(
            stem=request.stem,
            transcript_txt="Ана (0:00 - 0:10): Здраво.",
            metadata_json='{"date": "2024-01-01"}',
            resolution_report="## Speaker Resolution Report",
        )

    container.ingest.ingest_audio = ingest_audio

    response = client.post(
        "/api/pipeline/run",
        files={"audio_file": ("sednica_77.wav", b"audio-bytes", "audio/wav")},
        data={"date": "2024-01-01", "show_name": "Шоу", "known_speakers": "Ана, Бојан"},
    )

    assert response.status_code == 200
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["log", "log", "done"]
    assert events[0][1] == {"message": "Step 1/3: Saving audio to MinIO..."}
    assert set(events[-1][1]) == {"metadata", "transcript", "resolution_report"}
    assert events[-1][1]["transcript"] == "Ана (0:00 - 0:10): Здраво."


def test_pipeline_derives_the_stem_and_speakers_from_the_upload(client, container):
    captured = {}

    async def ingest_audio(audio_path, request, progress):
        captured["request"] = request
        return IngestionResult(request.stem, "", "{}", "")

    container.ingest.ingest_audio = ingest_audio

    client.post(
        "/api/pipeline/run",
        files={"audio_file": ("sednica_77.wav", b"audio", "audio/wav")},
        data={"date": "2024-01-01", "show_name": "Шоу", "known_speakers": "Ана, Бојан , "},
    )

    request = captured["request"]
    assert request.stem == "sednica_77"
    assert request.known_speakers == ["Ана", "Бојан"]
    assert request.session.date == "2024-01-01"
    assert request.session.show == "Шоу"


def test_pipeline_failure_is_streamed_as_an_error_event(client, container):
    async def ingest_audio(audio_path, request, progress):
        raise RuntimeError("diarization exploded")

    container.ingest.ingest_audio = ingest_audio

    response = client.post(
        "/api/pipeline/run",
        files={"audio_file": ("x.wav", b"audio", "audio/wav")},
    )

    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert "diarization exploded" in events[-1][1]["message"]


def test_resume_without_cached_diarization_streams_an_error(client, container):
    async def resume(request, progress):
        raise DiarizationNotCached(request.stem)

    container.ingest.resume = resume

    response = client.post("/api/pipeline/resume", json={"stem": "ghost"})

    events = parse_events(response.text)
    assert events[-1][0] == "error"
    assert "No cached NeMo outputs for 'ghost'" in events[-1][1]["message"]
