"""Upload and resume routes for ingestion — both stream progress over SSE."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from application.ingest_transcript import IngestionResult
from composition import Container
from presentation.api.dependencies import get_container
from presentation.api.schemas import RecordingDetails, ResumeParams
from presentation.api.sse import stream_use_case

router = APIRouter()


def _payload(result: IngestionResult) -> dict:
    return {
        "metadata": result.metadata_json,
        "transcript": result.transcript_txt,
        "resolution_report": result.resolution_report,
    }


@router.post("/pipeline/run")
async def run_pipeline(
    audio_file: UploadFile = File(...),
    date: str = Form(""),
    time: str = Form(""),
    show_name: str = Form(""),
    location: str = Form(""),
    known_speakers: str = Form(""),
    container: Container = Depends(get_container),
):
    upload_dir = tempfile.mkdtemp(prefix="upload_")
    audio_path = os.path.join(upload_dir, audio_file.filename)
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    details = RecordingDetails(
        date=date, time=time, show_name=show_name,
        location=location, known_speakers=known_speakers,
    )
    request = details.to_ingestion_request(Path(audio_file.filename).stem)

    async def run(progress):
        return _payload(await container.ingest.ingest_audio(audio_path, request, progress))

    return stream_use_case(run, on_finish=lambda: shutil.rmtree(upload_dir, ignore_errors=True))


@router.post("/pipeline/resume")
async def resume_pipeline(
    params: ResumeParams,
    container: Container = Depends(get_container),
):
    request = params.to_ingestion_request(params.stem)

    async def run(progress):
        return _payload(await container.ingest.resume(request, progress))

    return stream_use_case(run)
