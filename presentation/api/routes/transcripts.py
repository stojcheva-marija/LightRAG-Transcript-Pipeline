from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from composition import Container
from domain.errors import AudioNotFound
from presentation.api.dependencies import get_container
from presentation.api.schemas import AudioUrlResponse, TranscriptListResponse

router = APIRouter()


@router.get("/transcripts", response_model=TranscriptListResponse)
async def list_transcripts(container: Container = Depends(get_container)):
    return TranscriptListResponse(stems=container.catalog.list_stems())


@router.get("/transcripts/{stem}/audio", response_model=AudioUrlResponse)
async def get_audio_url(stem: str, container: Container = Depends(get_container)):
    try:
        return AudioUrlResponse(url=container.catalog.audio_url(stem))
    except AudioNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
