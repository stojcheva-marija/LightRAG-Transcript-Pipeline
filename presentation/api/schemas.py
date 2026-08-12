"""HTTP request/response shapes, and how they map to/from application types."""

from __future__ import annotations

from pydantic import BaseModel

from application.ingest_transcript import IngestionRequest
from application.query_transcripts import Answer
from domain.metadata import SessionDetails


class QueryRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []


class SourceStem(BaseModel):
    stem: str
    start_seconds: float
    audio_url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    source_stems: list[SourceStem]

    @classmethod
    def from_answer(cls, answer: Answer) -> QueryResponse:
        return cls(
            answer=answer.text,
            source_stems=[
                SourceStem(
                    stem=source.stem,
                    start_seconds=source.start_seconds,
                    audio_url=source.audio_url,
                )
                for source in answer.sources
            ],
        )


class RecordingDetails(BaseModel):
    """Recording details a client sends along with the audio."""

    date: str = ""
    time: str = ""
    show_name: str = ""
    location: str = ""
    known_speakers: str = ""

    def to_ingestion_request(self, stem: str) -> IngestionRequest:
        return IngestionRequest(
            stem=stem,
            session=SessionDetails(
                date=self.date,
                time=self.time,
                show=self.show_name,
                location=self.location,
            ),
            known_speakers=[s.strip() for s in self.known_speakers.split(",") if s.strip()],
        )


class ResumeParams(RecordingDetails):
    stem: str


class TranscriptListResponse(BaseModel):
    stems: list[str]


class AudioUrlResponse(BaseModel):
    url: str
