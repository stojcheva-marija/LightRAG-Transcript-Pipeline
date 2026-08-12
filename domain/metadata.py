"""Data classes for a transcript's date, speakers, topics, and other metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from domain.transcript import Transcript

# Fallback only — the configured language wins wherever one is available.
DEFAULT_LANGUAGE = "mk"


def metadata_header(metadata_json: str) -> str:
    """The ``[DATE=…][SHOW=…]`` preamble prepended to every indexed chunk.

    Retrieval answers temporal questions off these tags, so they travel with
    the chunk text rather than living only in the metadata store.
    """
    if not metadata_json:
        return ""
    try:
        metadata = json.loads(metadata_json)
    except Exception:
        return ""
    parts = [
        f"[DATE={metadata['date']}]" if "date" in metadata else "",
        f"[SHOW={metadata['show']}]" if "show" in metadata else "",
    ]
    parts = [part for part in parts if part]
    return "\n".join(parts) + "\n" if parts else ""


@dataclass(frozen=True)
class SessionDetails:
    """Date, time, show, and location entered when uploading a recording."""

    date: str = ""
    time: str = ""
    show: str = ""
    location: str = ""


@dataclass(frozen=True)
class TranscriptSummary:
    """LLM-derived description of what was discussed."""

    topics: list[str] = field(default_factory=list)
    summary: str = ""
    language: str = DEFAULT_LANGUAGE

    @classmethod
    def empty(cls) -> TranscriptSummary:
        return cls()

    @classmethod
    def from_payload(
        cls, payload: dict | None, default_language: str = DEFAULT_LANGUAGE
    ) -> TranscriptSummary:
        payload = payload or {}
        return cls(
            topics=payload.get("topics", []),
            summary=payload.get("summary", ""),
            language=payload.get("language") or default_language,
        )


@dataclass(frozen=True)
class TranscriptMetadata:
    """The metadata document stored next to a transcript and indexed in Postgres."""

    date: str
    time: str
    duration_seconds: int
    speakers: list[str]
    topics: list[str]
    language: str
    show: str
    location: str
    summary: str

    @classmethod
    def build(
        cls,
        transcript: Transcript,
        speaker_names: dict[str, str],
        session: SessionDetails,
        summary: TranscriptSummary,
    ) -> TranscriptMetadata:
        return cls(
            date=session.date,
            time=session.time,
            duration_seconds=transcript.duration_seconds,
            speakers=transcript.speaker_names(speaker_names),
            topics=summary.topics,
            language=summary.language,
            show=session.show,
            location=session.location,
            summary=summary.summary,
        )

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "time": self.time,
            "duration_seconds": self.duration_seconds,
            "speakers": self.speakers,
            "speaker_count": len(self.speakers),
            "topics": self.topics,
            "language": self.language,
            "show": self.show,
            "location": self.location,
            "summary": self.summary,
        }
