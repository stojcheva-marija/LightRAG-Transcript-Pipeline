"""Interfaces for external dependencies (LLM, storage, database, ASR, knowledge base).

Real implementations live in ``infrastructure/`` and get wired in ``composition.py``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from domain.retrieval import QueryKeywords, RetrievedAnswer
from domain.speaker import SpeakerClustering, SpeakerMatch
from domain.transcript import SpeakerTurn


@runtime_checkable
class TextCompleter(Protocol):
    """A chat LLM. The one way anything in this system talks to a model."""

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_object: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...


class TranscriptArchive(Protocol):
    """Object storage holding audio, transcripts and cached diarization output."""

    def list_stems(self) -> list[str]: ...

    def store_audio(self, local_path: str) -> str: ...

    def store_outputs(self, stem: str, transcript_txt: str, metadata_json: str) -> None: ...

    def load_transcript(self, stem: str) -> tuple[str, str]:
        """Return ``(transcript_text, metadata_json)``; metadata may be empty."""

    def download_audio(self, stem: str, target_dir: str) -> str: ...

    def audio_url(self, stem: str, expires_in: int = 3600) -> str: ...

    def store_diarization(self, stem: str, source_dir: str) -> None: ...

    def has_diarization(self, stem: str) -> bool: ...

    def fetch_diarization(self, stem: str, target_dir: str) -> str: ...


class TranscriptMetadataStore(Protocol):
    """Where transcript metadata is indexed for lookup."""

    def setup(self) -> None: ...

    def save(self, doc_id: str, file_path: str, metadata: dict) -> None: ...

    def known_speaker_names(self) -> list[str]: ...


class SpeakerDirectory(Protocol):
    """Known speakers and their voice embeddings."""

    def setup(self) -> None: ...

    def match(self, embedding: np.ndarray, threshold: float) -> SpeakerMatch | None: ...

    def remember(
        self,
        name: str,
        embedding: np.ndarray,
        notes: str = "",
        files: int = 0,
        turns: int = 0,
    ) -> None: ...


class KnowledgeBase(Protocol):
    """The retrieval-augmented index transcripts are inserted into and queried from."""

    async def initialize(self) -> None: ...

    async def insert_document(
        self, transcript: str, doc_id: str, file_path: str, metadata: str = ""
    ) -> bool: ...

    async def query(
        self,
        question: str,
        keywords: QueryKeywords,
        history: list[dict[str, str]] | None = None,
    ) -> RetrievedAnswer: ...

    async def finalize(self) -> None: ...


class AudioDiarizer(Protocol):
    """Splits audio, works out who spoke when, and groups voices into speakers."""

    def split(self, audio_path: str, work_dir: str) -> int:
        """Split into segments under ``work_dir``; return the segment count."""

    def diarize(self, work_dir: str) -> str:
        """Run diarization over the segments; return the output directory."""

    def cluster(self, diarization_dir: str) -> SpeakerClustering: ...


class SpeechTranscriber(Protocol):
    """Turns diarized audio segments into speaker turns."""

    def transcribe(
        self,
        work_dir: str,
        diarization_dir: str,
        clustering: SpeakerClustering,
        num_segments: int,
    ) -> list[SpeakerTurn]: ...
