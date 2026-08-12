"""Browsing what has already been ingested."""

from __future__ import annotations

from domain.ports import TranscriptArchive


class TranscriptCatalog:

    def __init__(self, archive: TranscriptArchive) -> None:
        self.archive = archive

    def list_stems(self) -> list[str]:
        return self.archive.list_stems()

    def audio_url(self, stem: str) -> str:
        """Raises ``AudioNotFound`` when the recording has no audio stored."""
        return self.archive.audio_url(stem)
