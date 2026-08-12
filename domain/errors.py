from __future__ import annotations


class DomainError(Exception):
    """Base class for expected, meaningful failures."""


class TranscriptNotFound(DomainError):
    def __init__(self, stem: str) -> None:
        super().__init__(f"No transcript found for stem '{stem}'.")
        self.stem = stem


class AudioNotFound(DomainError):
    def __init__(self, stem: str) -> None:
        super().__init__(f"No audio found for stem '{stem}'.")
        self.stem = stem


class DiarizationNotCached(DomainError):
    def __init__(self, stem: str) -> None:
        super().__init__(
            f"No cached NeMo outputs for '{stem}'. Run the full pipeline first."
        )
        self.stem = stem
