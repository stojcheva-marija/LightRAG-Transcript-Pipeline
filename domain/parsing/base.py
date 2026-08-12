from __future__ import annotations

from abc import ABC, abstractmethod

from domain.transcript import TranscriptLine


class BaseParser(ABC):
    """Reads a stored transcript back into lines, and formats lines for prompts."""

    @abstractmethod
    def parse(self, transcript: str) -> list[TranscriptLine]: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    @abstractmethod
    def format_line(self, line: TranscriptLine) -> str: ...

    @abstractmethod
    def context_window(self, lines: list[TranscriptLine], idx: int, window: int) -> tuple[str, str]: ...
