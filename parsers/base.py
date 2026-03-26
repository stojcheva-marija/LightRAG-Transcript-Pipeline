from __future__ import annotations

from abc import ABC, abstractmethod
from models.models import SpeakerTurn


class BaseParser(ABC):

    @abstractmethod
    def parse(self, transcript: str) -> list[SpeakerTurn]: ...

    @abstractmethod
    def normalize(self, text: str) -> str: ...

    @abstractmethod
    def format_speaker_turn(self, turn: SpeakerTurn) -> str: ...

    @abstractmethod
    def context_window(self, turns: list[SpeakerTurn], idx: int, window: int) -> tuple[str, str]: ...
