from __future__ import annotations

import re
from parsers.base import BaseParser
from models.models import SpeakerTurn

TURN_RE = re.compile(
    r"^\s*(?P<speaker>[^(\n]+?)\s*"
    r"\((?P<start>[\d:]+)\s*-\s*(?P<end>[\d:]+)\)\s*:\s*"
    r"(?P<text>.*)\s*$",
    re.MULTILINE,
)


class TranscriptParser(BaseParser):

    def parse(self, transcript: str) -> list[SpeakerTurn]:
        return [
            SpeakerTurn(
                speaker=self.normalize(match.group("speaker")),
                timestamp_start=match.group("start"),
                timestamp_end=match.group("end"),
                text=self.normalize(match.group("text")),
            )
            for match in TURN_RE.finditer(transcript or "")
        ]

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def format_speaker_turn(self, turn: SpeakerTurn) -> str:
        return f"{turn.speaker} ({turn.timestamp_start}-{turn.timestamp_end}): {turn.text}"

    def context_window(self, turns: list[SpeakerTurn], idx: int, window: int) -> tuple[str, str]:
        window = max(0, int(window))
        preceding_turns = "\n".join(self.format_speaker_turn(turns[j]) for j in range(max(0, idx - window), idx)).strip() or "None"
        following_turns = "\n".join(self.format_speaker_turn(turns[j]) for j in range(idx + 1, min(len(turns), idx + 1 + window))).strip() or "None"
        return preceding_turns, following_turns
