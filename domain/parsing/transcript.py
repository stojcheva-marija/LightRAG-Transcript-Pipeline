from __future__ import annotations

import re

from domain.parsing.base import BaseParser
from domain.transcript import TranscriptLine

LINE_RE = re.compile(
    r"^\s*(?P<speaker>[^(\n]+?)\s*"
    r"\((?P<start>[\d:]+)\s*-\s*(?P<end>[\d:]+)\)\s*:\s*"
    r"(?P<text>.*)\s*$",
    re.MULTILINE,
)


class TranscriptParser(BaseParser):
    """Reads the ``Speaker (start - end): text`` format written by ``Transcript.render``."""

    def parse(self, transcript: str) -> list[TranscriptLine]:
        return [
            TranscriptLine(
                speaker=self.normalize(match.group("speaker")),
                timestamp_start=match.group("start"),
                timestamp_end=match.group("end"),
                text=self.normalize(match.group("text")),
            )
            for match in LINE_RE.finditer(transcript or "")
        ]

    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def format_line(self, line: TranscriptLine) -> str:
        return f"{line.speaker} ({line.timestamp_start}-{line.timestamp_end}): {line.text}"

    def context_window(self, lines: list[TranscriptLine], idx: int, window: int) -> tuple[str, str]:
        window = max(0, int(window))
        preceding = "\n".join(
            self.format_line(lines[j]) for j in range(max(0, idx - window), idx)
        ).strip() or "None"
        following = "\n".join(
            self.format_line(lines[j]) for j in range(idx + 1, min(len(lines), idx + 1 + window))
        ).strip() or "None"
        return preceding, following
