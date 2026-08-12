"""Core transcript concepts.

Two representations of a spoken turn exist on purpose:

* ``SpeakerTurn`` is what diarization + transcription produce — offsets in
  seconds, speaker still identified by its cluster id.
* ``TranscriptLine`` is what parsing a *stored* transcript yields — display
  timestamps as written in the file, speaker already resolved to a name.

``Transcript.render`` is the single bridge between them: it is the format the
parser later reads back.
"""

from __future__ import annotations

from dataclasses import dataclass


def format_timestamp(seconds: float) -> str:
    """Render seconds as ``M:SS`` (or ``H:MM:SS`` past the hour)."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


@dataclass(frozen=True)
class SpeakerTurn:
    """One uninterrupted stretch of speech, as produced by ingestion."""

    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


# Two stretches of the same speaker separated by less than this are one turn:
# the gap is a breath, not a handover.
MERGE_GAP_SECONDS = 2.0


def merge_adjacent_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    """Join consecutive turns by the same speaker that are only a breath apart."""
    merged: list[SpeakerTurn] = []
    for turn in turns:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.speaker == turn.speaker
            and turn.start - previous.end < MERGE_GAP_SECONDS
        ):
            merged[-1] = SpeakerTurn(
                speaker=previous.speaker,
                start=previous.start,
                end=turn.end,
                text=f"{previous.text} {turn.text}",
            )
        else:
            merged.append(turn)
    return merged


@dataclass(frozen=True)
class Transcript:
    """An ordered set of speaker turns, plus the rules that read them."""

    turns: list[SpeakerTurn]

    def __len__(self) -> int:
        return len(self.turns)

    def __bool__(self) -> bool:
        return bool(self.turns)

    @property
    def duration_seconds(self) -> int:
        if not self.turns:
            return 0
        return int(self.turns[-1].end - self.turns[0].start)

    def turn_count_for(self, speaker_id: str) -> int:
        return sum(1 for turn in self.turns if turn.speaker == speaker_id)

    def speaker_names(self, names: dict[str, str]) -> list[str]:
        """Resolved speaker names in order of first appearance, deduplicated."""
        return list(dict.fromkeys(names.get(turn.speaker, turn.speaker) for turn in self.turns))

    def render(self, names: dict[str, str]) -> str:
        """The stored ``.txt`` format — must stay readable by ``TranscriptParser``."""
        return "\n\n".join(
            f"{names.get(turn.speaker, turn.speaker)} "
            f"({format_timestamp(turn.start)} - {format_timestamp(turn.end)}): {turn.text}"
            for turn in self.turns
        )

    def excerpt(self) -> str:
        """Plain ``speaker: text`` lines, for prompting an LLM about the content."""
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)


@dataclass(frozen=True)
class TranscriptLine:
    """One line parsed back out of a stored transcript."""

    speaker: str
    timestamp_start: str
    timestamp_end: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of a transcript, as indexed in the knowledge base."""

    chunk_id: str
    doc_id: str
    speaker: str
    timestamp_start: str
    timestamp_end: str
    text: str

    @property
    def formatted_text(self) -> str:
        return (
            f"[SOURCE={self.doc_id}]\n"
            f"[SPEAKER={self.speaker}]\n"
            f"[TIMESTAMP={self.timestamp_start}-{self.timestamp_end}]\n\n"
            f"{self.text}\n"
        )
