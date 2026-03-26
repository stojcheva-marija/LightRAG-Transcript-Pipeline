from dataclasses import dataclass

@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    timestamp_start: str
    timestamp_end: str
    text: str

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    speaker: str
    timestamp_start: str
    timestamp_end: str
    text: str

    @property
    def formatted_text(self) -> str:
        return (
            f"[SPEAKER={self.speaker}]\n"
            f"[TIMESTAMP={self.timestamp_start}-{self.timestamp_end}]\n\n"
            f"{self.text}\n"
        )