"""Speaker identity: clustering results and how a cluster is named.

Precedence rule: an embedding match against the speaker directory constitutes
evidence, whereas an LLM inference does not; a database match therefore always
takes priority. A cluster that cannot be named retains its cluster id.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ResolutionSource(str, Enum):
    DATABASE = "db"
    LLM = "llm"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SpeakerMatch:
    """A hit from the speaker directory."""

    name: str
    similarity: float


@dataclass(frozen=True)
class SpeakerClustering:
    """Diarization output: which cluster each segment belongs to, and its centroid."""

    remap: dict[tuple[str, str], str]
    centroids: dict[str, np.ndarray]

    @property
    def speaker_ids(self) -> list[str]:
        return sorted(self.centroids)

    def __len__(self) -> int:
        return len(self.centroids)


@dataclass(frozen=True)
class SpeakerResolution:
    """How one diarized cluster came to be named."""

    speaker_id: str
    name: str
    source: ResolutionSource
    similarity: float | None = None

    @classmethod
    def matched(cls, speaker_id: str, match: SpeakerMatch) -> SpeakerResolution:
        return cls(
            speaker_id=speaker_id,
            name=match.name,
            source=ResolutionSource.DATABASE,
            similarity=round(match.similarity, 4),
        )

    @classmethod
    def unknown(cls, speaker_id: str) -> SpeakerResolution:
        return cls(speaker_id=speaker_id, name=speaker_id, source=ResolutionSource.UNKNOWN)

    @property
    def is_resolved(self) -> bool:
        return self.source is not ResolutionSource.UNKNOWN


@dataclass(frozen=True)
class SpeakerResolutions:
    """All resolutions for one recording."""

    items: dict[str, SpeakerResolution]

    @classmethod
    def empty(cls) -> SpeakerResolutions:
        return cls(items={})

    def __iter__(self):
        return iter(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    @property
    def unresolved_ids(self) -> list[str]:
        return [r.speaker_id for r in self.items.values() if not r.is_resolved]

    @property
    def matched_count(self) -> int:
        return sum(1 for r in self.items.values() if r.source is ResolutionSource.DATABASE)

    def with_llm_identities(self, identities: dict[str, str]) -> SpeakerResolutions:
        """Name still-unknown clusters from LLM guesses. Directory matches win."""
        updated = dict(self.items)
        for speaker_id, name in identities.items():
            current = updated.get(speaker_id)
            if current is None or current.is_resolved or not name:
                continue
            updated[speaker_id] = SpeakerResolution(
                speaker_id=speaker_id, name=name, source=ResolutionSource.LLM
            )
        return SpeakerResolutions(items=updated)

    def name_mapping(self) -> dict[str, str]:
        """Cluster id -> display name, for every cluster."""
        return {speaker_id: r.name for speaker_id, r in self.items.items()}

    def report(self) -> str:
        """Human-readable audit of how each speaker was named."""
        lines = ["## Speaker Resolution Report", ""]
        for speaker_id in sorted(self.items):
            resolution = self.items[speaker_id]
            if resolution.source is ResolutionSource.DATABASE:
                lines.append(
                    f"**{speaker_id}** → `{resolution.name}` matched from DB "
                    f"(cosine similarity: {resolution.similarity:.3f})"
                )
            elif resolution.source is ResolutionSource.LLM:
                lines.append(f"**{speaker_id}** → `{resolution.name}` identified by LLM")
            else:
                lines.append(
                    f"**{speaker_id}** → `{resolution.name}` unknown (not in DB, not identified)"
                )
        return "\n".join(lines)
