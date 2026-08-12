"""The answer from asking the knowledge base a question."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# The knowledge base appends its own reference list; we cite sources ourselves.
_TRAILING_REFERENCES = re.compile(r"\n+#+\s*References.*", re.DOTALL)


@dataclass(frozen=True)
class QueryKeywords:
    """Retrieval hints pulled out of a question."""

    speakers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)

    @classmethod
    def none(cls) -> QueryKeywords:
        return cls()


@dataclass(frozen=True)
class SourceReference:
    """A point in a recording that supports the answer."""

    stem: str
    start_seconds: float


@dataclass(frozen=True)
class RetrievedAnswer:
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    context_chunks: list[str] = field(default_factory=list)

    @property
    def clean_answer(self) -> str:
        return _TRAILING_REFERENCES.sub("", self.answer).strip()
