"""Answering a question, with playable citations back into the recordings."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from domain.document import resolve_stem
from domain.keywords import extract_keywords
from domain.ports import KnowledgeBase, TextCompleter, TranscriptArchive, TranscriptMetadataStore

logger = logging.getLogger(__name__)

MAX_SOURCES = 3


@dataclass(frozen=True)
class PlayableSource:
    stem: str
    start_seconds: float
    audio_url: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[PlayableSource] = field(default_factory=list)


class QueryTranscriptsUseCase:

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        archive: TranscriptArchive,
        metadata_store: TranscriptMetadataStore,
        completer: TextCompleter,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.archive = archive
        self.metadata_store = metadata_store
        self.completer = completer

    async def ask(self, question: str, history: list[dict[str, str]] | None = None) -> Answer:
        known_speakers = self.metadata_store.known_speaker_names()
        keywords = await extract_keywords(question, self.completer, known_speakers)
        logger.info("keywords speakers=%s dates=%s", keywords.speakers, keywords.dates)

        retrieved = await self.knowledge_base.query(question, keywords, history)
        return Answer(
            text=retrieved.clean_answer,
            sources=self._playable_sources(retrieved.sources),
        )

    def _playable_sources(self, references) -> list[PlayableSource]:
        """Match citations to archived recordings; drop anything we cannot play."""
        archived_stems = self.archive.list_stems()
        url_cache: dict[str, str | None] = {}
        sources: list[PlayableSource] = []

        for reference in references:
            stem = resolve_stem(reference.stem, archived_stems) or reference.stem
            if stem not in url_cache:
                url_cache[stem] = self._audio_url(stem)
            url = url_cache[stem]
            if url is None:
                continue
            sources.append(PlayableSource(
                stem=stem, start_seconds=reference.start_seconds, audio_url=url
            ))
            if len(sources) == MAX_SOURCES:
                break

        return sources

    def _audio_url(self, stem: str) -> str | None:
        try:
            return self.archive.audio_url(stem)
        except Exception as exc:
            logger.warning("Could not get audio URL for stem '%s': %s", stem, exc)
            return None
