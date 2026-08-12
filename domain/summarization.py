"""Deriving topics and a summary from a transcript."""

from __future__ import annotations

import json
import logging

from domain.metadata import DEFAULT_LANGUAGE, TranscriptSummary
from domain.ports import TextCompleter
from domain.prompts.summary import (
    build_topics_and_summary_system_prompt,
    build_topics_and_summary_user_prompt,
)
from domain.transcript import Transcript

logger = logging.getLogger(__name__)

MAX_SUMMARY_TOKENS = 800


async def summarize(
    transcript: Transcript,
    speaker_names: dict[str, str],
    completer: TextCompleter,
    language: str = DEFAULT_LANGUAGE,
) -> TranscriptSummary:
    """Best effort: a missing summary must not fail the ingestion."""
    body = "\n".join(
        f"{speaker_names.get(turn.speaker, turn.speaker)}: {turn.text}" for turn in transcript.turns
    )
    try:
        response = await completer.complete(
            build_topics_and_summary_user_prompt(body),
            system=build_topics_and_summary_system_prompt(),
            json_object=True,
            temperature=0.3,
            max_tokens=MAX_SUMMARY_TOKENS,
        )
        return TranscriptSummary.from_payload(json.loads(response), language)
    except Exception as exc:
        logger.error("LLM summary generation failed: %s", exc)
        return TranscriptSummary(language=language)
