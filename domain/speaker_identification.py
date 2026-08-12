"""Identifies speakers the voice directory could not, from the transcript text."""

from __future__ import annotations

import json
import logging

from domain.ports import TextCompleter
from domain.prompts.speaker_identification import (
    build_speaker_identification_system_prompt,
    build_speaker_identification_user_prompt,
)
from domain.transcript import Transcript

logger = logging.getLogger(__name__)


async def identify_speakers(
    transcript: Transcript,
    unknown_speaker_ids: list[str],
    hints: list[str],
    completer: TextCompleter,
) -> dict[str, str]:
    """Return ``{speaker_id: name}`` for the clusters the model could name."""
    if not unknown_speaker_ids:
        return {}

    try:
        response = await completer.complete(
            build_speaker_identification_user_prompt(
                hints=hints,
                excerpt=transcript.excerpt(),
                unknown_ids=unknown_speaker_ids,
            ),
            system=build_speaker_identification_system_prompt(),
            json_object=True,
            temperature=0.1,
        )
        identities = json.loads(response.strip())
    except Exception as exc:
        logger.error("LLM identity inference failed: %s", exc)
        return {}

    return {
        speaker_id: name
        for speaker_id, name in identities.items()
        if isinstance(name, str) and name.strip()
    }
