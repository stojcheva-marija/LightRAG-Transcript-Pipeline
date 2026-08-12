"""Pulling the retrievable facts out of a user's question.

Speaker names and dates are what the knowledge base indexes most reliably, so
they are extracted up front and passed to retrieval as keywords.
"""

from __future__ import annotations

import json
import logging
import re

from domain.ports import TextCompleter
from domain.prompts.keywords import build_keyword_extraction_prompt
from domain.retrieval import QueryKeywords

logger = logging.getLogger(__name__)

# Accepted date shapes: MM, MM-DD, YYYY-MM-DD.
_DATE_LENGTHS = {2, 5, 10}
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


async def extract_keywords(
    question: str,
    completer: TextCompleter,
    known_speakers: list[str],
) -> QueryKeywords:
    """Extracts keywords, or returns none if the LLM response can't be used."""
    raw = None
    try:
        raw = await completer.complete(build_keyword_extraction_prompt(question, known_speakers))
        match = _JSON_OBJECT.search(raw or "")
        if not match:
            raise ValueError(f"No JSON found in response: {raw!r}")
        data = json.loads(match.group())
        return QueryKeywords(
            speakers=data.get("speakers", []),
            dates=[d for d in data.get("dates", []) if len(d) in _DATE_LENGTHS],
        )
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s | raw resp: %r", exc, raw)
        return QueryKeywords.none()
