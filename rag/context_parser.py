from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

_FILE_PATH_RE = re.compile(r'transcripts/([^/]+)/\1\.txt')
_TIME_RE = re.compile(r'\[TIMESTAMP=(\d+:\d+)-\d+:\d+\]')


def _timestamp_to_seconds(ts: str) -> float:
    try:
        minutes, seconds = ts.split(":")
        return int(minutes) * 60 + int(seconds)
    except Exception:
        return 0.0


def parse_source_stems(context_str: str) -> list[tuple[str, float]]:
    stems = list(dict.fromkeys(_FILE_PATH_RE.findall(context_str)))
    time_matches = _TIME_RE.findall(context_str)
    start_seconds = _timestamp_to_seconds(time_matches[0]) if time_matches else 0.0
    logger.info("Source stems: %s, first chunk start: %.2fs", stems, start_seconds)
    return [(stem, start_seconds) for stem in stems]


def parse_context_chunks(context_str: str) -> list[str]:
    chunks = [c.strip() for c in context_str.split("\n\n") if c.strip()]
    return chunks if chunks else [context_str]
