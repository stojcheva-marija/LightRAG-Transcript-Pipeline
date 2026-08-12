"""Reading source citations back out of a LightRAG context string."""

from __future__ import annotations

import re
import logging

from domain.retrieval import SourceReference

logger = logging.getLogger(__name__)

_TIME_RE = re.compile(r'\[TIMESTAMP=([\d:]+)-([\d:]+)\]')

# Matches {"reference_id": "N", "content": "...[TIMESTAMP=start-end]..."}
# [^{]* keeps us within one JSON chunk object (chunks start with '{')
_CHUNK_REF_TS_RE = re.compile(
    r'"reference_id":\s*"(\d+)"[^{]*?\[TIMESTAMP=([\d:]+)-([\d:]+)\]'
)

# Matches the Reference Document List at the end of the context:
# [1] transcripts/stem/stem.txt  (any separator between [N] and the path)
_REF_DOC_RE = re.compile(r'\[(\d+)\][^\n]*transcripts/([^/\s]+)/\2\.txt')

# Fallback for newly-indexed chunks that embed [SOURCE=stem][TIMESTAMP=...]
_SOURCE_TS_RE = re.compile(
    r'\[SOURCE=([^\]]+)\]'
    r'(?:\s*\[[^\]]+\])*'
    r'\s*\[TIMESTAMP=([\d:]+)-([\d:]+)\]'
)


def _timestamp_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        return 0.0
    except Exception:
        return 0.0


def parse_source_references(context_str: str) -> list[SourceReference]:
    all_results: list[SourceReference] = []
    seen_pairs: set[tuple[str, float]] = set()

    # Primary: LightRAG context format — reference_id in each chunk + Reference Document List
    ref_to_stem: dict[str, str] = {
        m.group(1): m.group(2)
        for m in _REF_DOC_RE.finditer(context_str)
    }
    if ref_to_stem:
        for m in _CHUNK_REF_TS_RE.finditer(context_str):
            stem = ref_to_stem.get(m.group(1))
            if not stem:
                continue
            _add(all_results, seen_pairs, stem, _timestamp_to_seconds(m.group(2)))

    # Fallback: [SOURCE=stem] tag embedded in the chunk text itself
    if not all_results:
        for m in _SOURCE_TS_RE.finditer(context_str):
            _add(all_results, seen_pairs, m.group(1), _timestamp_to_seconds(m.group(2)))

    logger.info("Source stems with timestamps: %s", all_results)
    return all_results


def _add(
    results: list[SourceReference],
    seen: set[tuple[str, float]],
    stem: str,
    seconds: float,
) -> None:
    key = (stem, seconds)
    if key not in seen:
        seen.add(key)
        results.append(SourceReference(stem=stem, start_seconds=seconds))


def parse_context_chunks(context_str: str) -> list[str]:
    chunks = [c.strip() for c in context_str.split("\n\n") if c.strip()]
    return chunks if chunks else [context_str]
