"""Document identity: how a recording's stem maps to keys, ids and citations.

A transcript is identified end to end by its *stem* (the audio filename without
extension). The stem is the document id in the knowledge base, the folder name
in the archive, and what a citation resolves back to.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def transcript_key(stem: str) -> str:
    """Archive key of the stored transcript text for ``stem``."""
    return f"transcripts/{stem}/{stem}.txt"


def normalize_stem(stem: str) -> str:
    """Lowercase and drop non-alphanumerics, for tolerant comparison."""
    return _NON_ALNUM.sub("", stem.lower())


def resolve_stem(cited_stem: str, known_stems: list[str]) -> str | None:
    """Match a stem cited by the knowledge base against the archive's stems.

    Citations can carry a suffix the archive does not use, so a known stem
    matches when it is a prefix of the citation once both are normalized.
    """
    normalized = normalize_stem(cited_stem)
    for known in known_stems:
        if normalized.startswith(normalize_stem(known)):
            return known
    return None
