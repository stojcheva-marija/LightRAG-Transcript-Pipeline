from __future__ import annotations

from domain.document import normalize_stem, resolve_stem, transcript_key


def test_transcript_key_layout():
    assert transcript_key("sednica_77") == "transcripts/sednica_77/sednica_77.txt"


def test_normalize_drops_case_and_punctuation():
    assert normalize_stem("Sednica_77-A") == "sednica77a"


def test_resolve_exact_stem():
    assert resolve_stem("sednica_77", ["sednica_77", "sednica_68"]) == "sednica_77"


def test_resolve_ignores_separators_and_case():
    assert resolve_stem("Sednica-77", ["sednica_77"]) == "sednica_77"


def test_resolve_matches_stem_with_suffix():
    assert resolve_stem("sednica_77_chunk_3", ["sednica_77"]) == "sednica_77"


def test_resolve_returns_none_when_nothing_matches():
    assert resolve_stem("unrelated", ["sednica_77"]) is None


def test_resolve_against_empty_archive():
    assert resolve_stem("sednica_77", []) is None
