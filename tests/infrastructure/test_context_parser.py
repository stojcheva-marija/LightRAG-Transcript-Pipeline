from __future__ import annotations

from infrastructure.rag.context_parser import (
    _timestamp_to_seconds,
    parse_context_chunks,
    parse_source_references,
)


# --- _timestamp_to_seconds ---

def test_timestamp_to_seconds_basic():
    assert _timestamp_to_seconds("1:30") == 90.0

def test_timestamp_to_seconds_zero():
    assert _timestamp_to_seconds("0:00") == 0.0

def test_timestamp_to_seconds_hours():
    assert _timestamp_to_seconds("1:00:00") == 3600.0

def test_timestamp_to_seconds_hours_expressed_as_minutes():
    assert _timestamp_to_seconds("90:00") == 5400.0

def test_timestamp_to_seconds_invalid_returns_zero():
    assert _timestamp_to_seconds("bad") == 0.0

def test_timestamp_to_seconds_empty_returns_zero():
    assert _timestamp_to_seconds("") == 0.0


# --- parse_source_references ---

def test_parse_source_references_empty_context():
    assert parse_source_references("") == []

def test_parse_source_references_no_match():
    assert parse_source_references("nothing relevant here") == []

def test_parse_source_references_mismatched_path_ignored():
    # folder name != file name stem — not a transcript path we recognise
    assert parse_source_references("transcripts/show_a/show_b.txt\n") == []

def test_parse_source_references_from_reference_list():
    context = (
        '{"reference_id": "1", "content": "text [TIMESTAMP=1:30-1:45]"}\n'
        "[1] transcripts/show_a/show_a.txt\n"
    )
    references = parse_source_references(context)
    assert len(references) == 1
    assert references[0].stem == "show_a"
    assert references[0].start_seconds == 90.0

def test_parse_source_references_deduplicates_identical_hits():
    context = (
        '{"reference_id": "1", "content": "a [TIMESTAMP=1:30-1:45]"}\n'
        '{"reference_id": "1", "content": "b [TIMESTAMP=1:30-1:50]"}\n'
        "[1] transcripts/show_a/show_a.txt\n"
    )
    assert len(parse_source_references(context)) == 1

def test_parse_source_references_falls_back_to_embedded_source_tag():
    context = "[SOURCE=show_a]\n[SPEAKER=Ana]\n[TIMESTAMP=0:10-0:20]\n\ntext"
    references = parse_source_references(context)
    assert len(references) == 1
    assert references[0].stem == "show_a"
    assert references[0].start_seconds == 10.0


# --- parse_context_chunks ---

def test_parse_context_chunks_splits_on_double_newline():
    assert parse_context_chunks("chunk one\n\nchunk two\n\nchunk three") == [
        "chunk one", "chunk two", "chunk three"
    ]

def test_parse_context_chunks_strips_whitespace():
    assert parse_context_chunks("  chunk one  \n\n  chunk two  ") == ["chunk one", "chunk two"]

def test_parse_context_chunks_filters_blank_sections():
    assert parse_context_chunks("chunk one\n\n\n\nchunk two") == ["chunk one", "chunk two"]

def test_parse_context_chunks_empty_returns_list_with_empty_string():
    assert parse_context_chunks("") == [""]

def test_parse_context_chunks_no_double_newline_returns_single():
    assert parse_context_chunks("just one chunk") == ["just one chunk"]
