from __future__ import annotations

import tempfile
import os

import pytest

from ingestion.transcriber import (
    _split_long_segments,
    _merge_or_append,
    MAX_CHUNK_SECONDS,
    MERGE_GAP_SECONDS,
    MIN_SEGMENT_DURATION,
)
from ingestion.speaker_resolver import build_final_speaker_mapping, generate_resolution_report
from ingestion.metadata_builder import _fmt_time, build_metadata_json, build_transcript_txt


# ---------------------------------------------------------------------------
# _fmt_time
# ---------------------------------------------------------------------------

def test_fmt_time_seconds_only():
    assert _fmt_time(45.0) == "0:45"

def test_fmt_time_minutes_and_seconds():
    assert _fmt_time(125.9) == "2:05"

def test_fmt_time_with_hours():
    assert _fmt_time(3661.0) == "1:01:01"

def test_fmt_time_exactly_one_hour():
    assert _fmt_time(3600.0) == "1:00:00"

def test_fmt_time_zero():
    assert _fmt_time(0.0) == "0:00"


# ---------------------------------------------------------------------------
# _split_long_segments
# ---------------------------------------------------------------------------

def _seg(start, end, speaker="SPK_1"):
    return {"start": start, "end": end, "speaker": speaker, "duration": end - start}


def test_split_short_segment_unchanged():
    seg = _seg(0, MAX_CHUNK_SECONDS - 1)
    result = _split_long_segments([seg])
    assert result == [seg]

def test_split_exactly_max_unchanged():
    seg = _seg(0, MAX_CHUNK_SECONDS)
    result = _split_long_segments([seg])
    assert result == [seg]

def test_split_long_segment_produces_multiple():
    seg = _seg(0, MAX_CHUNK_SECONDS * 3)
    result = _split_long_segments([seg])
    assert len(result) == 3

def test_split_preserves_speaker():
    seg = _seg(0, MAX_CHUNK_SECONDS * 2, speaker="SPK_2")
    result = _split_long_segments([seg])
    assert all(s["speaker"] == "SPK_2" for s in result)

def test_split_chunks_cover_full_duration():
    seg = _seg(10, 10 + MAX_CHUNK_SECONDS * 2.5)
    result = _split_long_segments([seg])
    assert result[0]["start"] == pytest.approx(10)
    assert result[-1]["end"] == pytest.approx(seg["end"])

def test_split_empty_list():
    assert _split_long_segments([]) == []

def test_split_mixed_segments():
    short = _seg(0, 5)
    long = _seg(5, 5 + MAX_CHUNK_SECONDS * 2)
    result = _split_long_segments([short, long])
    assert result[0] == short
    assert len(result) == 3


# ---------------------------------------------------------------------------
# _merge_or_append
# ---------------------------------------------------------------------------

def _turn(start, end, speaker="SPK_1", text="hello"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def test_merge_same_speaker_close_gap():
    turns = [_turn(0, 5, text="Hello")]
    _merge_or_append(turns, _turn(5.5, 10, text="world"))
    assert len(turns) == 1
    assert turns[0]["text"] == "Hello world"
    assert turns[0]["end"] == 10

def test_no_merge_different_speaker():
    turns = [_turn(0, 5, speaker="SPK_1")]
    _merge_or_append(turns, _turn(5.5, 10, speaker="SPK_2"))
    assert len(turns) == 2

def test_no_merge_large_gap():
    turns = [_turn(0, 5)]
    _merge_or_append(turns, _turn(5 + MERGE_GAP_SECONDS + 1, 10))
    assert len(turns) == 2

def test_merge_appends_to_empty():
    turns = []
    _merge_or_append(turns, _turn(0, 5))
    assert len(turns) == 1

def test_merge_updates_end_time():
    turns = [_turn(0, 5)]
    _merge_or_append(turns, _turn(6, 12))
    assert turns[0]["end"] == 12


# ---------------------------------------------------------------------------
# build_final_speaker_mapping
# ---------------------------------------------------------------------------

def test_mapping_db_source():
    results = {"SPK_1": {"source": "db", "name": "Ana", "similarity": 0.9}}
    mapping = build_final_speaker_mapping(results, {})
    assert mapping["SPK_1"] == "Ana"

def test_mapping_llm_source():
    results = {"SPK_1": {"source": "unknown", "name": "SPK_1", "similarity": None}}
    mapping = build_final_speaker_mapping(results, {"SPK_1": "Bojan"})
    assert mapping["SPK_1"] == "Bojan"

def test_mapping_fallback_to_id():
    results = {"SPK_1": {"source": "unknown", "name": "SPK_1", "similarity": None}}
    mapping = build_final_speaker_mapping(results, {})
    assert mapping["SPK_1"] == "SPK_1"

def test_mapping_db_takes_priority_over_llm():
    results = {"SPK_1": {"source": "db", "name": "Ana", "similarity": 0.9}}
    mapping = build_final_speaker_mapping(results, {"SPK_1": "Bojan"})
    assert mapping["SPK_1"] == "Ana"


# ---------------------------------------------------------------------------
# generate_resolution_report
# ---------------------------------------------------------------------------

def test_report_contains_header():
    report = generate_resolution_report({}, {}, {})
    assert "## Speaker Resolution Report" in report

def test_report_db_match():
    results = {"SPK_1": {"source": "db", "name": "Ana", "similarity": 0.923}}
    mapping = {"SPK_1": "Ana"}
    report = generate_resolution_report(results, {}, mapping)
    assert "matched from DB" in report
    assert "0.923" in report

def test_report_llm_identified():
    results = {"SPK_1": {"source": "unknown", "name": "SPK_1", "similarity": None}}
    mapping = {"SPK_1": "Bojan"}
    report = generate_resolution_report(results, {"SPK_1": "Bojan"}, mapping)
    assert "identified by LLM" in report

def test_report_unknown():
    results = {"SPK_1": {"source": "unknown", "name": "SPK_1", "similarity": None}}
    mapping = {"SPK_1": "SPK_1"}
    report = generate_resolution_report(results, {}, mapping)
    assert "unknown" in report

def test_report_speakers_sorted():
    results = {
        "SPK_3": {"source": "unknown", "name": "SPK_3", "similarity": None},
        "SPK_1": {"source": "unknown", "name": "SPK_1", "similarity": None},
    }
    mapping = {"SPK_1": "SPK_1", "SPK_3": "SPK_3"}
    report = generate_resolution_report(results, {}, mapping)
    assert report.index("SPK_1") < report.index("SPK_3")


# ---------------------------------------------------------------------------
# build_metadata_json
# ---------------------------------------------------------------------------

TRANSCRIPT = [
    {"speaker": "SPK_1", "start": 0.0,  "end": 10.0, "text": "Hello"},
    {"speaker": "SPK_2", "start": 10.0, "end": 20.0, "text": "Hi"},
]
MAPPING = {"SPK_1": "Ana", "SPK_2": "Bojan"}
LLM_SUMMARY = {"topics": ["greeting"], "summary": "A short exchange.", "language": "mk"}


def test_metadata_duration():
    meta = build_metadata_json(TRANSCRIPT, MAPPING, "2024-01-01", "10:00", "Show", "Skopje", LLM_SUMMARY)
    assert meta["duration_seconds"] == 20

def test_metadata_speakers_ordered():
    meta = build_metadata_json(TRANSCRIPT, MAPPING, "2024-01-01", "10:00", "Show", "Skopje", LLM_SUMMARY)
    assert meta["speakers"] == ["Ana", "Bojan"]

def test_metadata_speaker_count():
    meta = build_metadata_json(TRANSCRIPT, MAPPING, "2024-01-01", "10:00", "Show", "Skopje", LLM_SUMMARY)
    assert meta["speaker_count"] == 2

def test_metadata_topics_and_summary():
    meta = build_metadata_json(TRANSCRIPT, MAPPING, "2024-01-01", "10:00", "Show", "Skopje", LLM_SUMMARY)
    assert meta["topics"] == ["greeting"]
    assert meta["summary"] == "A short exchange."

def test_metadata_empty_llm_summary_defaults():
    meta = build_metadata_json(TRANSCRIPT, MAPPING, "2024-01-01", "10:00", "Show", "Skopje", {})
    assert meta["topics"] == []
    assert meta["summary"] == ""
    assert meta["language"] == "mk"


# ---------------------------------------------------------------------------
# build_transcript_txt
# ---------------------------------------------------------------------------

def test_transcript_txt_format():
    txt = build_transcript_txt(TRANSCRIPT, MAPPING)
    lines = txt.split("\n\n")
    assert lines[0].startswith("Ana (")
    assert "Hello" in lines[0]

def test_transcript_txt_separator():
    txt = build_transcript_txt(TRANSCRIPT, MAPPING)
    assert "\n\n" in txt

def test_transcript_txt_empty():
    assert build_transcript_txt([], {}) == ""


# ---------------------------------------------------------------------------
# _parse_rttm
# ---------------------------------------------------------------------------

def test_parse_rttm_basic():
    from ingestion.transcriber import _parse_rttm

    rttm_content = (
        "SPEAKER seg1 1 0.50 2.00 <NA> <NA> SPK_A <NA> <NA>\n"
        "SPEAKER seg1 1 3.00 1.50 <NA> <NA> SPK_B <NA> <NA>\n"
    )
    remap = {("seg1", "SPK_A"): "speaker_1", ("seg1", "SPK_B"): "speaker_2"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
        f.write(rttm_content)
        path = f.name

    try:
        segs = _parse_rttm(path, "seg1", remap)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "speaker_1"
        assert segs[0]["start"] == pytest.approx(0.5)
        assert segs[0]["end"] == pytest.approx(2.5)
        assert segs[1]["speaker"] == "speaker_2"
    finally:
        os.unlink(path)

def test_parse_rttm_sorted_by_start():
    from ingestion.transcriber import _parse_rttm

    rttm_content = (
        "SPEAKER seg1 1 5.00 1.00 <NA> <NA> SPK_B <NA> <NA>\n"
        "SPEAKER seg1 1 1.00 1.00 <NA> <NA> SPK_A <NA> <NA>\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
        f.write(rttm_content)
        path = f.name

    try:
        segs = _parse_rttm(path, "seg1", {})
        assert segs[0]["start"] < segs[1]["start"]
    finally:
        os.unlink(path)

def test_parse_rttm_unknown_speaker_kept_as_is():
    from ingestion.transcriber import _parse_rttm

    rttm_content = "SPEAKER seg1 1 0.00 1.00 <NA> <NA> SPK_X <NA> <NA>\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
        f.write(rttm_content)
        path = f.name

    try:
        segs = _parse_rttm(path, "seg1", {})
        assert segs[0]["speaker"] == "SPK_X"
    finally:
        os.unlink(path)
