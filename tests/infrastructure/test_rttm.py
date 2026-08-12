from __future__ import annotations

import os
import tempfile

import pytest

from infrastructure.audio.rttm import MAX_CHUNK_SECONDS, parse_rttm, split_long_segments


def write_rttm(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rttm", delete=False) as f:
        f.write(content)
        return f.name


def _seg(start, end, speaker="SPK_1"):
    return {"start": start, "end": end, "speaker": speaker, "duration": end - start}


# --- parse_rttm ---

def test_parse_rttm_basic():
    path = write_rttm(
        "SPEAKER seg1 1 0.50 2.00 <NA> <NA> SPK_A <NA> <NA>\n"
        "SPEAKER seg1 1 3.00 1.50 <NA> <NA> SPK_B <NA> <NA>\n"
    )
    remap = {("seg1", "SPK_A"): "speaker_1", ("seg1", "SPK_B"): "speaker_2"}
    try:
        segs = parse_rttm(path, "seg1", remap)
        assert len(segs) == 2
        assert segs[0]["speaker"] == "speaker_1"
        assert segs[0]["start"] == pytest.approx(0.5)
        assert segs[0]["end"] == pytest.approx(2.5)
        assert segs[1]["speaker"] == "speaker_2"
    finally:
        os.unlink(path)

def test_parse_rttm_sorted_by_start():
    path = write_rttm(
        "SPEAKER seg1 1 5.00 1.00 <NA> <NA> SPK_B <NA> <NA>\n"
        "SPEAKER seg1 1 1.00 1.00 <NA> <NA> SPK_A <NA> <NA>\n"
    )
    try:
        segs = parse_rttm(path, "seg1", {})
        assert segs[0]["start"] < segs[1]["start"]
    finally:
        os.unlink(path)

def test_parse_rttm_unmapped_speaker_kept_as_is():
    path = write_rttm("SPEAKER seg1 1 0.00 1.00 <NA> <NA> SPK_X <NA> <NA>\n")
    try:
        assert parse_rttm(path, "seg1", {})[0]["speaker"] == "SPK_X"
    finally:
        os.unlink(path)

def test_parse_rttm_skips_zero_length_segments():
    path = write_rttm("SPEAKER seg1 1 1.00 0.00 <NA> <NA> SPK_A <NA> <NA>\n")
    try:
        assert parse_rttm(path, "seg1", {}) == []
    finally:
        os.unlink(path)


# --- split_long_segments ---

def test_split_short_segment_unchanged():
    seg = _seg(0, MAX_CHUNK_SECONDS - 1)
    assert split_long_segments([seg]) == [seg]

def test_split_exactly_max_unchanged():
    seg = _seg(0, MAX_CHUNK_SECONDS)
    assert split_long_segments([seg]) == [seg]

def test_split_long_segment_produces_multiple():
    assert len(split_long_segments([_seg(0, MAX_CHUNK_SECONDS * 3)])) == 3

def test_split_preserves_speaker():
    result = split_long_segments([_seg(0, MAX_CHUNK_SECONDS * 2, speaker="SPK_2")])
    assert all(s["speaker"] == "SPK_2" for s in result)

def test_split_chunks_cover_full_duration():
    seg = _seg(10, 10 + MAX_CHUNK_SECONDS * 2.5)
    result = split_long_segments([seg])
    assert result[0]["start"] == pytest.approx(10)
    assert result[-1]["end"] == pytest.approx(seg["end"])

def test_split_empty_list():
    assert split_long_segments([]) == []

def test_split_mixed_segments():
    short = _seg(0, 5)
    long = _seg(5, 5 + MAX_CHUNK_SECONDS * 2)
    result = split_long_segments([short, long])
    assert result[0] == short
    assert len(result) == 3
