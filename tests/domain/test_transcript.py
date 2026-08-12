from __future__ import annotations

from domain.transcript import (
    MERGE_GAP_SECONDS,
    Chunk,
    SpeakerTurn,
    Transcript,
    format_timestamp,
    merge_adjacent_turns,
)

TURNS = [
    SpeakerTurn(speaker="SPK_1", start=0.0, end=10.0, text="Hello"),
    SpeakerTurn(speaker="SPK_2", start=10.0, end=20.0, text="Hi"),
]
NAMES = {"SPK_1": "Ana", "SPK_2": "Bojan"}


# --- format_timestamp ---

def test_format_timestamp_seconds_only():
    assert format_timestamp(45.0) == "0:45"

def test_format_timestamp_minutes_and_seconds():
    assert format_timestamp(125.9) == "2:05"

def test_format_timestamp_with_hours():
    assert format_timestamp(3661.0) == "1:01:01"

def test_format_timestamp_exactly_one_hour():
    assert format_timestamp(3600.0) == "1:00:00"

def test_format_timestamp_zero():
    assert format_timestamp(0.0) == "0:00"


# --- merge_adjacent_turns ---

def _turn(start, end, speaker="SPK_1", text="hello"):
    return SpeakerTurn(speaker=speaker, start=start, end=end, text=text)


def test_merge_same_speaker_close_gap():
    merged = merge_adjacent_turns([_turn(0, 5, text="Hello"), _turn(5.5, 10, text="world")])
    assert len(merged) == 1
    assert merged[0].text == "Hello world"
    assert merged[0].end == 10

def test_no_merge_different_speaker():
    merged = merge_adjacent_turns([_turn(0, 5, speaker="SPK_1"), _turn(5.5, 10, speaker="SPK_2")])
    assert len(merged) == 2

def test_no_merge_large_gap():
    merged = merge_adjacent_turns([_turn(0, 5), _turn(5 + MERGE_GAP_SECONDS + 1, 10)])
    assert len(merged) == 2

def test_merge_empty_list():
    assert merge_adjacent_turns([]) == []

def test_merge_updates_end_time():
    merged = merge_adjacent_turns([_turn(0, 5), _turn(6, 12)])
    assert merged[0].end == 12

def test_merge_chains_three_turns():
    merged = merge_adjacent_turns([_turn(0, 5, text="a"), _turn(6, 10, text="b"), _turn(11, 15, text="c")])
    assert len(merged) == 1
    assert merged[0].text == "a b c"


# --- Transcript ---

def test_duration_spans_first_to_last_turn():
    assert Transcript(TURNS).duration_seconds == 20

def test_duration_of_empty_transcript_is_zero():
    assert Transcript([]).duration_seconds == 0

def test_speaker_names_ordered_by_first_appearance():
    assert Transcript(TURNS).speaker_names(NAMES) == ["Ana", "Bojan"]

def test_speaker_names_deduplicated():
    turns = TURNS + [SpeakerTurn(speaker="SPK_1", start=20.0, end=25.0, text="Again")]
    assert Transcript(turns).speaker_names(NAMES) == ["Ana", "Bojan"]

def test_speaker_names_fall_back_to_cluster_id():
    assert Transcript(TURNS).speaker_names({}) == ["SPK_1", "SPK_2"]

def test_turn_count_for_speaker():
    turns = TURNS + [SpeakerTurn(speaker="SPK_1", start=20.0, end=25.0, text="Again")]
    assert Transcript(turns).turn_count_for("SPK_1") == 2

def test_render_uses_resolved_names_and_blank_line_separator():
    rendered = Transcript(TURNS).render(NAMES)
    lines = rendered.split("\n\n")
    assert lines[0].startswith("Ana (")
    assert "Hello" in lines[0]
    assert len(lines) == 2

def test_render_empty_transcript():
    assert Transcript([]).render({}) == ""

def test_render_round_trips_through_the_parser():
    from domain.parsing.transcript import TranscriptParser

    lines = TranscriptParser().parse(Transcript(TURNS).render(NAMES))
    assert [line.speaker for line in lines] == ["Ana", "Bojan"]
    assert lines[0].timestamp_start == "0:00"

def test_excerpt_uses_cluster_ids():
    assert Transcript(TURNS).excerpt() == "SPK_1: Hello\nSPK_2: Hi"

def test_len_and_bool():
    assert len(Transcript(TURNS)) == 2
    assert Transcript(TURNS)
    assert not Transcript([])


# --- Chunk ---

def test_chunk_formatted_text_carries_source_speaker_and_timestamp():
    chunk = Chunk(
        chunk_id="doc1::block::00001", doc_id="doc1", speaker="Ana",
        timestamp_start="00:00", timestamp_end="00:30", text="Hello",
    )
    assert "[SOURCE=doc1]" in chunk.formatted_text
    assert "[SPEAKER=Ana]" in chunk.formatted_text
    assert "[TIMESTAMP=00:00-00:30]" in chunk.formatted_text
