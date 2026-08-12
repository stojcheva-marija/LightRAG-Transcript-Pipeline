from __future__ import annotations

import json

from domain.metadata import (
    SessionDetails,
    TranscriptMetadata,
    TranscriptSummary,
    metadata_header,
)
from domain.transcript import SpeakerTurn, Transcript

TRANSCRIPT = Transcript([
    SpeakerTurn(speaker="SPK_1", start=0.0, end=10.0, text="Hello"),
    SpeakerTurn(speaker="SPK_2", start=10.0, end=20.0, text="Hi"),
])
NAMES = {"SPK_1": "Ana", "SPK_2": "Bojan"}
SESSION = SessionDetails(date="2024-01-01", time="10:00", show="Show", location="Skopje")
SUMMARY = TranscriptSummary(topics=["greeting"], summary="A short exchange.", language="mk")


def build(summary=SUMMARY) -> dict:
    return TranscriptMetadata.build(TRANSCRIPT, NAMES, SESSION, summary).to_dict()


# --- TranscriptMetadata ---

def test_duration_comes_from_the_transcript():
    assert build()["duration_seconds"] == 20

def test_speakers_are_ordered_and_resolved():
    assert build()["speakers"] == ["Ana", "Bojan"]

def test_speaker_count_matches_speakers():
    assert build()["speaker_count"] == 2

def test_topics_and_summary_come_from_the_summary():
    meta = build()
    assert meta["topics"] == ["greeting"]
    assert meta["summary"] == "A short exchange."

def test_session_details_are_carried_through():
    meta = build()
    assert meta["date"] == "2024-01-01"
    assert meta["time"] == "10:00"
    assert meta["show"] == "Show"
    assert meta["location"] == "Skopje"

def test_empty_summary_falls_back_to_defaults():
    meta = build(TranscriptSummary.empty())
    assert meta["topics"] == []
    assert meta["summary"] == ""
    assert meta["language"] == "mk"


# --- TranscriptSummary ---

def test_summary_from_payload():
    summary = TranscriptSummary.from_payload({"topics": ["a"], "summary": "s", "language": "en"})
    assert (summary.topics, summary.summary, summary.language) == (["a"], "s", "en")

def test_summary_from_empty_payload_defaults_to_macedonian():
    assert TranscriptSummary.from_payload({}).language == "mk"

def test_summary_from_none_payload():
    assert TranscriptSummary.from_payload(None).topics == []


# --- metadata_header ---

def test_header_includes_date_and_show():
    header = metadata_header(json.dumps({"date": "2024-01-01", "show": "My Show"}))
    assert "[DATE=2024-01-01]" in header
    assert "[SHOW=My Show]" in header

def test_header_omits_missing_fields():
    assert metadata_header(json.dumps({"date": "2024-01-01"})) == "[DATE=2024-01-01]\n"

def test_header_of_empty_metadata_is_empty():
    assert metadata_header("") == ""

def test_header_of_invalid_json_is_empty():
    assert metadata_header("not json") == ""

def test_header_without_known_fields_is_empty():
    assert metadata_header(json.dumps({"summary": "x"})) == ""
