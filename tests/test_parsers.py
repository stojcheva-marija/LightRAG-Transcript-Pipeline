import pytest
from parsers.transcript import TranscriptParser
from models.models import SpeakerTurn

PARSER = TranscriptParser()


TRANSCRIPT = """\
Alice (00:00-00:05): Hello everyone.
Bob (00:05-00:10): Hi Alice.
Alice (00:10-00:15): How are you?
"""


def make_turn(speaker, start, end, text):
    return SpeakerTurn(speaker=speaker, timestamp_start=start, timestamp_end=end, text=text)


# --- parse ---

class TestParse:

    def test_returns_correct_number_of_turns(self):
        turns = PARSER.parse(TRANSCRIPT)
        assert len(turns) == 3

    def test_parses_speaker(self):
        turns = PARSER.parse(TRANSCRIPT)
        assert turns[0].speaker == "Alice"
        assert turns[1].speaker == "Bob"

    def test_parses_timestamps(self):
        turns = PARSER.parse(TRANSCRIPT)
        assert turns[0].timestamp_start == "00:00"
        assert turns[0].timestamp_end == "00:05"

    def test_parses_text(self):
        turns = PARSER.parse(TRANSCRIPT)
        assert turns[0].text == "Hello everyone."

    def test_empty_transcript_returns_empty_list(self):
        assert PARSER.parse("") == []

    def test_none_transcript_returns_empty_list(self):
        assert PARSER.parse(None) == []

    def test_normalizes_extra_whitespace_in_speaker(self):
        turns = PARSER.parse("  Alice   (00:00-00:05): Hello.")
        assert turns[0].speaker == "Alice"

    def test_normalizes_extra_whitespace_in_text(self):
        turns = PARSER.parse("Alice (00:00-00:05):  Hello   world.")
        assert turns[0].text == "Hello world."

    def test_returns_speaker_turn_instances(self):
        turns = PARSER.parse(TRANSCRIPT)
        assert all(isinstance(t, SpeakerTurn) for t in turns)


# --- normalize ---

class TestNormalize:

    def test_collapses_whitespace(self):
        assert PARSER.normalize("hello   world") == "hello world"

    def test_strips_leading_and_trailing(self):
        assert PARSER.normalize("  hello  ") == "hello"

    def test_empty_string(self):
        assert PARSER.normalize("") == ""

    def test_none_returns_empty(self):
        assert PARSER.normalize(None) == ""

    def test_newlines_collapsed(self):
        assert PARSER.normalize("hello\nworld") == "hello world"


# --- format_speaker_turn ---

class TestFormatSpeakerTurn:

    def test_format(self):
        turn = make_turn("Alice", "00:00", "00:05", "Hello.")
        assert PARSER.format_speaker_turn(turn) == "Alice (00:00-00:05): Hello."


# --- context_window ---

class TestContextWindow:

    def setup_method(self):
        self.turns = [
            make_turn("Alice", "00:00", "00:05", "Hello."),
            make_turn("Bob",   "00:05", "00:10", "Hi."),
            make_turn("Alice", "00:10", "00:15", "Bye."),
            make_turn("Bob",   "00:15", "00:20", "See you."),
            make_turn("Alice", "00:20", "00:25", "Take care."),
        ]

    def test_preceding_and_following(self):
        preceding, following = PARSER.context_window(self.turns, idx=2, window=1)
        assert "Bob" in preceding
        assert "Bob" in following

    def test_no_preceding_at_start(self):
        preceding, _ = PARSER.context_window(self.turns, idx=0, window=2)
        assert preceding == "None"

    def test_no_following_at_end(self):
        _, following = PARSER.context_window(self.turns, idx=4, window=2)
        assert following == "None"

    def test_window_zero_returns_none_for_both(self):
        preceding, following = PARSER.context_window(self.turns, idx=2, window=0)
        assert preceding == "None"
        assert following == "None"

    def test_window_larger_than_turns(self):
        preceding, following = PARSER.context_window(self.turns, idx=2, window=10)
        assert preceding.count("\n") == 1
        assert following.count("\n") == 1

    def test_negative_window_treated_as_zero(self):
        preceding, following = PARSER.context_window(self.turns, idx=2, window=-5)
        assert preceding == "None"
        assert following == "None"
