from __future__ import annotations

from domain.parsing.transcript import TranscriptParser
from domain.transcript import TranscriptLine

PARSER = TranscriptParser()

TRANSCRIPT = """\
Alice (00:00-00:05): Hello everyone.
Bob (00:05-00:10): Hi Alice.
Alice (00:10-00:15): How are you?
"""


def make_line(speaker, start, end, text):
    return TranscriptLine(speaker=speaker, timestamp_start=start, timestamp_end=end, text=text)


# --- parse ---

class TestParse:

    def test_returns_correct_number_of_lines(self):
        assert len(PARSER.parse(TRANSCRIPT)) == 3

    def test_parses_speaker(self):
        lines = PARSER.parse(TRANSCRIPT)
        assert lines[0].speaker == "Alice"
        assert lines[1].speaker == "Bob"

    def test_parses_timestamps(self):
        lines = PARSER.parse(TRANSCRIPT)
        assert lines[0].timestamp_start == "00:00"
        assert lines[0].timestamp_end == "00:05"

    def test_parses_text(self):
        assert PARSER.parse(TRANSCRIPT)[0].text == "Hello everyone."

    def test_empty_transcript_returns_empty_list(self):
        assert PARSER.parse("") == []

    def test_none_transcript_returns_empty_list(self):
        assert PARSER.parse(None) == []

    def test_normalizes_extra_whitespace_in_speaker(self):
        assert PARSER.parse("  Alice   (00:00-00:05): Hello.")[0].speaker == "Alice"

    def test_normalizes_extra_whitespace_in_text(self):
        assert PARSER.parse("Alice (00:00-00:05):  Hello   world.")[0].text == "Hello world."

    def test_accepts_spaced_timestamp_separator(self):
        assert PARSER.parse("Alice (0:00 - 0:05): Hi.")[0].timestamp_end == "0:05"

    def test_returns_transcript_line_instances(self):
        assert all(isinstance(line, TranscriptLine) for line in PARSER.parse(TRANSCRIPT))


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


# --- format_line ---

class TestFormatLine:

    def test_format(self):
        line = make_line("Alice", "00:00", "00:05", "Hello.")
        assert PARSER.format_line(line) == "Alice (00:00-00:05): Hello."


# --- context_window ---

class TestContextWindow:

    def setup_method(self):
        self.lines = [
            make_line("Alice", "00:00", "00:05", "Hello."),
            make_line("Bob",   "00:05", "00:10", "Hi."),
            make_line("Alice", "00:10", "00:15", "Bye."),
            make_line("Bob",   "00:15", "00:20", "See you."),
            make_line("Alice", "00:20", "00:25", "Take care."),
        ]

    def test_preceding_and_following(self):
        preceding, following = PARSER.context_window(self.lines, idx=2, window=1)
        assert "Bob" in preceding
        assert "Bob" in following

    def test_no_preceding_at_start(self):
        preceding, _ = PARSER.context_window(self.lines, idx=0, window=2)
        assert preceding == "None"

    def test_no_following_at_end(self):
        _, following = PARSER.context_window(self.lines, idx=4, window=2)
        assert following == "None"

    def test_window_zero_returns_none_for_both(self):
        preceding, following = PARSER.context_window(self.lines, idx=2, window=0)
        assert preceding == "None"
        assert following == "None"

    def test_window_larger_than_available_lines(self):
        preceding, following = PARSER.context_window(self.lines, idx=2, window=10)
        assert preceding.count("\n") == 1
        assert following.count("\n") == 1

    def test_negative_window_treated_as_zero(self):
        preceding, following = PARSER.context_window(self.lines, idx=2, window=-5)
        assert preceding == "None"
        assert following == "None"
