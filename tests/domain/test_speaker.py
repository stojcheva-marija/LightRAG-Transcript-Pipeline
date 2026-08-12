from __future__ import annotations

from domain.speaker import (
    ResolutionSource,
    SpeakerMatch,
    SpeakerResolution,
    SpeakerResolutions,
)


def matched(speaker_id="SPK_1", name="Ana", similarity=0.923):
    return SpeakerResolution.matched(speaker_id, SpeakerMatch(name=name, similarity=similarity))


def unknown(speaker_id="SPK_1"):
    return SpeakerResolution.unknown(speaker_id)


def resolutions(*items):
    return SpeakerResolutions(items={r.speaker_id: r for r in items})


# --- naming precedence ---

def test_database_match_keeps_its_name():
    result = resolutions(matched()).with_llm_identities({"SPK_1": "Bojan"})
    assert result.name_mapping()["SPK_1"] == "Ana"

def test_llm_names_an_unknown_speaker():
    result = resolutions(unknown()).with_llm_identities({"SPK_1": "Bojan"})
    assert result.name_mapping()["SPK_1"] == "Bojan"
    assert result.items["SPK_1"].source is ResolutionSource.LLM

def test_unnamed_speaker_keeps_its_cluster_id():
    assert resolutions(unknown()).name_mapping()["SPK_1"] == "SPK_1"

def test_empty_llm_name_is_ignored():
    result = resolutions(unknown()).with_llm_identities({"SPK_1": ""})
    assert result.name_mapping()["SPK_1"] == "SPK_1"

def test_llm_identity_for_unknown_cluster_is_ignored():
    result = resolutions(unknown("SPK_1")).with_llm_identities({"SPK_9": "Ghost"})
    assert set(result.items) == {"SPK_1"}

def test_with_llm_identities_does_not_mutate_the_original():
    original = resolutions(unknown())
    original.with_llm_identities({"SPK_1": "Bojan"})
    assert original.name_mapping()["SPK_1"] == "SPK_1"


# --- collection helpers ---

def test_unresolved_ids_lists_only_unknown_speakers():
    result = resolutions(matched("SPK_1"), unknown("SPK_2"))
    assert result.unresolved_ids == ["SPK_2"]

def test_matched_count_counts_database_hits_only():
    result = resolutions(matched("SPK_1"), unknown("SPK_2")).with_llm_identities({"SPK_2": "Bojan"})
    assert result.matched_count == 1

def test_similarity_is_rounded():
    assert matched(similarity=0.1234567).similarity == 0.1235


# --- report ---

def test_report_contains_header():
    assert "## Speaker Resolution Report" in resolutions().report()

def test_report_shows_database_match_with_similarity():
    report = resolutions(matched()).report()
    assert "matched from DB" in report
    assert "0.923" in report

def test_report_shows_llm_identification():
    report = resolutions(unknown()).with_llm_identities({"SPK_1": "Bojan"}).report()
    assert "identified by LLM" in report

def test_report_shows_unknown():
    assert "unknown" in resolutions(unknown()).report()

def test_report_is_sorted_by_speaker_id():
    report = resolutions(unknown("SPK_3"), unknown("SPK_1")).report()
    assert report.index("SPK_1") < report.index("SPK_3")
