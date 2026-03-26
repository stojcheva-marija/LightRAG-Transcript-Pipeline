from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from storage.metadata_repository import MetadataRepository
from storage.speaker_repository import SpeakerRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta_repo() -> tuple[MetadataRepository, MagicMock]:
    cfg = MagicMock()
    cfg.to_dict.return_value = {}
    repo = MetadataRepository(cfg)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    @contextmanager
    def mock_cursor(commit=False, **kwargs):
        yield mock_cur
        if commit:
            mock_conn.commit()

    repo._cursor = mock_cursor
    return repo, mock_cur


def make_speaker_repo() -> tuple[SpeakerRepository, MagicMock]:
    cfg = MagicMock()
    cfg.to_dict.return_value = {}
    repo = SpeakerRepository(cfg)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur

    @contextmanager
    def mock_cursor(commit=False, **kwargs):
        yield mock_cur
        if commit:
            mock_conn.commit()

    repo._cursor = mock_cursor
    return repo, mock_cur


# ---------------------------------------------------------------------------
# MetadataRepository — save
# ---------------------------------------------------------------------------

def test_save_executes_upsert():
    repo, cur = make_meta_repo()
    repo.save("doc1", "path/to/file", {"summary": "test"})
    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args[0]
    assert "INSERT INTO document_metadata" in sql
    assert params[0] == "doc1"
    assert params[1] == "path/to/file"
    assert json.loads(params[2]) == {"summary": "test"}

def test_save_serializes_metadata_as_json():
    repo, cur = make_meta_repo()
    repo.save("doc1", "path", {"key": "value", "num": 42})
    _, params = cur.execute.call_args[0]
    assert json.loads(params[2]) == {"key": "value", "num": 42}

def test_save_handles_exception_gracefully():
    repo, cur = make_meta_repo()
    cur.execute.side_effect = Exception("DB error")
    repo.save("doc1", "path", {})  # should not raise


# ---------------------------------------------------------------------------
# MetadataRepository — get_file_path
# ---------------------------------------------------------------------------

def test_get_file_path_returns_path():
    repo, cur = make_meta_repo()
    cur.fetchone.return_value = ("path/to/audio.wav",)
    result = repo.get_file_path("doc1")
    assert result == "path/to/audio.wav"

def test_get_file_path_returns_none_when_missing():
    repo, cur = make_meta_repo()
    cur.fetchone.return_value = None
    assert repo.get_file_path("doc1") is None

def test_get_file_path_handles_exception():
    repo, cur = make_meta_repo()
    cur.execute.side_effect = Exception("DB error")
    assert repo.get_file_path("doc1") is None


# ---------------------------------------------------------------------------
# MetadataRepository — get_metadata
# ---------------------------------------------------------------------------

def test_get_metadata_returns_dict():
    repo, cur = make_meta_repo()
    cur.fetchone.return_value = ({"summary": "test"},)
    result = repo.get_metadata("doc1")
    assert result == {"summary": "test"}

def test_get_metadata_returns_none_when_missing():
    repo, cur = make_meta_repo()
    cur.fetchone.return_value = None
    assert repo.get_metadata("doc1") is None

def test_get_metadata_handles_exception():
    repo, cur = make_meta_repo()
    cur.execute.side_effect = Exception("DB error")
    assert repo.get_metadata("doc1") is None


# ---------------------------------------------------------------------------
# MetadataRepository — get_all_speakers
# ---------------------------------------------------------------------------

def test_get_all_speakers_returns_list():
    repo, cur = make_meta_repo()
    cur.fetchall.return_value = [("Ana",), ("Bojan",)]
    result = repo.get_all_speakers()
    assert result == ["Ana", "Bojan"]

def test_get_all_speakers_empty():
    repo, cur = make_meta_repo()
    cur.fetchall.return_value = []
    assert repo.get_all_speakers() == []

def test_get_all_speakers_handles_exception():
    repo, cur = make_meta_repo()
    cur.execute.side_effect = Exception("DB error")
    assert repo.get_all_speakers() == []


# ---------------------------------------------------------------------------
# SpeakerRepository — _normalize
# ---------------------------------------------------------------------------

def test_normalize_unit_vector():
    v = np.array([3.0, 4.0])
    result = SpeakerRepository._normalize(v)
    assert np.linalg.norm(result) == pytest.approx(1.0)

def test_normalize_zero_vector_does_not_raise():
    v = np.zeros(5)
    result = SpeakerRepository._normalize(v)
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# SpeakerRepository — upsert_speaker (insert path)
# ---------------------------------------------------------------------------

def test_upsert_speaker_inserts_when_new():
    repo, cur = make_speaker_repo()
    cur.fetchone.return_value = None
    emb = np.random.rand(192)
    repo.upsert_speaker("Ana", emb)
    calls = [c[0][0] for c in cur.execute.call_args_list]
    assert any("INSERT INTO speakers" in sql for sql in calls)

def test_upsert_speaker_normalizes_embedding():
    repo, cur = make_speaker_repo()
    cur.fetchone.return_value = None
    emb = np.array([3.0, 4.0] + [0.0] * 190)
    repo.upsert_speaker("Ana", emb)
    insert_call = next(c for c in cur.execute.call_args_list if "INSERT" in c[0][0])
    stored_emb = np.array(insert_call[0][1][2])
    assert np.linalg.norm(stored_emb) == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SpeakerRepository — upsert_speaker (update path)
# ---------------------------------------------------------------------------

def test_upsert_speaker_updates_when_exists():
    repo, cur = make_speaker_repo()
    existing_emb = np.random.rand(192).tolist()
    cur.fetchone.return_value = (1, existing_emb)
    emb = np.random.rand(192)
    repo.upsert_speaker("Ana", emb, no_files=1, no_turns=5)
    calls = [c[0][0] for c in cur.execute.call_args_list]
    assert any("UPDATE speakers" in sql for sql in calls)

def test_upsert_speaker_handles_exception():
    repo, cur = make_speaker_repo()
    cur.execute.side_effect = Exception("DB error")
    repo.upsert_speaker("Ana", np.random.rand(192))  # should not raise


# ---------------------------------------------------------------------------
# SpeakerRepository — match_speaker
# ---------------------------------------------------------------------------

def test_match_speaker_returns_match_above_threshold():
    repo, cur = make_speaker_repo()
    cur.fetchone.return_value = {"id": 1, "name": "Ana", "similarity": 0.92}
    result = repo.match_speaker(np.random.rand(192), threshold=0.75)
    assert result == {"id": 1, "name": "Ana", "similarity": 0.92}

def test_match_speaker_returns_none_below_threshold():
    repo, cur = make_speaker_repo()
    cur.fetchone.return_value = {"id": 1, "name": "Ana", "similarity": 0.50}
    result = repo.match_speaker(np.random.rand(192), threshold=0.75)
    assert result is None

def test_match_speaker_returns_none_when_empty():
    repo, cur = make_speaker_repo()
    cur.fetchone.return_value = None
    assert repo.match_speaker(np.random.rand(192), threshold=0.75) is None

def test_match_speaker_handles_exception():
    repo, cur = make_speaker_repo()
    cur.execute.side_effect = Exception("DB error")
    assert repo.match_speaker(np.random.rand(192), threshold=0.75) is None


# ---------------------------------------------------------------------------
# SpeakerRepository — get_all_speakers
# ---------------------------------------------------------------------------

def test_get_all_speakers_returns_dicts():
    repo, cur = make_speaker_repo()
    cur.fetchall.return_value = [
        {"id": 1, "name": "Ana",   "notes": "", "no_files": 2, "no_turns": 10},
        {"id": 2, "name": "Bojan", "notes": "", "no_files": 1, "no_turns": 5},
    ]
    result = repo.get_all_speakers()
    assert len(result) == 2
    assert result[0]["name"] == "Ana"

def test_get_all_speakers_empty():
    repo, cur = make_speaker_repo()
    cur.fetchall.return_value = []
    assert repo.get_all_speakers() == []

def test_get_all_speakers_handles_exception():
    repo, cur = make_speaker_repo()
    cur.execute.side_effect = Exception("DB error")
    assert repo.get_all_speakers() == []
