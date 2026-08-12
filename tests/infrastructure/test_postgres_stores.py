from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

from infrastructure.storage.postgres_metadata_store import PostgresMetadataStore
from infrastructure.storage.postgres_speaker_directory import PostgresSpeakerDirectory


def _with_mock_cursor(adapter):
    """Swap the adapter's database for one yielding a mock cursor."""
    cursor = MagicMock()
    conn = MagicMock()

    @contextmanager
    def mock_cursor(commit=False, **kwargs):
        yield cursor
        if commit:
            conn.commit()

    adapter._db = MagicMock()
    adapter._db.cursor = mock_cursor
    return adapter, cursor


def make_metadata_store():
    return _with_mock_cursor(PostgresMetadataStore(MagicMock()))


def make_speaker_directory():
    return _with_mock_cursor(PostgresSpeakerDirectory(MagicMock()))


# --- metadata store: save ---

def test_save_executes_upsert():
    store, cur = make_metadata_store()
    store.save("doc1", "path/to/file", {"summary": "test"})
    cur.execute.assert_called_once()
    sql, params = cur.execute.call_args[0]
    assert "INSERT INTO document_metadata" in sql
    assert params[0] == "doc1"
    assert params[1] == "path/to/file"
    assert json.loads(params[2]) == {"summary": "test"}

def test_save_serializes_metadata_as_json():
    store, cur = make_metadata_store()
    store.save("doc1", "path", {"key": "value", "num": 42})
    _, params = cur.execute.call_args[0]
    assert json.loads(params[2]) == {"key": "value", "num": 42}

def test_save_handles_exception_gracefully():
    store, cur = make_metadata_store()
    cur.execute.side_effect = Exception("DB error")
    store.save("doc1", "path", {})  # must not raise


# --- metadata store: reads ---

def test_known_speaker_names_returns_list():
    store, cur = make_metadata_store()
    cur.fetchall.return_value = [("Ana",), ("Bojan",)]
    assert store.known_speaker_names() == ["Ana", "Bojan"]

def test_known_speaker_names_empty():
    store, cur = make_metadata_store()
    cur.fetchall.return_value = []
    assert store.known_speaker_names() == []

def test_known_speaker_names_handles_exception():
    store, cur = make_metadata_store()
    cur.execute.side_effect = Exception("DB error")
    assert store.known_speaker_names() == []


# --- speaker directory: normalization ---

def test_normalize_unit_vector():
    result = PostgresSpeakerDirectory._normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(result) == pytest.approx(1.0)

def test_normalize_zero_vector_does_not_raise():
    result = PostgresSpeakerDirectory._normalize(np.zeros(5))
    assert not np.any(np.isnan(result))


# --- speaker directory: remember ---

def test_remember_inserts_when_new():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = None
    directory.remember("Ana", np.random.rand(192))
    assert any("INSERT INTO speakers" in c[0][0] for c in cur.execute.call_args_list)

def test_remember_normalizes_embedding():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = None
    directory.remember("Ana", np.array([3.0, 4.0] + [0.0] * 190))
    insert_call = next(c for c in cur.execute.call_args_list if "INSERT" in c[0][0])
    stored = np.array(insert_call[0][1][2])
    assert np.linalg.norm(stored) == pytest.approx(1.0, abs=1e-5)

def test_remember_updates_when_speaker_exists():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = (1, np.random.rand(192).tolist())
    directory.remember("Ana", np.random.rand(192), files=1, turns=5)
    assert any("UPDATE speakers" in c[0][0] for c in cur.execute.call_args_list)

def test_remember_averages_the_stored_embedding():
    directory, cur = make_speaker_directory()
    existing = np.array([1.0] + [0.0] * 191)
    cur.fetchone.return_value = (1, existing.tolist())
    directory.remember("Ana", np.array([0.0, 1.0] + [0.0] * 190))
    update_call = next(c for c in cur.execute.call_args_list if "UPDATE" in c[0][0])
    stored = np.array(update_call[0][1][0])
    assert stored[0] == pytest.approx(stored[1])

def test_remember_handles_exception():
    directory, cur = make_speaker_directory()
    cur.execute.side_effect = Exception("DB error")
    directory.remember("Ana", np.random.rand(192))  # must not raise


# --- speaker directory: match ---

def test_match_returns_speaker_above_threshold():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = {"id": 1, "name": "Ana", "similarity": 0.92}
    match = directory.match(np.random.rand(192), threshold=0.75)
    assert (match.name, match.similarity) == ("Ana", 0.92)

def test_match_returns_none_below_threshold():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = {"id": 1, "name": "Ana", "similarity": 0.50}
    assert directory.match(np.random.rand(192), threshold=0.75) is None

def test_match_returns_none_when_directory_is_empty():
    directory, cur = make_speaker_directory()
    cur.fetchone.return_value = None
    assert directory.match(np.random.rand(192), threshold=0.75) is None

def test_match_handles_exception():
    directory, cur = make_speaker_directory()
    cur.execute.side_effect = Exception("DB error")
    assert directory.match(np.random.rand(192), threshold=0.75) is None
