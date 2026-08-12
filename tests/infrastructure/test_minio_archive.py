from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from domain.errors import AudioNotFound, TranscriptNotFound
from infrastructure.storage.minio_archive import MinIOTranscriptArchive


def make_archive() -> tuple[MinIOTranscriptArchive, MagicMock]:
    """An archive with a mocked boto3 S3 client."""
    with patch("infrastructure.storage.minio_archive.boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        cfg = MagicMock()
        cfg.bucket = "test-bucket"
        cfg.endpoint = "http://localhost:9000"
        cfg.access_key = "key"
        cfg.secret_key = "secret"
        archive = MinIOTranscriptArchive(cfg)
        archive.client = mock_s3
        return archive, mock_s3


# --- key helpers ---

def test_transcript_prefix():
    assert MinIOTranscriptArchive._transcript_prefix("my_show") == "transcripts/my_show/"

def test_nemo_prefix():
    assert MinIOTranscriptArchive._nemo_prefix("my_show") == "nemo/my_show/"


# --- store_audio ---

def test_store_audio_key():
    archive, s3 = make_archive()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"audio")
        path = f.name
    try:
        stem = os.path.splitext(os.path.basename(path))[0]
        key = archive.store_audio(path)
        assert key == f"transcripts/{stem}/{os.path.basename(path)}"
        s3.upload_file.assert_called_once_with(path, "test-bucket", key)
    finally:
        os.unlink(path)

def test_store_audio_returns_key():
    archive, _ = make_archive()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        key = archive.store_audio(path)
        assert key.startswith("transcripts/")
        assert key.endswith(".mp3")
    finally:
        os.unlink(path)


# --- store_outputs ---

def test_store_outputs_puts_txt_and_json():
    archive, s3 = make_archive()
    archive.store_outputs("show1", "transcript text", '{"key": "value"}')
    assert s3.put_object.call_count == 2
    keys = {c.kwargs["Key"] for c in s3.put_object.call_args_list}
    assert keys == {"transcripts/show1/show1.txt", "transcripts/show1/show1.json"}

def test_store_outputs_encodes_utf8():
    archive, s3 = make_archive()
    archive.store_outputs("show1", "Македонија", "{}")
    txt_call = next(c for c in s3.put_object.call_args_list if c.kwargs["Key"].endswith(".txt"))
    assert txt_call.kwargs["Body"] == "Македонија".encode("utf-8")


# --- diarization cache ---

def test_store_diarization_walks_directory():
    archive, s3 = make_archive()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "file1.rttm"), "w").close()
        open(os.path.join(tmpdir, "file2.pkl"), "w").close()
        archive.store_diarization("show1", tmpdir)
        assert s3.upload_file.call_count == 2

def test_store_diarization_key_prefix():
    archive, s3 = make_archive()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "output.rttm"), "w").close()
        archive.store_diarization("show1", tmpdir)
        assert s3.upload_file.call_args[0][2].startswith("nemo/show1/")

def test_has_diarization_true():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "nemo/show1/file.rttm"}]}
    assert archive.has_diarization("show1") is True

def test_has_diarization_false():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {}
    assert archive.has_diarization("show1") is False


# --- list_stems ---

def _paginated(archive, s3, pages):
    paginator = MagicMock()
    paginator.paginate.return_value = pages
    s3.get_paginator.return_value = paginator

def test_list_stems_returns_sorted():
    archive, s3 = make_archive()
    _paginated(archive, s3, [{"Contents": [
        {"Key": "transcripts/zebra/zebra.txt"},
        {"Key": "transcripts/alpha/alpha.txt"},
        {"Key": "transcripts/alpha/alpha.json"},
    ]}])
    assert archive.list_stems() == ["alpha", "zebra"]

def test_list_stems_ignores_non_txt():
    archive, s3 = make_archive()
    _paginated(archive, s3, [{"Contents": [
        {"Key": "transcripts/show1/show1.json"},
        {"Key": "transcripts/show1/show1.txt"},
    ]}])
    assert archive.list_stems() == ["show1"]

def test_list_stems_empty():
    archive, s3 = make_archive()
    _paginated(archive, s3, [{}])
    assert archive.list_stems() == []


# --- load_transcript ---

def _mock_read(content: str) -> dict:
    body = MagicMock()
    body.read.return_value = content.encode("utf-8")
    return {"Body": body}

def test_load_transcript_returns_txt_and_json():
    archive, s3 = make_archive()
    s3.get_object.side_effect = [_mock_read("transcript content"), _mock_read('{"summary": "test"}')]
    txt, meta = archive.load_transcript("show1")
    assert txt == "transcript content"
    assert meta == '{"summary": "test"}'

def test_load_transcript_missing_txt_raises_domain_error():
    archive, s3 = make_archive()
    s3.get_object.side_effect = Exception("not found")
    with pytest.raises(TranscriptNotFound):
        archive.load_transcript("show1")

def test_load_transcript_missing_json_returns_empty():
    archive, s3 = make_archive()
    s3.get_object.side_effect = [_mock_read("transcript content"), Exception("not found")]
    txt, meta = archive.load_transcript("show1")
    assert txt == "transcript content"
    assert meta == ""


# --- audio lookup ---

def test_audio_url_skips_txt_and_json():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.json"},
        {"Key": "transcripts/show1/show1.wav"},
    ]}
    s3.generate_presigned_url.return_value = "http://presigned-url"
    assert archive.audio_url("show1") == "http://presigned-url"
    s3.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": "transcripts/show1/show1.wav"},
        ExpiresIn=3600,
    )

def test_audio_url_without_audio_raises_domain_error():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.json"},
    ]}
    with pytest.raises(AudioNotFound):
        archive.audio_url("show1")

def test_download_audio_returns_local_path():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.wav"},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = archive.download_audio("show1", tmpdir)
        assert result == os.path.join(tmpdir, "show1.wav")
        s3.download_file.assert_called_once_with(
            "test-bucket", "transcripts/show1/show1.wav", result
        )

def test_download_audio_without_audio_raises_domain_error():
    archive, s3 = make_archive()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "transcripts/show1/show1.txt"}]}
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(AudioNotFound):
            archive.download_audio("show1", tmpdir)
