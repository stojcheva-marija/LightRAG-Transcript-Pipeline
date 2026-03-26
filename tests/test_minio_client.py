from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest

from storage.minio_client import MinIOClient


def make_client() -> tuple[MinIOClient, MagicMock]:
    """Return a MinIOClient with a mocked boto3 S3 client."""
    with patch("storage.minio_client.boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        cfg = MagicMock()
        cfg.minio.bucket = "test-bucket"
        cfg.minio.endpoint = "http://localhost:9000"
        cfg.minio.access_key = "key"
        cfg.minio.secret_key = "secret"
        client = MinIOClient(cfg)
        client.client = mock_s3
        return client, mock_s3


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def test_transcript_prefix():
    assert MinIOClient._transcript_prefix("my_show") == "transcripts/my_show/"

def test_nemo_prefix():
    assert MinIOClient._nemo_prefix("my_show") == "nemo/my_show/"


# ---------------------------------------------------------------------------
# upload_raw_audio
# ---------------------------------------------------------------------------

def test_upload_raw_audio_key():
    client, s3 = make_client()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"audio")
        path = f.name
    try:
        stem = os.path.splitext(os.path.basename(path))[0]
        key = client.upload_raw_audio(path)
        assert key == f"transcripts/{stem}/{os.path.basename(path)}"
        s3.upload_file.assert_called_once_with(path, "test-bucket", key)
    finally:
        os.unlink(path)

def test_upload_raw_audio_returns_key():
    client, s3 = make_client()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        key = client.upload_raw_audio(path)
        assert key.startswith("transcripts/")
        assert key.endswith(".mp3")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# upload_ingestion_outputs
# ---------------------------------------------------------------------------

def test_upload_ingestion_outputs_puts_txt_and_json():
    client, s3 = make_client()
    client.upload_ingestion_outputs("show1", "transcript text", '{"key": "value"}')
    assert s3.put_object.call_count == 2
    calls = s3.put_object.call_args_list
    keys = {c.kwargs["Key"] for c in calls}
    assert "transcripts/show1/show1.txt" in keys
    assert "transcripts/show1/show1.json" in keys

def test_upload_ingestion_outputs_encodes_utf8():
    client, s3 = make_client()
    client.upload_ingestion_outputs("show1", "Македонија", "{}")
    txt_call = next(c for c in s3.put_object.call_args_list if c.kwargs["Key"].endswith(".txt"))
    assert txt_call.kwargs["Body"] == "Македонија".encode("utf-8")


# ---------------------------------------------------------------------------
# upload_nemo_outputs
# ---------------------------------------------------------------------------

def test_upload_nemo_outputs_walks_directory():
    client, s3 = make_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "file1.rttm"), "w").close()
        open(os.path.join(tmpdir, "file2.pkl"), "w").close()
        client.upload_nemo_outputs("show1", tmpdir)
        assert s3.upload_file.call_count == 2

def test_upload_nemo_outputs_key_prefix():
    client, s3 = make_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "output.rttm"), "w").close()
        client.upload_nemo_outputs("show1", tmpdir)
        key = s3.upload_file.call_args[0][2]
        assert key.startswith("nemo/show1/")


# ---------------------------------------------------------------------------
# nemo_outputs_exist
# ---------------------------------------------------------------------------

def test_nemo_outputs_exist_true():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {"Contents": [{"Key": "nemo/show1/file.rttm"}]}
    assert client.nemo_outputs_exist("show1") is True

def test_nemo_outputs_exist_false():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {}
    assert client.nemo_outputs_exist("show1") is False


# ---------------------------------------------------------------------------
# list_transcripts
# ---------------------------------------------------------------------------

def test_list_transcripts_returns_sorted_stems():
    client, s3 = make_client()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [
        {"Key": "transcripts/zebra/zebra.txt"},
        {"Key": "transcripts/alpha/alpha.txt"},
        {"Key": "transcripts/alpha/alpha.json"},
    ]}]
    s3.get_paginator.return_value = paginator
    result = client.list_transcripts()
    assert result == ["alpha", "zebra"]

def test_list_transcripts_ignores_non_txt():
    client, s3 = make_client()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [
        {"Key": "transcripts/show1/show1.json"},
        {"Key": "transcripts/show1/show1.txt"},
    ]}]
    s3.get_paginator.return_value = paginator
    assert client.list_transcripts() == ["show1"]

def test_list_transcripts_empty():
    client, s3 = make_client()
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]
    s3.get_paginator.return_value = paginator
    assert client.list_transcripts() == []


# ---------------------------------------------------------------------------
# load_transcript_pair
# ---------------------------------------------------------------------------

def _mock_read(content: str) -> MagicMock:
    body = MagicMock()
    body.read.return_value = content.encode("utf-8")
    return {"Body": body}

def test_load_transcript_pair_returns_txt_and_json():
    client, s3 = make_client()
    s3.get_object.side_effect = [
        _mock_read("transcript content"),
        _mock_read('{"summary": "test"}'),
    ]
    txt, meta = client.load_transcript_pair("show1")
    assert txt == "transcript content"
    assert meta == '{"summary": "test"}'

def test_load_transcript_pair_missing_txt_raises():
    client, s3 = make_client()
    s3.get_object.side_effect = Exception("not found")
    with pytest.raises(FileNotFoundError):
        client.load_transcript_pair("show1")

def test_load_transcript_pair_missing_json_returns_empty():
    client, s3 = make_client()
    s3.get_object.side_effect = [
        _mock_read("transcript content"),
        Exception("not found"),
    ]
    txt, meta = client.load_transcript_pair("show1")
    assert txt == "transcript content"
    assert meta == ""


# ---------------------------------------------------------------------------
# get_audio_url
# ---------------------------------------------------------------------------

def test_get_audio_url_skips_txt_and_json():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.json"},
        {"Key": "transcripts/show1/show1.wav"},
    ]}
    s3.generate_presigned_url.return_value = "http://presigned-url"
    url = client.get_audio_url("show1")
    assert url == "http://presigned-url"
    s3.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": "transcripts/show1/show1.wav"},
        ExpiresIn=3600,
    )

def test_get_audio_url_no_audio_raises():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.json"},
    ]}
    with pytest.raises(StopIteration):
        client.get_audio_url("show1")


# ---------------------------------------------------------------------------
# download_audio
# ---------------------------------------------------------------------------

def test_download_audio_returns_local_path():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
        {"Key": "transcripts/show1/show1.json"},
        {"Key": "transcripts/show1/show1.wav"},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        result = client.download_audio("show1", tmpdir)
        assert result == os.path.join(tmpdir, "show1.wav")
        s3.download_file.assert_called_once_with(
            "test-bucket", "transcripts/show1/show1.wav", result
        )

def test_download_audio_no_audio_raises():
    client, s3 = make_client()
    s3.list_objects_v2.return_value = {"Contents": [
        {"Key": "transcripts/show1/show1.txt"},
    ]}
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(StopIteration):
            client.download_audio("show1", tmpdir)
