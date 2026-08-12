from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3

from config.settings import MinIOConfig
from domain.errors import AudioNotFound, TranscriptNotFound

logger = logging.getLogger(__name__)


class MinIOTranscriptArchive:
    """``TranscriptArchive`` backed by S3-compatible object storage.

    Layout:
        transcripts/{stem}/   audio file + {stem}.txt + {stem}.json
        nemo/{stem}/          cached diarization outputs
    """

    def __init__(self, config: MinIOConfig) -> None:
        self.bucket = config.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
        )

    @staticmethod
    def _transcript_prefix(stem: str) -> str:
        return f"transcripts/{stem}/"

    @staticmethod
    def _nemo_prefix(stem: str) -> str:
        return f"nemo/{stem}/"

    # --- audio -------------------------------------------------------------

    def store_audio(self, local_path: str) -> str:
        path = Path(local_path)
        key = f"{self._transcript_prefix(path.stem)}{path.name}"
        self.client.upload_file(local_path, self.bucket, key)
        logger.info("Uploaded audio %s -> %s", path.name, key)
        return key

    def _audio_key(self, stem: str) -> str:
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=self._transcript_prefix(stem)
        )
        for obj in response.get("Contents", []):
            if not obj["Key"].endswith((".txt", ".json")):
                return obj["Key"]
        raise AudioNotFound(stem)

    def download_audio(self, stem: str, target_dir: str) -> str:
        audio_key = self._audio_key(stem)
        local_path = os.path.join(target_dir, audio_key.split("/")[-1])
        self.client.download_file(self.bucket, audio_key, local_path)
        logger.info("Downloaded audio %s -> %s", audio_key, local_path)
        return local_path

    def audio_url(self, stem: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": self._audio_key(stem)},
            ExpiresIn=expires_in,
        )

    # --- transcripts -------------------------------------------------------

    def store_outputs(self, stem: str, transcript_txt: str, metadata_json: str) -> None:
        prefix = self._transcript_prefix(stem)
        for key, content in [
            (f"{prefix}{stem}.txt", transcript_txt),
            (f"{prefix}{stem}.json", metadata_json),
        ]:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content.encode("utf-8"))
        logger.info("Uploaded ingestion outputs for stem=%s", stem)

    def list_stems(self) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        stems: set[str] = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix="transcripts/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".txt"):
                    stems.add(key.split("/")[1])
        return sorted(stems)

    def load_transcript(self, stem: str) -> tuple[str, str]:
        prefix = self._transcript_prefix(stem)
        try:
            transcript_text = self._read_text(f"{prefix}{stem}.txt")
        except Exception:
            raise TranscriptNotFound(stem)
        try:
            metadata_text = self._read_text(f"{prefix}{stem}.json")
        except Exception:
            logger.warning("No metadata .json found for stem '%s', continuing without it.", stem)
            metadata_text = ""
        return transcript_text, metadata_text

    # --- cached diarization ------------------------------------------------

    def store_diarization(self, stem: str, source_dir: str) -> None:
        prefix = self._nemo_prefix(stem)
        uploaded = 0
        for root, _, files in os.walk(source_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                key = f"{prefix}{os.path.relpath(local_path, source_dir)}"
                try:
                    self.client.upload_file(local_path, self.bucket, key)
                    uploaded += 1
                except Exception as exc:
                    logger.error("Failed to upload NeMo file %s: %s", key, exc)
        logger.info("Uploaded %d NeMo output files for stem=%s", uploaded, stem)

    def has_diarization(self, stem: str) -> bool:
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=self._nemo_prefix(stem), MaxKeys=1
        )
        return bool(response.get("Contents"))

    def fetch_diarization(self, stem: str, target_dir: str) -> str:
        prefix = self._nemo_prefix(stem)
        paginator = self.client.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local_path = os.path.join(target_dir, key[len(prefix):])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.client.download_file(self.bucket, key, local_path)
                downloaded += 1
        logger.info("Downloaded %d NeMo output files for stem=%s", downloaded, stem)
        return target_dir

    def _read_text(self, key: str) -> str:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")
