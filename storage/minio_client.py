from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3

from config.settings import Config

logger = logging.getLogger(__name__)


class MinIOClient:
    def __init__(self, config: Config) -> None:
        self.bucket = config.minio.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=config.minio.endpoint,
            aws_access_key_id=config.minio.access_key,
            aws_secret_access_key=config.minio.secret_key,
        )

    # transcripts/{stem}/  ← audio file + .txt transcript + .json metadata
    # nemo/{stem}/         ← NeMo diarization outputs

    @staticmethod
    def _transcript_prefix(stem: str) -> str:
        return f"transcripts/{stem}/"

    @staticmethod
    def _nemo_prefix(stem: str) -> str:
        return f"nemo/{stem}/"

    def upload_raw_audio(self, local_path: str) -> str:
        path = Path(local_path)
        key = f"{self._transcript_prefix(path.stem)}{path.name}"
        self.client.upload_file(local_path, self.bucket, key)
        logger.info("Uploaded audio %s -> %s", path.name, key)
        return key

    def upload_nemo_outputs(self, stem: str, nemo_output_dir: str) -> None:
        prefix = self._nemo_prefix(stem)
        uploaded = 0
        for root, _, files in os.walk(nemo_output_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                key = f"{prefix}{os.path.relpath(local_path, nemo_output_dir)}"
                try:
                    self.client.upload_file(local_path, self.bucket, key)
                    uploaded += 1
                except Exception as e:
                    logger.error("Failed to upload NeMo file %s: %s", key, e)
        logger.info("Uploaded %d NeMo output files for stem=%s", uploaded, stem)

    def upload_ingestion_outputs(self, stem: str, transcript_txt: str, metadata_json: str) -> None:
        prefix = self._transcript_prefix(stem)
        for key, content in [
            (f"{prefix}{stem}.txt", transcript_txt),
            (f"{prefix}{stem}.json", metadata_json),
        ]:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=content.encode("utf-8"))
        logger.info("Uploaded ingestion outputs for stem=%s", stem)

    def download_nemo_outputs(self, stem: str, target_dir: str) -> str:
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

    def nemo_outputs_exist(self, stem: str) -> bool:
        response = self.client.list_objects_v2(
            Bucket=self.bucket, Prefix=self._nemo_prefix(stem), MaxKeys=1
        )
        return bool(response.get("Contents"))

    def list_transcripts(self) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        stems: set[str] = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix="transcripts/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".txt"):
                    stems.add(key.split("/")[1])
        return sorted(stems)

    def load_transcript_pair(self, stem: str) -> tuple[str, str]:
        prefix = self._transcript_prefix(stem)
        try:
            transcript_text = self._read_text(f"{prefix}{stem}.txt")
        except Exception:
            raise FileNotFoundError(f"Transcript .txt not found for stem '{stem}'")
        try:
            metadata_text = self._read_text(f"{prefix}{stem}.json")
        except Exception:
            logger.warning("No metadata .json found for stem '%s', continuing without it.", stem)
            metadata_text = ""
        return transcript_text, metadata_text

    def download_audio(self, stem: str, target_dir: str) -> str:
        prefix = self._transcript_prefix(stem)
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        audio_key = next(
            obj["Key"] for obj in response.get("Contents", [])
            if not obj["Key"].endswith((".txt", ".json"))
        )
        filename = audio_key.split("/")[-1]
        local_path = os.path.join(target_dir, filename)
        self.client.download_file(self.bucket, audio_key, local_path)
        logger.info("Downloaded audio %s -> %s", audio_key, local_path)
        return local_path

    def get_audio_url(self, stem: str, expires_in: int = 3600) -> str:
        prefix = self._transcript_prefix(stem)
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        audio_key = next(
            obj["Key"] for obj in response.get("Contents", [])
            if not obj["Key"].endswith((".txt", ".json"))
        )
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": audio_key},
            ExpiresIn=expires_in,
        )

    def _read_text(self, key: str) -> str:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read().decode("utf-8")
