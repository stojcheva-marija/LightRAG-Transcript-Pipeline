"""Splitting a long recording into fixed-length segments NeMo can diarize."""

from __future__ import annotations

import json
import logging
import math
import os

import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

SEGMENTS_DIRNAME = "audio_segments"
MANIFEST_FILENAME = "manifest.jsonl"


def segments_dir(work_dir: str) -> str:
    return os.path.join(work_dir, SEGMENTS_DIRNAME)


def split_audio(audio_path: str, work_dir: str, segment_minutes: int) -> int:
    """Cut the recording into fixed-length wav segments. Returns the count."""
    audio, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    segment_length = segment_minutes * 60
    num_segments = math.ceil(total_duration / segment_length)

    target_dir = segments_dir(work_dir)
    os.makedirs(target_dir, exist_ok=True)

    logger.info("Splitting audio into %d segment(s)", num_segments)
    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, total_duration)
        sf.write(
            os.path.join(target_dir, f"segment_{i + 1}.wav"),
            audio[int(start * sr):int(end * sr)],
            sr,
        )

    return num_segments


def create_manifest(work_dir: str) -> str:
    """Write the NeMo inference manifest describing every segment."""
    source_dir = segments_dir(work_dir)
    manifest_path = os.path.join(work_dir, MANIFEST_FILENAME)

    with open(manifest_path, "w") as f:
        for seg_file in sorted(os.listdir(source_dir)):
            if seg_file.endswith(".wav"):
                seg_path = os.path.join(source_dir, seg_file)
                info = sf.info(seg_path)
                entry = {
                    "audio_filepath": seg_path,
                    "offset": 0,
                    "duration": info.frames / info.samplerate,
                    "label": "infer",
                    "text": "-",
                    "rttm_filepath": None,
                }
                f.write(json.dumps(entry) + "\n")
    return manifest_path
