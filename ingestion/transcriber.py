from __future__ import annotations

import logging
import os
from operator import itemgetter

import librosa
import numpy as np

from models.asr import WhisperPipeline
from utils.progress import Progress

logger = logging.getLogger(__name__)

MAX_CHUNK_SECONDS = 25
MIN_SEGMENT_DURATION = 1.0
MERGE_GAP_SECONDS = 2.0
AUDIO_PADDING_SECONDS = 0.3
MIN_TEXT_LENGTH = 3


def _parse_rttm(rttm_path: str, segment_name: str, speaker_remap: dict) -> list[dict]:
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                end = start + float(parts[4])
                speaker = speaker_remap.get((segment_name, parts[7]), parts[7])
                if end > start:
                    segments.append({"start": start, "end": end, "speaker": speaker, "duration": end - start})
    segments.sort(key=itemgetter("start"))
    return segments


def _split_long_segments(speaker_segs: list[dict]) -> list[dict]:
    result = []
    for seg in speaker_segs:
        if seg["duration"] <= MAX_CHUNK_SECONDS:
            result.append(seg)
        else:
            num_chunks = int(np.ceil(seg["duration"] / MAX_CHUNK_SECONDS))
            chunk_duration = seg["duration"] / num_chunks
            for i in range(num_chunks):
                chunk_start = seg["start"] + i * chunk_duration
                chunk_end = min(seg["start"] + (i + 1) * chunk_duration, seg["end"])
                result.append({"start": chunk_start, "end": chunk_end, "speaker": seg["speaker"], "duration": chunk_end - chunk_start})
    return result


def _merge_or_append(all_turns: list[dict], turn: dict) -> None:
    if (all_turns
            and all_turns[-1]["speaker"] == turn["speaker"]
            and turn["start"] - all_turns[-1]["end"] < MERGE_GAP_SECONDS):
        all_turns[-1]["end"] = turn["end"]
        all_turns[-1]["text"] += " " + turn["text"]
    else:
        all_turns.append(turn)


def transcribe_segments(
    work_dir: str,
    output_dir: str,
    speaker_remap: dict,
    num_segments: int,
    whisper: WhisperPipeline,
    segment_minutes: int,
    reporter: Progress = Progress(),
) -> list[dict]:
    reporter(65, "Loading Whisper model...")
    pipe = whisper.pipe

    all_turns = []
    segment_seconds = segment_minutes * 60

    for seg_num in range(1, num_segments + 1):
        pct = 65 + int((seg_num / num_segments) * 20)
        reporter(pct, f"Transcribing segment {seg_num}/{num_segments}...")

        rttm_path = os.path.join(output_dir, "pred_rttms", f"segment_{seg_num}.rttm")
        audio_path = os.path.join(work_dir, "audio_segments", f"segment_{seg_num}.wav")
        segment_name = f"segment_{seg_num}"

        if not os.path.exists(rttm_path):
            logger.warning(f"RTTM not found: {rttm_path}")
            continue

        audio, sr = librosa.load(audio_path, sr=16000)
        offset = (seg_num - 1) * segment_seconds

        speaker_segs = _parse_rttm(rttm_path, segment_name, speaker_remap)
        short_segs = _split_long_segments(speaker_segs)

        for seg in short_segs:
            if seg["duration"] < MIN_SEGMENT_DURATION:
                continue

            padding = int(AUDIO_PADDING_SECONDS * sr)
            seg_audio = audio[
                max(0, int(seg["start"] * sr) - padding):
                min(len(audio), int(seg["end"] * sr) + padding)
            ]

            try:
                output = pipe(
                    seg_audio,
                    generate_kwargs={"language": "mk", "task": "transcribe"},
                    return_timestamps=False,
                )
                text = output["text"].strip() if isinstance(output, dict) else ""
                if len(text) < MIN_TEXT_LENGTH:
                    continue

                _merge_or_append(all_turns, {
                    "start": seg["start"] + offset,
                    "end": seg["end"] + offset,
                    "speaker": seg["speaker"],
                    "text": text,
                })

            except Exception as e:
                logger.warning(f"Transcription error: {e}")

    return all_turns
