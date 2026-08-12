"""Reading diarization output and shaping it into transcribable stretches."""

from __future__ import annotations

from operator import itemgetter

import numpy as np

MAX_CHUNK_SECONDS = 25
MIN_SEGMENT_DURATION = 1.0
MERGE_GAP_SECONDS = 2.0

RttmSegment = dict


def parse_rttm(rttm_path: str, segment_name: str, speaker_remap: dict) -> list[RttmSegment]:
    """Read an RTTM file, mapping per-segment speaker labels to global ones."""
    segments: list[RttmSegment] = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                start = float(parts[3])
                end = start + float(parts[4])
                speaker = speaker_remap.get((segment_name, parts[7]), parts[7])
                if end > start:
                    segments.append(
                        {"start": start, "end": end, "speaker": speaker, "duration": end - start}
                    )
    segments.sort(key=itemgetter("start"))
    return segments


def split_long_segments(segments: list[RttmSegment]) -> list[RttmSegment]:
    """Cut anything longer than the ASR model's comfortable window into equal parts."""
    result: list[RttmSegment] = []
    for seg in segments:
        if seg["duration"] <= MAX_CHUNK_SECONDS:
            result.append(seg)
            continue
        num_chunks = int(np.ceil(seg["duration"] / MAX_CHUNK_SECONDS))
        chunk_duration = seg["duration"] / num_chunks
        for i in range(num_chunks):
            chunk_start = seg["start"] + i * chunk_duration
            chunk_end = min(seg["start"] + (i + 1) * chunk_duration, seg["end"])
            result.append({
                "start": chunk_start,
                "end": chunk_end,
                "speaker": seg["speaker"],
                "duration": chunk_end - chunk_start,
            })
    return result
