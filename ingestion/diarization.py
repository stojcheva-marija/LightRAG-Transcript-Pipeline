from __future__ import annotations

import json
import logging
import math
import os
import pickle
from collections import defaultdict

import hdbscan
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler

from utils.progress import Progress

logger = logging.getLogger(__name__)


def split_audio(
    audio_path: str,
    work_dir: str,
    segment_minutes: int,
    reporter: Progress = Progress(),
) -> int:
    reporter(5, "Loading audio file...")
    audio, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    segment_length = segment_minutes * 60
    num_segments = math.ceil(total_duration / segment_length)

    segments_dir = os.path.join(work_dir, "audio_segments")
    os.makedirs(segments_dir, exist_ok=True)

    reporter(10, f"Splitting into {num_segments} segments...")
    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, total_duration)
        sf.write(os.path.join(segments_dir, f"segment_{i+1}.wav"), audio[int(start * sr):int(end * sr)], sr)

    return num_segments


def create_manifest(work_dir: str) -> str:
    segments_dir = os.path.join(work_dir, "audio_segments")
    manifest_path = os.path.join(work_dir, "manifest.jsonl")

    with open(manifest_path, "w") as f:
        for seg_file in sorted(os.listdir(segments_dir)):
            if seg_file.endswith(".wav"):
                seg_path = os.path.join(segments_dir, seg_file)
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


def load_embeddings_and_cluster(
    output_dir: str,
    reporter: Progress = Progress(),
) -> tuple[dict, dict]:
    reporter(55, "Loading speaker embeddings...")

    embeddings_path = os.path.join(output_dir, "speaker_outputs", "embeddings", "subsegments_scale4_embeddings.pkl")
    cluster_labels_path = os.path.join(output_dir, "speaker_outputs", "subsegments_scale4_cluster.label")

    with open(embeddings_path, "rb") as f:
        embeddings_data = pickle.load(f)

    embeddings = [vec for _, tensor in embeddings_data.items() for vec in tensor.tolist()]

    cluster_df = pd.read_csv(cluster_labels_path, delimiter=" ")
    cluster_df.columns = ["segment_name", "offset", "duration", "label"]

    grouped_embeddings: dict = defaultdict(list)
    for (segment_name, label), vec in zip(cluster_df[["segment_name", "label"]].values, embeddings):
        grouped_embeddings[(label, segment_name)].append(vec)

    grouped_json_data = [
        {
            "segment_name": segment_name,
            "label": label,
            "aggregated_embedding": np.mean(vecs, axis=0).tolist(),
        }
        for (label, segment_name), vecs in grouped_embeddings.items()
    ]

    reporter(60, "Running HDBSCAN clustering...")
    agg_embeddings = np.array([item["aggregated_embedding"] for item in grouped_json_data])
    normed_embeddings = agg_embeddings / np.linalg.norm(agg_embeddings, axis=1, keepdims=True)
    scaled_embeddings = StandardScaler().fit_transform(normed_embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
    )
    labels = clusterer.fit_predict(scaled_embeddings)

    if np.all(labels == -1):
        logger.warning("HDBSCAN assigned all points to noise — falling back to single cluster")
        labels = np.zeros(len(scaled_embeddings), dtype=int)

    shifted_labels = np.where(labels >= 0, labels + 1, labels)
    for i, item in enumerate(grouped_json_data):
        item["hdbscan_cluster_label"] = f"speaker_{shifted_labels[i]}"

    speaker_remap = {
        (item["segment_name"], item["label"]): item["hdbscan_cluster_label"]
        for item in grouped_json_data
    }

    speaker_centroids = {
        f"speaker_{label + 1}": (c := normed_embeddings[labels == label].mean(axis=0)) / np.linalg.norm(c)
        for label in set(labels)
        if label != -1
    }

    return speaker_remap, speaker_centroids
