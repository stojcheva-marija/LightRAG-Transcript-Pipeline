"""Turning NeMo's raw per-segment speaker embeddings into recording-wide speakers."""

from __future__ import annotations

import logging
import os
import pickle
from collections import defaultdict

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from domain.speaker import SpeakerClustering

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = ("speaker_outputs", "embeddings", "subsegments_scale4_embeddings.pkl")
CLUSTER_LABELS_PATH = ("speaker_outputs", "subsegments_scale4_cluster.label")


def cluster_speaker_embeddings(diarization_dir: str) -> SpeakerClustering:
    """Re-cluster NeMo's per-segment speaker labels into recording-wide speakers.

    NeMo labels speakers per segment, so the same person gets different labels
    in different segments. HDBSCAN over the segment embeddings stitches those
    back together and yields one centroid per actual speaker.
    """
    embeddings_path = os.path.join(diarization_dir, *EMBEDDINGS_PATH)
    cluster_labels_path = os.path.join(diarization_dir, *CLUSTER_LABELS_PATH)

    with open(embeddings_path, "rb") as f:
        embeddings_data = pickle.load(f)

    embeddings = [vec for _, tensor in embeddings_data.items() for vec in tensor.tolist()]

    cluster_df = pd.read_csv(cluster_labels_path, delimiter=" ")
    cluster_df.columns = ["segment_name", "offset", "duration", "label"]

    grouped_embeddings: dict = defaultdict(list)
    for (segment_name, label), vec in zip(cluster_df[["segment_name", "label"]].values, embeddings):
        grouped_embeddings[(label, segment_name)].append(vec)

    grouped = [
        {
            "segment_name": segment_name,
            "label": label,
            "aggregated_embedding": np.mean(vecs, axis=0).tolist(),
        }
        for (label, segment_name), vecs in grouped_embeddings.items()
    ]

    agg_embeddings = np.array([item["aggregated_embedding"] for item in grouped])
    normed_embeddings = agg_embeddings / np.linalg.norm(agg_embeddings, axis=1, keepdims=True)
    scaled_embeddings = StandardScaler().fit_transform(normed_embeddings)

    labels = _fit_labels(scaled_embeddings)

    shifted_labels = np.where(labels >= 0, labels + 1, labels)
    for i, item in enumerate(grouped):
        item["hdbscan_cluster_label"] = f"speaker_{shifted_labels[i]}"

    remap = {
        (item["segment_name"], item["label"]): item["hdbscan_cluster_label"]
        for item in grouped
    }
    centroids = {
        f"speaker_{label + 1}": (c := normed_embeddings[labels == label].mean(axis=0))
        / np.linalg.norm(c)
        for label in set(labels)
        if label != -1
    }

    return SpeakerClustering(remap=remap, centroids=centroids)


def _fit_labels(scaled_embeddings: np.ndarray) -> np.ndarray:
    if len(scaled_embeddings) < 2:
        logger.warning("Only 1 embedding point — falling back to single cluster")
        return np.zeros(len(scaled_embeddings), dtype=int)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.3,
    )
    labels = clusterer.fit_predict(scaled_embeddings)

    if np.all(labels == -1):
        logger.warning("HDBSCAN assigned all points to noise — falling back to single cluster")
        return np.zeros(len(scaled_embeddings), dtype=int)
    return labels
