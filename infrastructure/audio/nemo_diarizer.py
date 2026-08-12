from __future__ import annotations

import os

import wget
from omegaconf import OmegaConf

from domain.speaker import SpeakerClustering
from infrastructure.audio.clustering import cluster_speaker_embeddings
from infrastructure.audio.segmenter import create_manifest, split_audio


DIARIZATION_DIRNAME = "oracle_vad"


class NemoDiarizer:
    """``AudioDiarizer`` backed by NVIDIA NeMo's neural diarizer."""

    def __init__(self, config_url: str, segment_minutes: int) -> None:
        self.config_url = config_url
        self.segment_minutes = segment_minutes

    def split(self, audio_path: str, work_dir: str) -> int:
        return split_audio(audio_path, work_dir, self.segment_minutes)

    def diarize(self, work_dir: str) -> str:
        # Imported lazily: NeMo pulls in torch and its model registry.
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer

        manifest_path = create_manifest(work_dir)
        output_dir = os.path.join(work_dir, DIARIZATION_DIRNAME)
        os.makedirs(output_dir, exist_ok=True)

        config = self._load_config(work_dir, manifest_path, output_dir)
        NeuralDiarizer(cfg=config).diarize()
        return output_dir

    def cluster(self, diarization_dir: str) -> SpeakerClustering:
        return cluster_speaker_embeddings(diarization_dir)

    def _load_config(self, work_dir: str, manifest_path: str, output_dir: str):
        config_path = os.path.join(work_dir, "diar_infer_meeting.yaml")
        if not os.path.exists(config_path):
            wget.download(self.config_url, config_path)

        config = OmegaConf.load(config_path)
        OmegaConf.update(config, "diarizer.msdd_model.model_path", "diar_msdd_telephonic", merge=True)
        OmegaConf.update(config, "diarizer.msdd_model.parameters.sigmoid_threshold", [0.7, 1.0], merge=True)
        OmegaConf.update(config, "diarizer.msdd_model.parameters.diar_window_length", 50, merge=True)
        OmegaConf.update(config, "diarizer.msdd_model.parameters.infer_batch_size", 25, merge=True)
        OmegaConf.update(config, "diarizer.msdd_model.parameters.seq_eval_mode", False, merge=True)
        OmegaConf.update(config, "diarizer.msdd_model.parameters.split_infer", True, merge=True)
        config.num_workers = 0
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = output_dir
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.1]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
        config.diarizer.clustering.parameters.oracle_num_speakers = False
        return config
