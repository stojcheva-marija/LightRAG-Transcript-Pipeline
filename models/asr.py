from __future__ import annotations

import logging
import os

import wget
from omegaconf import OmegaConf

from nemo.collections.asr.models.msdd_models import NeuralDiarizer

logger = logging.getLogger(__name__)


class WhisperPipeline:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._load()
        return self._pipe

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

        # Force CPU: MPS has known bugs with Whisper timestamp and beam search
        torch_dtype = torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to("cpu")
        processor = AutoProcessor.from_pretrained(self.model_id)

        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            batch_size=1,
            torch_dtype=torch_dtype,
            device="cpu",
            generate_kwargs={"language": "mk", "task": "transcribe"},
        )
        logger.info(f"Whisper model loaded: {self.model_id}")


class NemoDiarizer:
    def __init__(self, config_url: str) -> None:
        self.config_url = config_url

    def run(self, work_dir: str, manifest_path: str) -> str:
        output_dir = os.path.join(work_dir, "oracle_vad")
        os.makedirs(output_dir, exist_ok=True)

        config_path = os.path.join(work_dir, "diar_infer_meeting.yaml")
        if not os.path.exists(config_path):
            wget.download(self.config_url, config_path)

        config = OmegaConf.load(config_path)
        config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"
        config.num_workers = 0
        config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]
        config.diarizer.manifest_filepath = manifest_path
        config.diarizer.out_dir = output_dir
        config.diarizer.speaker_embeddings.model_path = "titanet_large"
        config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5, 1.25, 1.0, 0.75, 0.5]
        config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75, 0.625, 0.5, 0.375, 0.1]
        config.diarizer.speaker_embeddings.parameters.multiscale_weights = [1, 1, 1, 1, 1]
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        NeuralDiarizer(cfg=config).diarize()
        return output_dir
