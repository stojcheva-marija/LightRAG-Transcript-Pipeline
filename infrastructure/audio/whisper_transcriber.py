from __future__ import annotations

import logging
import os

import librosa

from domain.speaker import SpeakerClustering
from domain.transcript import SpeakerTurn, merge_adjacent_turns
from infrastructure.audio.rttm import (
    MIN_SEGMENT_DURATION,
    parse_rttm,
    split_long_segments,
)
from infrastructure.audio.segmenter import segments_dir

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
AUDIO_PADDING_SECONDS = 0.3
MIN_TEXT_LENGTH = 3


class WhisperTranscriber:
    """``SpeechTranscriber`` backed by a Whisper checkpoint.

    ``model_id`` and ``language`` travel together: point them at a checkpoint
    fine-tuned for the language you are transcribing.

    The model is loaded on first use — constructing this is cheap.
    """

    def __init__(self, model_id: str, language: str, segment_minutes: int) -> None:
        self.model_id = model_id
        self.language = language
        self.segment_minutes = segment_minutes
        self._pipe = None

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._load()
        return self._pipe

    def _load(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline

        # Force CPU: MPS has known bugs with Whisper timestamp and beam search
        torch_dtype = torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to("cpu")
        processor = AutoProcessor.from_pretrained(self.model_id)

        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            batch_size=1,
            torch_dtype=torch_dtype,
            device="cpu",
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        logger.info("Whisper model loaded: %s", self.model_id)
        return pipe

    def transcribe(
        self,
        work_dir: str,
        diarization_dir: str,
        clustering: SpeakerClustering,
        num_segments: int,
    ) -> list[SpeakerTurn]:
        turns: list[SpeakerTurn] = []
        segment_seconds = self.segment_minutes * 60

        for seg_num in range(1, num_segments + 1):
            rttm_path = os.path.join(diarization_dir, "pred_rttms", f"segment_{seg_num}.rttm")
            audio_path = os.path.join(segments_dir(work_dir), f"segment_{seg_num}.wav")
            segment_name = f"segment_{seg_num}"

            if not os.path.exists(rttm_path):
                logger.warning("RTTM not found: %s", rttm_path)
                continue

            logger.info("Transcribing segment %d/%d", seg_num, num_segments)
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            offset = (seg_num - 1) * segment_seconds

            speech = split_long_segments(parse_rttm(rttm_path, segment_name, clustering.remap))
            for seg in speech:
                if seg["duration"] < MIN_SEGMENT_DURATION:
                    continue
                text = self._transcribe_span(audio, sr, seg["start"], seg["end"])
                if text is None:
                    continue
                turns.append(SpeakerTurn(
                    speaker=seg["speaker"],
                    start=seg["start"] + offset,
                    end=seg["end"] + offset,
                    text=text,
                ))

        return merge_adjacent_turns(turns)

    def _transcribe_span(self, audio, sr: int, start: float, end: float) -> str | None:
        padding = int(AUDIO_PADDING_SECONDS * sr)
        span = audio[
            max(0, int(start * sr) - padding):
            min(len(audio), int(end * sr) + padding)
        ]
        try:
            output = self.pipe(
                span,
                generate_kwargs={"language": self.language, "task": "transcribe"},
                return_timestamps=False,
            )
        except Exception as exc:
            logger.warning("Transcription error: %s", exc)
            return None

        text = output["text"].strip() if isinstance(output, dict) else ""
        return text if len(text) >= MIN_TEXT_LENGTH else None
