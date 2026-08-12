"""Turning a recording into a searchable transcript
(diarization, speaker recognition, transcription, enrichment and persistence).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field

from application.progress import LoggingProgress, ProgressReporter
from domain.document import transcript_key
from domain.errors import DiarizationNotCached
from domain.metadata import SessionDetails, TranscriptMetadata
from domain.ports import (
    AudioDiarizer,
    KnowledgeBase,
    SpeakerDirectory,
    SpeechTranscriber,
    TextCompleter,
    TranscriptArchive,
    TranscriptMetadataStore,
)
from domain.speaker import (
    SpeakerClustering,
    SpeakerResolution,
    SpeakerResolutions,
)
from domain.speaker_identification import identify_speakers
from domain.summarization import summarize
from domain.transcript import Transcript


@dataclass(frozen=True)
class IngestionRequest:
    stem: str
    session: SessionDetails = field(default_factory=SessionDetails)
    known_speakers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IngestionResult:
    stem: str
    transcript_txt: str
    metadata_json: str
    resolution_report: str


class IngestTranscriptUseCase:

    def __init__(
        self,
        archive: TranscriptArchive,
        speakers: SpeakerDirectory,
        metadata_store: TranscriptMetadataStore,
        knowledge_base: KnowledgeBase,
        diarizer: AudioDiarizer,
        transcriber: SpeechTranscriber,
        completer: TextCompleter,
        speaker_similarity_threshold: float,
        language: str,
    ) -> None:
        self.archive = archive
        self.speakers = speakers
        self.metadata_store = metadata_store
        self.knowledge_base = knowledge_base
        self.diarizer = diarizer
        self.transcriber = transcriber
        self.completer = completer
        self.speaker_similarity_threshold = speaker_similarity_threshold
        self.language = language

    async def ingest_audio(
        self,
        audio_path: str,
        request: IngestionRequest,
        progress: ProgressReporter | None = None,
    ) -> IngestionResult:
        """Full pipeline for a freshly uploaded recording."""
        progress = progress or LoggingProgress()
        work_dir = tempfile.mkdtemp(prefix="pipeline_")
        try:
            progress.report("Step 1/3: Saving audio to MinIO...")
            audio_key = await asyncio.to_thread(self.archive.store_audio, audio_path)
            progress.report(f"Saved: {audio_key}")

            progress.report("Step 2/3: Running NeMo diarization...")
            num_segments = await asyncio.to_thread(self.diarizer.split, audio_path, work_dir)
            diarization_dir = await asyncio.to_thread(self.diarizer.diarize, work_dir)
            progress.report(f"Diarization complete ({num_segments} segment(s))")

            try:
                await asyncio.to_thread(
                    self.archive.store_diarization, request.stem, diarization_dir
                )
                progress.report("NeMo outputs cached to MinIO")
            except Exception as exc:
                progress.report(f"NeMo caching failed (non-fatal): {exc}")

            progress.report("Step 3/3: Processing transcript...")
            return await self._process(work_dir, diarization_dir, num_segments, request, progress)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def resume(
        self,
        request: IngestionRequest,
        progress: ProgressReporter | None = None,
    ) -> IngestionResult:
        """Re-run everything after diarization, reusing the cached NeMo outputs."""
        progress = progress or LoggingProgress()
        work_dir = tempfile.mkdtemp(prefix="resume_")
        try:
            progress.report(f"Step 1/2: Checking cached NeMo outputs for '{request.stem}'...")
            if not await asyncio.to_thread(self.archive.has_diarization, request.stem):
                raise DiarizationNotCached(request.stem)

            diarization_dir = os.path.join(work_dir, "oracle_vad")
            await asyncio.to_thread(
                self.archive.fetch_diarization, request.stem, diarization_dir
            )
            progress.report("NeMo outputs downloaded")

            audio_path = await asyncio.to_thread(
                self.archive.download_audio, request.stem, work_dir
            )
            progress.report("Audio downloaded")

            num_segments = await asyncio.to_thread(self.diarizer.split, audio_path, work_dir)
            progress.report(f"Step 2/2: Processing transcript ({num_segments} segment(s))...")

            return await self._process(work_dir, diarization_dir, num_segments, request, progress)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    # --- shared pipeline ---------------------------------------------------

    async def _process(
        self,
        work_dir: str,
        diarization_dir: str,
        num_segments: int,
        request: IngestionRequest,
        progress: ProgressReporter,
    ) -> IngestionResult:
        progress.report("Clustering speakers...")
        clustering = await asyncio.to_thread(self.diarizer.cluster, diarization_dir)
        progress.report(f"Found {len(clustering)} speaker(s) via HDBSCAN")

        progress.report("Matching speakers against database...")
        resolutions = await self._match_against_directory(clustering)
        progress.report(f"DB matching: {resolutions.matched_count}/{len(clustering)} matched")

        progress.report("Transcribing audio...")
        transcript = Transcript(await asyncio.to_thread(
            self.transcriber.transcribe, work_dir, diarization_dir, clustering, num_segments
        ))
        progress.report(f"Transcription done: {len(transcript)} turns")

        resolutions = await self._name_unknown_speakers(transcript, resolutions, request, progress)
        speaker_names = resolutions.name_mapping()

        await self._remember_speakers(clustering, resolutions, transcript, request.stem)

        progress.report("Generating metadata & summary...")
        summary = await summarize(transcript, speaker_names, self.completer, self.language)
        metadata = TranscriptMetadata.build(transcript, speaker_names, request.session, summary)
        transcript_txt = transcript.render(speaker_names)
        metadata_json = json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2)

        progress.report("Uploading outputs to MinIO...")
        await asyncio.to_thread(
            self.archive.store_outputs, request.stem, transcript_txt, metadata_json
        )

        file_path = transcript_key(request.stem)
        await asyncio.to_thread(
            self.metadata_store.save, request.stem, file_path, metadata.to_dict()
        )

        progress.report("Inserting into knowledge graph...")
        await self.knowledge_base.insert_document(
            transcript_txt, request.stem, file_path, metadata_json
        )
        progress.report("Knowledge graph updated")

        return IngestionResult(
            stem=request.stem,
            transcript_txt=transcript_txt,
            metadata_json=metadata_json,
            resolution_report=resolutions.report(),
        )

    async def _match_against_directory(self, clustering: SpeakerClustering) -> SpeakerResolutions:
        resolutions = {}
        for speaker_id, centroid in clustering.centroids.items():
            match = await asyncio.to_thread(
                self.speakers.match, centroid, self.speaker_similarity_threshold
            )
            resolutions[speaker_id] = (
                SpeakerResolution.matched(speaker_id, match) if match
                else SpeakerResolution.unknown(speaker_id)
            )
        return SpeakerResolutions(items=resolutions)

    async def _name_unknown_speakers(
        self,
        transcript: Transcript,
        resolutions: SpeakerResolutions,
        request: IngestionRequest,
        progress: ProgressReporter,
    ) -> SpeakerResolutions:
        unknown_ids = resolutions.unresolved_ids
        if not unknown_ids:
            progress.report("All speakers matched from DB — LLM step skipped")
            return resolutions

        progress.report(f"LLM identifying {len(unknown_ids)} unknown speaker(s)...")
        try:
            identities = await identify_speakers(
                transcript, unknown_ids, request.known_speakers, self.completer
            )
            progress.report(f"LLM identified: {list(identities.values())}")
        except Exception as exc:
            progress.report(f"LLM identification error (non-fatal): {exc}")
            return resolutions

        return resolutions.with_llm_identities(identities)

    async def _remember_speakers(
        self,
        clustering: SpeakerClustering,
        resolutions: SpeakerResolutions,
        transcript: Transcript,
        stem: str,
    ) -> None:
        """Feed every named speaker back into the directory so the next run recognises them."""
        for resolution in resolutions:
            centroid = clustering.centroids.get(resolution.speaker_id)
            if not resolution.is_resolved or centroid is None:
                continue
            await asyncio.to_thread(
                self.speakers.remember,
                name=resolution.name,
                embedding=centroid,
                notes=f"Auto-added from {stem}",
                files=1,
                turns=transcript.turn_count_for(resolution.speaker_id),
            )
