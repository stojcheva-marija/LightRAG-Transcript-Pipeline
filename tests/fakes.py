"""In-memory stand-ins for every port, so use cases can be tested without I/O."""

from __future__ import annotations

import numpy as np

from domain.errors import AudioNotFound, TranscriptNotFound
from domain.retrieval import RetrievedAnswer
from domain.speaker import SpeakerClustering, SpeakerMatch
from domain.transcript import SpeakerTurn


class FakeCompleter:
    """Returns scripted responses in order and records what it was asked."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    async def complete(self, prompt, *, system=None, json_object=False,
                       temperature=0.0, max_tokens=None) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


class FakeArchive:
    def __init__(self, stems: list[str] | None = None) -> None:
        self.stems = list(stems or [])
        self.transcripts: dict[str, tuple[str, str]] = {}
        self.diarization: set[str] = set()
        self.stored_outputs: dict[str, tuple[str, str]] = {}
        self.stored_audio: list[str] = []
        self.stored_diarization: list[tuple[str, str]] = []
        self.missing_audio: set[str] = set()

    def list_stems(self) -> list[str]:
        return sorted(self.stems)

    def store_audio(self, local_path: str) -> str:
        self.stored_audio.append(local_path)
        return f"transcripts/x/{local_path.split('/')[-1]}"

    def store_outputs(self, stem, transcript_txt, metadata_json) -> None:
        self.stored_outputs[stem] = (transcript_txt, metadata_json)
        self.stems.append(stem)

    def load_transcript(self, stem) -> tuple[str, str]:
        if stem not in self.transcripts:
            raise TranscriptNotFound(stem)
        return self.transcripts[stem]

    def download_audio(self, stem, target_dir) -> str:
        if stem in self.missing_audio:
            raise AudioNotFound(stem)
        return f"{target_dir}/{stem}.wav"

    def audio_url(self, stem, expires_in=3600) -> str:
        if stem in self.missing_audio or stem not in self.stems:
            raise AudioNotFound(stem)
        return f"https://audio.test/{stem}.wav"

    def store_diarization(self, stem, source_dir) -> None:
        self.stored_diarization.append((stem, source_dir))
        self.diarization.add(stem)

    def has_diarization(self, stem) -> bool:
        return stem in self.diarization

    def fetch_diarization(self, stem, target_dir) -> str:
        return target_dir


class FakeMetadataStore:
    def __init__(self, speakers: list[str] | None = None) -> None:
        self.saved: dict[str, tuple[str, dict]] = {}
        self.speakers = list(speakers or [])
        self.setup_calls = 0

    def setup(self) -> None:
        self.setup_calls += 1

    def save(self, doc_id, file_path, metadata) -> None:
        self.saved[doc_id] = (file_path, metadata)

    def known_speaker_names(self) -> list[str]:
        return self.speakers


class FakeSpeakerDirectory:
    """Matches are keyed by speaker position, the value ``make_clustering``
    writes into the centroid's first component."""

    def __init__(self, matches: dict[int, SpeakerMatch] | None = None) -> None:
        self.matches = matches or {}
        self.remembered: list[dict] = []

    def setup(self) -> None:
        pass

    def match(self, embedding, threshold) -> SpeakerMatch | None:
        match = self.matches.get(int(embedding[0]))
        return match if match and match.similarity >= threshold else None

    def remember(self, name, embedding, notes="", files=0, turns=0) -> None:
        self.remembered.append({"name": name, "notes": notes, "files": files, "turns": turns})


class FakeKnowledgeBase:
    def __init__(self, answer: RetrievedAnswer | None = None, insert_results=None) -> None:
        self.answer = answer or RetrievedAnswer(answer="")
        self.inserted: list[tuple[str, str, str, str]] = []
        self.insert_results = list(insert_results) if insert_results is not None else None
        self.initialized = False
        self.finalized = False
        self.queries: list[tuple[str, object]] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def insert_document(self, transcript, doc_id, file_path, metadata="") -> bool:
        self.inserted.append((transcript, doc_id, file_path, metadata))
        if self.insert_results is None:
            return True
        return self.insert_results.pop(0)

    async def query(self, question, keywords, history=None) -> RetrievedAnswer:
        self.queries.append((question, keywords))
        return self.answer

    async def finalize(self) -> None:
        self.finalized = True


class FakeDiarizer:
    def __init__(self, clustering: SpeakerClustering, num_segments: int = 1) -> None:
        self.clustering = clustering
        self.num_segments = num_segments
        self.diarized = False

    def split(self, audio_path, work_dir) -> int:
        return self.num_segments

    def diarize(self, work_dir) -> str:
        self.diarized = True
        return f"{work_dir}/oracle_vad"

    def cluster(self, diarization_dir) -> SpeakerClustering:
        return self.clustering


class FakeTranscriber:
    def __init__(self, turns: list[SpeakerTurn]) -> None:
        self.turns = turns

    def transcribe(self, work_dir, diarization_dir, clustering, num_segments):
        return list(self.turns)


class RecordingProgress:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def report(self, message: str) -> None:
        self.messages.append(message)

    def __contains__(self, fragment: str) -> bool:
        return any(fragment in message for message in self.messages)


def make_clustering(*speaker_ids: str) -> SpeakerClustering:
    """One centroid per speaker, first component encoding the speaker index."""
    centroids = {}
    for i, speaker_id in enumerate(speaker_ids, start=1):
        vec = np.zeros(4)
        vec[0] = float(i)
        centroids[speaker_id] = vec
    return SpeakerClustering(remap={}, centroids=centroids)
