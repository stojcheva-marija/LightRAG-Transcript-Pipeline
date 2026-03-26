from __future__ import annotations

import logging
import sys
import os
import json
import asyncio
import tempfile
import shutil
from pathlib import Path

import gradio as gr

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/gradio_app.log"),
    ],
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "ingestion"))

try:
    from config.settings import Config
    from chunkers.block import BlockChunker
    from llm.adapters import make_llm_func, make_embed_func, make_rerank_func
    from parsers.transcript import TranscriptParser
    from storage.metadata_repository import MetadataRepository
    from storage.minio_client import MinIOClient
    from rag.conversational_rag import ConversationalRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG modules not available — RAG tab will be disabled.")

try:
    from storage.speaker_repository import SpeakerRepository
    from ingestion.speaker_resolver import (
        resolve_speakers_from_db,
        llm_identify_speakers,
        build_final_speaker_mapping,
        generate_resolution_report,
    )
    from ingestion.metadata_builder import (
        generate_topics_and_summary_async,
        build_metadata_json,
        build_transcript_txt,
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    PREPROCESSING_AVAILABLE = False
    logger.warning(f"Preprocessing modules not fully available: {e}")

rag_instance: object | None = None
minio_instance: object | None = None


async def get_rag():
    global rag_instance, minio_instance
    if rag_instance is not None:
        return rag_instance, minio_instance
    if not RAG_AVAILABLE:
        return None, None

    config = Config.from_env()
    parser = TranscriptParser()
    metadata_repo = MetadataRepository(config.database)
    minio_instance = MinIOClient(config)

    chunker = BlockChunker(parser=parser, window_size=config.chunker.block_window_size)
    rag_instance = ConversationalRAG(
        config=config,
        chunker=chunker,
        llm_func=make_llm_func(config),
        embed_func=make_embed_func(config),
        rerank_func=make_rerank_func(config),
        metadata_repo=metadata_repo,
    )
    await rag_instance.initialize()
    return rag_instance, minio_instance


async def run_query(question, history):
    import re
    if not question.strip():
        return "Please enter a question.", "", history or []
    try:
        rag, minio = await get_rag()
        if rag is None:
            return "RAG system not available.", "", history or []

        answer, stem_times = await rag.query(question, mode="mix", history=history or [])
        answer = re.sub(r'\n+#{1,3}\s*References.*', '', answer, flags=re.DOTALL).strip()

        players = []
        for stem, start_seconds in stem_times:
            try:
                url = minio.get_audio_url(stem)
                players.append(
                    f'<audio controls style="width:100%;margin-bottom:8px" '
                    f'onloadedmetadata="if(!this._sought){{this.currentTime={start_seconds:.2f};this._sought=true;}}">'
                    f'<source src="{url}" type="audio/mpeg"></audio>'
                )
            except Exception:
                pass

        audio_html = f'<div>{"".join(players)}</div>' if players else "<p style='color:#888'>No audio found.</p>"
        updated_history = (history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return answer, audio_html, updated_history
    except Exception as e:
        logger.exception("Query failed")
        return f"Error: {e}", "", history or []


def run_preprocessing_pipeline(
    audio_file,
    date_str: str,
    time_str: str,
    show_name: str,
    location: str,
    known_speakers: str,
    progress=gr.Progress(),
):
    from ingestion.diarization import split_audio, create_manifest, load_embeddings_and_cluster
    from ingestion.transcriber import transcribe_segments
    from models.asr import WhisperPipeline, NemoDiarizer
    from utils.progress import Progress

    logs = []

    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    reporter = Progress(lambda pct, msg: progress(pct / 100, desc=msg))

    if audio_file is None:
        yield log("No audio file uploaded."), "", "", ""
        return

    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        yield log("OPENAI_API_KEY is not set in the environment."), "", "", ""
        return

    user_hints = [s.strip() for s in known_speakers.split(",") if s.strip()] if known_speakers else []

    work_dir = tempfile.mkdtemp(prefix="diarization_")
    try:
        reporter(5, "Saving audio to MinIO...")
        yield log("Step 1/9: Saving audio to MinIO..."), "", "", ""

        audio_path = audio_file.name if hasattr(audio_file, "name") else audio_file
        audio_filename = Path(audio_path).name
        stem = Path(audio_filename).stem

        minio_audio_key = None
        try:
            from storage.minio_client import MinIOClient
            from config.settings import Config
            from config.settings import DatabaseConfig
            cfg = Config.from_env()
            minio = MinIOClient(cfg)
            minio_audio_key = minio.upload_raw_audio(audio_path)
            yield log(f"Audio saved to MinIO: {minio_audio_key}"), "", "", ""
        except Exception as e:
            yield log(f"MinIO is not configured or upload failed: {e}\nPipeline stopped."), "", "", ""
            return

        whisper = WhisperPipeline(cfg.transcription.whisper_model)
        nemo = NemoDiarizer(cfg.diarization.nemo_config_url)

        db_config = DatabaseConfig.from_env()
        speaker_repo = SpeakerRepository(db_config)
        speaker_repo.setup()

        yield log("Step 2/9: Splitting audio into segments..."), "", "", ""
        num_segments = split_audio(audio_path, work_dir, cfg.diarization.segment_minutes, reporter)
        yield log(f"Split into {num_segments} segment(s)"), "", "", ""

        yield log("Step 3/9: Creating NeMo manifest..."), "", "", ""
        manifest_path = create_manifest(work_dir)
        yield log("Manifest created"), "", "", ""

        yield log("Step 4/9: Running NVIDIA NeMo diarization (may take several minutes)..."), "", "", ""
        reporter(28, "NeMo diarization running...")
        output_dir = nemo.run(work_dir, manifest_path)
        yield log("NeMo diarization complete"), "", "", ""

        try:
            minio.upload_nemo_outputs(stem, output_dir)
            yield log("NeMo outputs cached to MinIO"), "", "", ""
        except Exception as e:
            yield log(f"NeMo output caching failed (non-fatal): {e}"), "", "", ""

        yield log("Step 5/9: Loading embeddings & HDBSCAN clustering..."), "", "", ""
        speaker_remap, speaker_centroids = load_embeddings_and_cluster(output_dir, reporter)
        n_speakers = len(speaker_centroids)
        yield log(f"Found {n_speakers} global speakers via HDBSCAN"), "", "", ""

        yield log("Step 6/9: Matching speakers against database..."), "", "", ""
        reporter(62, "Matching against speaker DB...")
        speaker_results = resolve_speakers_from_db(speaker_centroids, speaker_repo, threshold=cfg.diarization.speaker_similarity_threshold)

        matched = sum(1 for v in speaker_results.values() if v["source"] == "db")
        unmatched_ids = [sid for sid, v in speaker_results.items() if v["source"] == "unknown"]

        db_log = f"DB matching: {matched}/{n_speakers} matched"
        for sid, info in speaker_results.items():
            if info["source"] == "db":
                db_log += f"\n   • {sid} → {info['name']} (sim={info['similarity']:.3f})"

        yield log(db_log), "", "", ""

        yield log("Step 7/9: Transcribing full audio with speaker labels..."), "", "", ""
        transcript = transcribe_segments(
            work_dir, output_dir, speaker_remap, num_segments,
            whisper=whisper,
            segment_minutes=cfg.diarization.segment_minutes,
            reporter=reporter,
        )
        yield log(f"Transcription done: {len(transcript)} turns"), "", "", ""

        llm_identities = {}
        if unmatched_ids:
            yield log(f"Step 8/9: LLM identifying {len(unmatched_ids)} unknown speaker(s)..."), "", "", ""
            reporter(85, "LLM speaker identification...")
            try:
                llm_identities = llm_identify_speakers(
                    transcript, unmatched_ids, user_hints, openai_api_key, model=cfg.model.llm_model
                )
                llm_log = "LLM identification results:"
                for sid, name in llm_identities.items():
                    llm_log += f"\n   • {sid} → {name or 'not identified'}"
                yield log(llm_log), "", "", ""
            except Exception as e:
                yield log(f"LLM identification error: {e}"), "", "", ""
        else:
            yield log("Step 8/9: All speakers matched from DB — LLM step skipped"), "", "", ""

        final_mapping = build_final_speaker_mapping(speaker_results, llm_identities)
        resolution_report = generate_resolution_report(speaker_results, llm_identities, final_mapping)

        for spk_id, centroid in speaker_centroids.items():
            if spk_id in llm_identities and llm_identities[spk_id]:
                no_turns = sum(1 for t in transcript if t["speaker"] == spk_id)
                speaker_repo.upsert_speaker(
                    name=llm_identities[spk_id],
                    embedding=centroid,
                    notes=f"Auto-added from {audio_filename}",
                    no_files=1,
                    no_turns=no_turns,
                )

        for spk_id, info in speaker_results.items():
            if info["source"] == "db":
                no_turns = sum(1 for t in transcript if t["speaker"] == spk_id)
                speaker_repo.upsert_speaker(
                    name=info["name"],
                    embedding=speaker_centroids[spk_id],
                    no_files=1,
                    no_turns=no_turns,
                )

        yield log("Step 9/9: Generating metadata & summary..."), "", "", ""
        reporter(92, "Generating metadata...")

        llm_summary = asyncio.run(
            generate_topics_and_summary_async(
                transcript, final_mapping, openai_api_key,
                model=cfg.model.llm_model,
            )
        )

        metadata = build_metadata_json(
            transcript, final_mapping,
            date=date_str, time_str=time_str,
            show_name=show_name, location=location,
            llm_summary=llm_summary,
        )

        transcript_txt = build_transcript_txt(transcript, final_mapping)

        out_dir = Path("outputs") / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata_json_str = json.dumps(metadata, ensure_ascii=False, indent=2)

        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            f.write(metadata_json_str)
        with open(out_dir / "transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_txt)

        try:
            minio.upload_ingestion_outputs(stem, transcript_txt, metadata_json_str)
            yield log("Outputs uploaded to MinIO"), metadata_json_str, transcript_txt, resolution_report
        except Exception as e:
            yield log(f"MinIO output upload failed: {e}"), metadata_json_str, transcript_txt, resolution_report

        shutil.rmtree(out_dir, ignore_errors=True)

        try:
            from storage.metadata_repository import MetadataRepository
            from config.settings import Config
            cfg = Config.from_env()
            repo = MetadataRepository(cfg.database)
            repo.setup()
            repo.save(
                doc_id=stem,
                file_path=minio_audio_key or f"audio/raw/{audio_filename}",
                metadata=metadata,
            )
            yield log("Metadata saved to database"), metadata_json_str, transcript_txt, resolution_report
        except Exception as e:
            yield log(f"Database save skipped (not configured): {e}"), metadata_json_str, transcript_txt, resolution_report

        yield (
            log("Speaker labeling and transcript extraction are done successfully."),
            metadata_json_str,
            transcript_txt,
            resolution_report,
        )

    except Exception as e:
        logger.exception("Pipeline failed")
        yield log(f"Pipeline error: {e}"), "", "", ""
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def resume_from_nemo(
    stem: str,
    date_str: str,
    time_str: str,
    show_name: str,
    location: str,
    known_speakers: str,
    progress=gr.Progress(),
):
    """
    Resume pipeline from cached NeMo outputs in MinIO.
    Skips Steps 1-4 (audio upload, split, manifest, NeMo diarization).
    """
    from ingestion.diarization import split_audio, load_embeddings_and_cluster
    from ingestion.transcriber import transcribe_segments
    from models.asr import WhisperPipeline
    from utils.progress import Progress

    logs = []

    def log(msg):
        logs.append(msg)
        return "\n".join(logs)

    reporter = Progress(lambda pct, msg: progress(pct / 100, desc=msg))

    if not stem or not stem.strip():
        yield log("Please enter a stem name."), "", "", ""
        return

    stem = stem.strip()
    user_hints = [s.strip() for s in known_speakers.split(",") if s.strip()] if known_speakers else []
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_api_key:
        yield log("OPENAI_API_KEY is not set in the environment."), "", "", ""
        return

    work_dir = tempfile.mkdtemp(prefix="resume_")
    try:
        from storage.minio_client import MinIOClient
        from config.settings import Config
        from config.settings import DatabaseConfig
        cfg = Config.from_env()
        minio = MinIOClient(cfg)
        db_config = DatabaseConfig.from_env()
        speaker_repo = SpeakerRepository(db_config)
        speaker_repo.setup()

        whisper = WhisperPipeline(cfg.transcription.whisper_model)

        reporter(10, "Downloading NeMo outputs...")
        yield log(f"Downloading cached NeMo outputs for '{stem}'..."), "", "", ""

        if not minio.nemo_outputs_exist(stem):
            yield log(f"No cached NeMo outputs found for stem '{stem}'. Run the full pipeline first."), "", "", ""
            return

        output_dir = os.path.join(work_dir, "oracle_vad")
        minio.download_nemo_outputs(stem, output_dir)
        yield log("NeMo outputs downloaded"), "", "", ""

        reporter(20, "Downloading audio...")
        yield log("Downloading raw audio for transcription..."), "", "", ""
        try:
            audio_path = minio.download_audio(stem, work_dir)
        except StopIteration:
            yield log(f"Could not download audio for stem '{stem}' — file not found in MinIO under transcripts/{stem}/."), "", "", ""
            return

        num_segments = split_audio(audio_path, work_dir, cfg.diarization.segment_minutes, reporter)
        yield log(f"Audio re-split into {num_segments} segment(s)"), "", "", ""

        yield log("Running HDBSCAN clustering..."), "", "", ""
        speaker_remap, speaker_centroids = load_embeddings_and_cluster(output_dir, reporter)
        n_speakers = len(speaker_centroids)
        yield log(f"Found {n_speakers} global speakers via HDBSCAN"), "", "", ""

        reporter(62, "Matching against speaker DB...")
        yield log("Matching speakers against database..."), "", "", ""
        speaker_results = resolve_speakers_from_db(speaker_centroids, speaker_repo, threshold=cfg.diarization.speaker_similarity_threshold)
        matched = sum(1 for v in speaker_results.values() if v["source"] == "db")
        unmatched_ids = [sid for sid, v in speaker_results.items() if v["source"] == "unknown"]
        yield log(f"DB matching: {matched}/{n_speakers} matched"), "", "", ""

        yield log("Transcribing full audio with speaker labels..."), "", "", ""
        transcript = transcribe_segments(
            work_dir, output_dir, speaker_remap, num_segments,
            whisper=whisper,
            segment_minutes=cfg.diarization.segment_minutes,
            reporter=reporter,
        )
        yield log(f"Transcription done: {len(transcript)} turns"), "", "", ""

        llm_identities = {}
        if unmatched_ids:
            yield log(f"LLM identifying {len(unmatched_ids)} unknown speaker(s)..."), "", "", ""
            reporter(82, "LLM speaker identification...")
            try:
                llm_identities = llm_identify_speakers(
                    transcript, unmatched_ids, user_hints, openai_api_key, model=cfg.model.llm_model
                )
                llm_log = "LLM identification results:"
                for sid, name in llm_identities.items():
                    llm_log += f"\n   • {sid} → {name or 'not identified'}"
                yield log(llm_log), "", "", ""
            except Exception as e:
                yield log(f"LLM identification error: {e}"), "", "", ""
        else:
            yield log("All speakers matched from DB — LLM step skipped"), "", "", ""

        final_mapping = build_final_speaker_mapping(speaker_results, llm_identities)
        resolution_report = generate_resolution_report(speaker_results, llm_identities, final_mapping)

        for spk_id, centroid in speaker_centroids.items():
            if spk_id in llm_identities and llm_identities[spk_id]:
                no_turns = sum(1 for t in transcript if t["speaker"] == spk_id)
                speaker_repo.upsert_speaker(
                    name=llm_identities[spk_id], embedding=centroid,
                    notes=f"Auto-added from {stem}", no_files=1, no_turns=no_turns,
                )
        for spk_id, info in speaker_results.items():
            if info["source"] == "db":
                no_turns = sum(1 for t in transcript if t["speaker"] == spk_id)
                speaker_repo.upsert_speaker(
                    name=info["name"], embedding=speaker_centroids[spk_id],
                    no_files=1, no_turns=no_turns,
                )

        yield log("Generating metadata & summary..."), "", "", ""
        reporter(90, "Generating metadata...")

        llm_summary = asyncio.run(
            generate_topics_and_summary_async(
                transcript, final_mapping, openai_api_key,
                model=cfg.model.llm_model,
            )
        )
        metadata = build_metadata_json(
            transcript, final_mapping,
            date=date_str, time_str=time_str,
            show_name=show_name, location=location,
            llm_summary=llm_summary,
        )
        transcript_txt = build_transcript_txt(transcript, final_mapping)
        metadata_json_str = json.dumps(metadata, ensure_ascii=False, indent=2)

        try:
            minio.upload_ingestion_outputs(stem, transcript_txt, metadata_json_str)
            yield log("Outputs uploaded to MinIO"), metadata_json_str, transcript_txt, resolution_report
        except Exception as e:
            yield log(f"MinIO output upload failed: {e}"), metadata_json_str, transcript_txt, resolution_report

        yield log("Done."), metadata_json_str, transcript_txt, resolution_report

    except Exception as e:
        logger.exception("Resume pipeline failed")
        yield log(f"Pipeline error: {e}"), "", "", ""
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
ANT_CSS = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    font-size: 14px !important;
    color: rgba(0,0,0,0.88) !important;
    background: #f5f5f5 !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
}

/* ── Header ── */
#ant-header {
    background: #fff;
    border-bottom: 1px solid #f0f0f0;
    padding: 0 24px;
    margin: 0 -24px 24px;
    display: flex;
    align-items: center;
    height: 64px;
    box-shadow: 0 1px 4px rgba(0,21,41,.08);
}

#ant-header .ant-logo {
    font-size: 18px;
    font-weight: 600;
    color: rgba(0,0,0,0.88);
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 10px;
}

#ant-header .ant-logo::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 28px;
    background: #1677ff;
    border-radius: 6px;
}

/* ── Top-level tabs — Ant "line" style ── */
.tabs-container > .tab-nav,
div.tabs > div.tab-nav {
    border-bottom: 1px solid #f0f0f0 !important;
    background: #fff !important;
    padding: 0 !important;
    margin-bottom: 16px !important;
    border-radius: 0 !important;
    gap: 0 !important;
}

div.tabs > div.tab-nav > button,
.tabs-container > .tab-nav > button {
    font-size: 14px !important;
    font-weight: 400 !important;
    color: rgba(0,0,0,0.65) !important;
    padding: 12px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    background: transparent !important;
    margin: 0 !important;
    transition: color .2s, border-color .2s !important;
}

div.tabs > div.tab-nav > button:hover {
    color: #1677ff !important;
}

div.tabs > div.tab-nav > button.selected {
    color: #1677ff !important;
    font-weight: 500 !important;
    border-bottom: 2px solid #1677ff !important;
    background: transparent !important;
}

/* ── Cards (sections) ── */
.ant-card {
    background: #fff;
    border: 1px solid #f0f0f0;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,.03);
}

.ant-card-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(0,0,0,0.88);
    margin: 0 0 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #f0f0f0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.ant-card-title .step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #1677ff;
    color: #fff;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
}

.ant-section-desc {
    font-size: 13px;
    color: rgba(0,0,0,0.45);
    margin: -8px 0 14px;
    line-height: 1.6;
}

/* ── Form labels ── */
label, .block > label span {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgba(0,0,0,0.88) !important;
    margin-bottom: 4px !important;
}

/* ── Inputs ── */
input[type=text], textarea, .gr-text-input input, .gr-textbox textarea {
    border: 1px solid #d9d9d9 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-family: inherit !important;
    padding: 7px 11px !important;
    transition: border-color .2s, box-shadow .2s !important;
    background: #fff !important;
    color: rgba(0,0,0,0.88) !important;
}

input[type=text]:focus, textarea:focus,
.gr-text-input input:focus, .gr-textbox textarea:focus {
    border-color: #1677ff !important;
    box-shadow: 0 0 0 2px rgba(22,119,255,.1) !important;
    outline: none !important;
}

input::placeholder, textarea::placeholder {
    color: rgba(0,0,0,0.25) !important;
}

/* ── Buttons ── */
button.primary, .gr-button-primary {
    background: #1677ff !important;
    border: 1px solid #1677ff !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    height: 36px !important;
    padding: 0 20px !important;
    cursor: pointer !important;
    transition: background .2s, border-color .2s, box-shadow .2s !important;
    box-shadow: 0 2px 0 rgba(5,145,255,.1) !important;
}

button.primary:hover, .gr-button-primary:hover {
    background: #4096ff !important;
    border-color: #4096ff !important;
}

button.primary:active, .gr-button-primary:active {
    background: #0958d9 !important;
    border-color: #0958d9 !important;
}

button.secondary, .gr-button-secondary {
    background: #fff !important;
    border: 1px solid #d9d9d9 !important;
    border-radius: 6px !important;
    color: rgba(0,0,0,0.88) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    height: 36px !important;
    padding: 0 16px !important;
    cursor: pointer !important;
    transition: border-color .2s, color .2s !important;
    box-shadow: 0 2px 0 rgba(0,0,0,.02) !important;
}

button.secondary:hover, .gr-button-secondary:hover {
    color: #1677ff !important;
    border-color: #1677ff !important;
}

/* ── File upload ── */
.upload-container, .gr-file {
    border: 1px dashed #d9d9d9 !important;
    border-radius: 8px !important;
    background: #fafafa !important;
    transition: border-color .2s !important;
}

.upload-container:hover, .gr-file:hover {
    border-color: #1677ff !important;
}

/* ── Log / code boxes ── */
#log-box textarea {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
    font-size: 12px !important;
    background: #141414 !important;
    color: #d4d4d4 !important;
    border: 1px solid #303030 !important;
    border-radius: 6px !important;
    padding: 12px !important;
    line-height: 1.6 !important;
}

/* ── Code component ── */
.cm-editor, .code-editor {
    border-radius: 6px !important;
    border: 1px solid #f0f0f0 !important;
    font-size: 12px !important;
}

/* ── Chatbot ── */
.chatbot {
    border: 1px solid #f0f0f0 !important;
    border-radius: 8px !important;
    background: #fff !important;
}

.chatbot .message.user {
    background: #1677ff !important;
    color: #fff !important;
    border-radius: 8px 8px 2px 8px !important;
    font-size: 14px !important;
}

.chatbot .message.bot {
    background: #f5f5f5 !important;
    color: rgba(0,0,0,0.88) !important;
    border-radius: 8px 8px 8px 2px !important;
    font-size: 14px !important;
}

/* ── Chat input row ── */
#chat-row {
    margin-top: 12px;
    border: 1px solid #d9d9d9;
    border-radius: 8px;
    background: #fff;
    display: flex;
    align-items: center;
    padding: 6px 6px 6px 12px;
    transition: border-color .2s, box-shadow .2s;
}

#chat-row:focus-within {
    border-color: #1677ff !important;
    box-shadow: 0 0 0 2px rgba(22,119,255,.1) !important;
}

#chat-row textarea {
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
    resize: none !important;
    background: transparent !important;
    font-size: 14px !important;
}

#chat-row button {
    height: 32px !important;
    padding: 0 16px !important;
    border-radius: 6px !important;
    flex-shrink: 0 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #f0f0f0 !important;
    margin: 16px 0 !important;
}

/* ── Output tabs ── */
.output-tabs > div.tab-nav {
    background: #fafafa !important;
    border: 1px solid #f0f0f0 !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0 8px !important;
}

.output-tabs > div.tab-nav > button {
    font-size: 13px !important;
    padding: 8px 14px !important;
}

/* ── Markdown content ── */
.prose, .md p { font-size: 14px !important; line-height: 1.7 !important; }

/* ── Audio players ── */
audio {
    width: 100%;
    border-radius: 6px;
    height: 36px;
    margin-bottom: 8px;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f5f5f5; }
::-webkit-scrollbar-thumb { background: #d9d9d9; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #bfbfbf; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Speech Recognition Pipeline",
        css=ANT_CSS,
    ) as demo:

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML("""
        <div id="ant-header">
          <div class="ant-logo">Speech Recognition and Understanding Pipeline</div>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Diarization and Autovoice Annotation ───────────────
            with gr.Tab("Diarization and Autovoice Annotation"):

                with gr.Tabs():

                    # ── Sub-tab A: Full pipeline ──────────────────────────
                    with gr.Tab("Full Pipeline"):

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title"><span class="step-badge">1</span> Upload Audio</div>""")
                        audio_input = gr.File(
                            label="Audio File (.wav / .mp3 / .mp4 / .m4a)",
                            file_types=[".wav", ".mp3", ".mp4", ".m4a"],
                        )
                        gr.HTML("</div>")

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title"><span class="step-badge">2</span> Recording Context <span style="font-weight:400;color:rgba(0,0,0,.45);font-size:12px;margin-left:6px;">optional</span></div>""")
                        with gr.Row():
                            date_input    = gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="e.g. 2025-04-08")
                            time_input    = gr.Textbox(label="Time (HH:MM)", placeholder="e.g. 20:00")
                            show_input    = gr.Textbox(label="Show Name", placeholder="e.g. Нешто конкретно")
                            location_input = gr.Textbox(label="Location", placeholder="e.g. Хотел Лимак")
                        gr.HTML("</div>")

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title"><span class="step-badge">3</span> Known Speakers <span style="font-weight:400;color:rgba(0,0,0,.45);font-size:12px;margin-left:6px;">optional — helps LLM identify unmatched voices</span></div>""")
                        known_speakers_input = gr.Textbox(
                            label="Comma-separated speaker names",
                            placeholder="e.g. Елена Ристеска, Ване Марковски, Даниел",
                            lines=1,
                        )
                        gr.HTML("</div>")

                        run_btn = gr.Button("▶  Start Pipeline", variant="primary", size="lg")

                    # ── Sub-tab B: Resume from NeMo ───────────────────────
                    with gr.Tab("Resume from NeMo"):

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title">Resume from Cached NeMo Outputs</div>
                          <p class="ant-section-desc">Re-run clustering, transcription, and speaker identification from cached NeMo diarization outputs. Skips voice activity detection and NeMo diarization — use when re-processing a previously diarized recording.</p>""")
                        stem_input = gr.Textbox(
                            label="Stem (filename without extension)",
                            placeholder="e.g. sobranie_68",
                        )
                        gr.HTML("</div>")

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title">Recording Context <span style="font-weight:400;color:rgba(0,0,0,.45);font-size:12px;margin-left:6px;">optional</span></div>""")
                        with gr.Row():
                            r_date_input     = gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="e.g. 2025-04-08")
                            r_time_input     = gr.Textbox(label="Time (HH:MM)", placeholder="e.g. 20:00")
                            r_show_input     = gr.Textbox(label="Show Name", placeholder="e.g. Нешто конкретно")
                            r_location_input = gr.Textbox(label="Location", placeholder="e.g. Хотел Лимак")
                        gr.HTML("</div>")

                        gr.HTML("""<div class="ant-card">
                          <div class="ant-card-title">Known Speakers <span style="font-weight:400;color:rgba(0,0,0,.45);font-size:12px;margin-left:6px;">optional</span></div>""")
                        r_known_speakers_input = gr.Textbox(
                            label="Comma-separated speaker names",
                            placeholder="e.g. Елена Ристеска, Ване Марковски",
                            lines=1,
                        )
                        gr.HTML("</div>")

                        resume_btn = gr.Button("▶  Resume Pipeline", variant="primary", size="lg")

                # ── Shared outputs ────────────────────────────────────────
                gr.HTML("<hr>")
                log_output = gr.Textbox(
                    label="Pipeline Log",
                    lines=14,
                    interactive=False,
                    elem_id="log-box",
                )

                with gr.Tabs(elem_classes=["output-tabs"]):
                    with gr.Tab("Metadata JSON"):
                        metadata_output = gr.Code(
                            label="metadata.json",
                            language="json",
                            interactive=False,
                        )
                    with gr.Tab("Transcript"):
                        transcript_output = gr.Textbox(
                            label="Labeled Transcript",
                            lines=25,
                            interactive=False,
                        )
                    with gr.Tab("Speaker Resolution Report"):
                        resolution_output = gr.Markdown()

                run_btn.click(
                    fn=run_preprocessing_pipeline,
                    inputs=[audio_input, date_input, time_input, show_input, location_input, known_speakers_input],
                    outputs=[log_output, metadata_output, transcript_output, resolution_output],
                )
                resume_btn.click(
                    fn=resume_from_nemo,
                    inputs=[stem_input, r_date_input, r_time_input, r_show_input, r_location_input, r_known_speakers_input],
                    outputs=[log_output, metadata_output, transcript_output, resolution_output],
                )

            # ── Tab 2: Retrieval-Augmented Generation ────────────────────
            with gr.Tab("Retrieval-Augmented Generation"):
                if not RAG_AVAILABLE:
                    gr.HTML("""<div class="ant-card">
                      <p style="color:rgba(0,0,0,.45);margin:0;">RAG modules not installed. Run the full setup to enable this tab.</p>
                    </div>""")
                else:
                    chatbot = gr.Chatbot(
                        label="",
                        height=500,
                        show_label=False,
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=rag"),
                    )
                    audio_output = gr.HTML(label="Source Audio Segments")
                    history_state = gr.State([])

                    with gr.Row(elem_id="chat-row"):
                        question_input = gr.Textbox(
                            placeholder="Ask about speakers, topics, or events in the transcripts...",
                            lines=1,
                            show_label=False,
                            scale=9,
                            container=False,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=72)

                    async def chat(question, chat_history, history):
                        answer, audio_html, updated_history = await run_query(question, history)
                        chat_history = chat_history or []
                        chat_history.append((question, answer))
                        return "", chat_history, audio_html, updated_history

                    submit_btn.click(
                        fn=chat,
                        inputs=[question_input, chatbot, history_state],
                        outputs=[question_input, chatbot, audio_output, history_state],
                    )
                    question_input.submit(
                        fn=chat,
                        inputs=[question_input, chatbot, history_state],
                        outputs=[question_input, chatbot, audio_output, history_state],
                    )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="gray",
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    )