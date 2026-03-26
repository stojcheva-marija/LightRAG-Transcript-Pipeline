from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


def require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Required environment variable '{key}' is not set properly.")
    return value


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        return cls(
            host=require_env("POSTGRES_HOST"),
            port=int(require_env("POSTGRES_PORT")),
            database=require_env("POSTGRES_DATABASE"),
            user=require_env("POSTGRES_USER"),
            password=require_env("POSTGRES_PASSWORD"),
        )

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str

    @classmethod
    def from_env(cls) -> MinIOConfig:
        return cls(
            endpoint=require_env("MINIO_ENDPOINT"),
            access_key=require_env("MINIO_ACCESS_KEY"),
            secret_key=require_env("MINIO_SECRET_KEY"),
            bucket=require_env("MINIO_BUCKET"),
        )


@dataclass(frozen=True)
class ModelConfig:
    llm_model: str
    embedding_model: str
    embedding_dim: int
    max_token_size: int

    @classmethod
    def from_env(cls) -> ModelConfig:
        return cls(
            llm_model=require_env("LLM_MODEL"),
            embedding_model=require_env("EMBEDDING_MODEL"),
            embedding_dim=int(require_env("EMBEDDING_DIM")),
            max_token_size=int(require_env("MAX_TOKEN_SIZE")),
        )


@dataclass(frozen=True)
class DiarizationConfig:
    segment_minutes: int
    nemo_config_url: str
    speaker_similarity_threshold: float

    @classmethod
    def from_env(cls) -> DiarizationConfig:
        return cls(
            segment_minutes=int(require_env("DIARIZATION_SEGMENT_MINUTES")),
            nemo_config_url=require_env("NEMO_CONFIG_URL"),
            speaker_similarity_threshold=float(require_env("SPEAKER_SIMILARITY_THRESHOLD")),
        )


@dataclass(frozen=True)
class TranscriptionConfig:
    whisper_model: str

    @classmethod
    def from_env(cls) -> TranscriptionConfig:
        return cls(
            whisper_model=require_env("WHISPER_MODEL"),
        )


@dataclass(frozen=True)
class ChunkerConfig:
    block_window_size: int
    contextual_window_size: int

    @classmethod
    def from_env(cls) -> ChunkerConfig:
        return cls(
            block_window_size=int(require_env("BLOCK_WINDOW_SIZE")),
            contextual_window_size=int(require_env("CONTEXTUAL_WINDOW_SIZE")),
        )


@dataclass(frozen=True)
class StorageConfig:
    kv_storage: str
    vector_storage: str
    graph_storage: str
    doc_status_storage: str

    @classmethod
    def from_env(cls) -> StorageConfig:
        return cls(
            kv_storage=require_env("KV_STORAGE"),
            vector_storage=require_env("VECTOR_STORAGE"),
            graph_storage=require_env("GRAPH_STORAGE"),
            doc_status_storage=require_env("DOC_STATUS_STORAGE"),
        )


@dataclass(frozen=True)
class RerankerConfig:
    model: str
    top_n: int

    @classmethod
    def from_env(cls) -> RerankerConfig:
        return cls(
            model=require_env("RERANKER_MODEL"),
            top_n=int(require_env("RERANKER_TOP_N")),
        )


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    working_dir: str
    workspace: str
    vector_namespace: str
    transcript_directory: str
    database: DatabaseConfig
    minio: MinIOConfig
    model: ModelConfig
    diarization: DiarizationConfig
    transcription: TranscriptionConfig
    chunker: ChunkerConfig
    storage: StorageConfig
    reranker: RerankerConfig

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()
        return cls(
            openai_api_key=require_env("OPENAI_API_KEY"),
            working_dir=require_env("WORKING_DIR"),
            workspace=require_env("WORKSPACE"),
            vector_namespace=require_env("VECTOR_NAMESPACE"),
            transcript_directory=require_env("TRANSCRIPT_DIR"),
            database=DatabaseConfig.from_env(),
            minio=MinIOConfig.from_env(),
            model=ModelConfig.from_env(),
            diarization=DiarizationConfig.from_env(),
            transcription=TranscriptionConfig.from_env(),
            chunker=ChunkerConfig.from_env(),
            storage=StorageConfig.from_env(),
            reranker=RerankerConfig.from_env(),
        )
