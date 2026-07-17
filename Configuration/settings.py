"""Load and expose application settings from YAML + environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")

SETTINGS_FILE = PROJECT_ROOT / "Configuration" / "settings.yaml"


def _resolve_path(value: str) -> Path:
    """Convert a config path to an absolute project-root path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass
class ProjectSettings:
    name: str
    version: str


@dataclass
class PathSettings:
    documents_dir: Path
    index_dir: Path
    faiss_index_file: Path
    chunks_metadata_file: Path


@dataclass
class ChunkingSettings:
    max_words: int
    min_words: int
    overlap_sentences: int


@dataclass
class RetrievalSettings:
    top_k: int
    max_context_chunks: int
    word_budget: int
    min_score_ratio: float
    min_absolute_score: float
    max_chunks_per_document: int


@dataclass
class EmbeddingSettings:
    model_name: str
    normalize: bool
    batch_size: int


@dataclass
class OpenAISettings:
    api_key_env: str
    model: str
    base_url: str | None

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)
    

@dataclass
class GeminiSettings:
    api_key_env: str
    model: str

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


@dataclass
class OllamaSettings:
    host: str
    model: str


@dataclass
class LLMSettings:
    provider: str
    temperature: float
    max_tokens: int
    openai: OpenAISettings
    ollama: OllamaSettings
    gemini: GeminiSettings


@dataclass
class ServerSettings:
    host: str
    port: int
    cors_origins: list[str]


@dataclass
class Settings:
    project: ProjectSettings
    paths: PathSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    embedding: EmbeddingSettings
    llm: LLMSettings
    server: ServerSettings


def load_settings(config_path: Path | None = None) -> Settings:
    """Read YAML config and build typed settings objects."""
    config_path = config_path or SETTINGS_FILE
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    paths_raw = raw["paths"]
    llm_raw = raw["llm"]

    return Settings(
        project=ProjectSettings(**raw["project"]),
        paths=PathSettings(
            documents_dir=_resolve_path(paths_raw["documents_dir"]),
            index_dir=_resolve_path(paths_raw["index_dir"]),
            faiss_index_file=_resolve_path(paths_raw["faiss_index_file"]),
            chunks_metadata_file=_resolve_path(paths_raw["chunks_metadata_file"]),
        ),
        chunking=ChunkingSettings(**raw["chunking"]),
        retrieval=RetrievalSettings(**raw["retrieval"]),
        embedding=EmbeddingSettings(**raw["embedding"]),
        llm=LLMSettings(
            provider=llm_raw["provider"],
            temperature=llm_raw["temperature"],
            max_tokens=llm_raw["max_tokens"],
            openai=OpenAISettings(**llm_raw["openai"]),
            ollama=OllamaSettings(**llm_raw["ollama"]),
            gemini=GeminiSettings(**llm_raw["gemini"]),
        ),
        server=ServerSettings(**raw["server"]),
    )


settings = load_settings()
