import os
from typing import Dict, List
from urllib.parse import quote
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    nvidia_api_key: str | None = Field(None, alias="NVIDIA_API_KEY")
    sarvam_api_key: str | None = Field(None, alias="SARVAM_API_KEY")

    # Database Settings
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_folder: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage")
    db_path: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage")
    qdrant_path: str = Field(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage"),
        alias="QDRANT_PATH"
    )
    qdrant_url: str | None = Field(None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field("agritech_knowledge", alias="QDRANT_COLLECTION_NAME")
    qdrant_collection_names: str = Field(
        "spring_corn_fertilizers_db,maize_production_manual_db,spring_corn_pest_and_diseases_db,spring_corn_pop_db,farmerbook_db",
        alias="QDRANT_COLLECTION_NAMES",
    )
    maize_faq_collection_name: str = Field("maize_faq_db", alias="MAIZE_FAQ_COLLECTION_NAME")
    maize_faq_tree_path: str = Field(
        os.path.join(base_dir, "data", "maize_knowledge_tree.json"),
        alias="MAIZE_FAQ_TREE_PATH",
    )
    qdrant_log_enabled: bool = Field(True, alias="QDRANT_LOG_ENABLED")
    qdrant_log_dir: str = Field(
        os.path.join(base_dir, "logs", "qdrant"),
        alias="QDRANT_LOG_DIR"
    )

    # PostgreSQL (Async SQLAlchemy) Settings
    postgres_user: str = Field("postgres", alias="POSTGRES_USER")
    postgres_password: str = Field("postgres", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field("agritech", alias="POSTGRES_DB")
    postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    database_url: str | None = Field(None, alias="DATABASE_URL")
    database_echo: bool = Field(False, alias="DATABASE_ECHO")
    database_pool_size: int = Field(20, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(40, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, alias="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(1800, alias="DATABASE_POOL_RECYCLE")
    database_pool_pre_ping: bool = Field(True, alias="DATABASE_POOL_PRE_PING")
    database_auto_create_tables: bool = Field(True, alias="DATABASE_AUTO_CREATE_TABLES")
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # Default Coordinates (UP)
    default_latitude: float = 26.8
    default_longitude: float = 80.9

    # Predefined Questions
    predefined_questions: Dict[str, str] = {
        "1": "What farm operations should I do today / tomorrow for my crop?",
        "2": "Is there any weather risk for my crop in the next 7 days?",
        "3": "When to apply fertilizer or pesticide?",
        "4": "My crop is showing disease symptoms – what should I do?",
        "5": "Show common diseases for my crop at this stage",
        "6": "How can I protect my Maize crop from heat?",
        "7": "What best practices should I follow at my crop's current stage?"
    }

    # LLM Settings
    llm_provider: str = Field("openai", alias="LLM_PROVIDER")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_LARGE_MODEL")
    embedding_provider: str = Field("openai", alias="EMBEDDING_PROVIDER")
    openai_embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    embedding_model: str = Field("models/text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")
    sentence_transformer_model: str = Field("all-MiniLM-L6-v2", alias="SENTENCE_TRANSFORMER_MODEL")
    llm_temperature: float = 0.2
    chat_safety_enabled: bool = Field(True, alias="CHAT_SAFETY_ENABLED")
    chat_safety_fail_closed: bool = Field(True, alias="CHAT_SAFETY_FAIL_CLOSED")
    chat_security_blocking_enabled: bool = Field(True, alias="CHAT_SECURITY_BLOCKING_ENABLED")
    chat_low_info_blocking_enabled: bool = Field(True, alias="CHAT_LOW_INFO_BLOCKING_ENABLED")
    chat_safety_model_classification_enabled: bool = Field(True, alias="CHAT_SAFETY_MODEL_CLASSIFICATION_ENABLED")
    safety_llm_provider: str = Field("openai", alias="SAFETY_LLM_PROVIDER")
    safety_llm_model: str = Field("gpt-4o-mini", alias="SAFETY_LLM_MODEL")
    safety_llm_temperature: float = Field(0.0, alias="SAFETY_LLM_TEMPERATURE")

    # Retrieval Mode Settings
    retrieval_mode: str = Field("hybrid", alias="RETRIEVAL_MODE")  # rag | pageindex | hybrid
    hybrid_top_k: int = Field(8, alias="HYBRID_TOP_K")
    bm25_top_k: int = Field(6, alias="BM25_TOP_K")
    bm25_markdown_dir: str = Field(
        os.path.join(base_dir, "data", "markdowns"),
        alias="BM25_MARKDOWN_DIR",
    )
    bm25_chunk_words: int = Field(260, alias="BM25_CHUNK_WORDS")
    pageindex_tree_path: str = Field(
        os.path.join(base_dir, "PageIndex", "results", "b_structure.json"),
        alias="PAGEINDEX_TREE_PATH"
    )
    pageindex_pdf_path: str = Field(
        os.path.join(base_dir, "PageIndex", "DATA", "b.pdf"),
        alias="PAGEINDEX_PDF_PATH"
    )
    pageindex_max_nodes: int = Field(3, alias="PAGEINDEX_MAX_NODES")
    pageindex_log_enabled: bool = Field(True, alias="PAGEINDEX_LOG_ENABLED")
    pageindex_log_dir: str = Field(
        os.path.join(base_dir, "logs", "pageindex"),
        alias="PAGEINDEX_LOG_DIR"
    )

    # Weather API Settings
    weather_api_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_forecast_hours: int = 72
    weather_timeout: int = 10

    # Sarvam Speech Settings
    sarvam_base_url: str = Field("https://api.sarvam.ai", alias="SARVAM_BASE_URL")
    sarvam_stt_model: str = Field("saaras:v3", alias="SARVAM_STT_MODEL")
    sarvam_stt_language_code: str = Field("unknown", alias="SARVAM_STT_LANGUAGE_CODE")
    sarvam_stt_mode: str = Field("transcribe", alias="SARVAM_STT_MODE")
    sarvam_tts_model: str = Field("bulbul:v3", alias="SARVAM_TTS_MODEL")
    sarvam_tts_language_code: str = Field("hi-IN", alias="SARVAM_TTS_LANGUAGE_CODE")
    sarvam_tts_speaker: str = Field("shubh", alias="SARVAM_TTS_SPEAKER")
    sarvam_tts_sample_rate: int = Field(24000, alias="SARVAM_TTS_SAMPLE_RATE")
    sarvam_tts_audio_format: str = Field("wav", alias="SARVAM_TTS_AUDIO_FORMAT")
    sarvam_tts_pace: float = Field(1.0, alias="SARVAM_TTS_PACE")
    sarvam_tts_temperature: float = Field(0.6, alias="SARVAM_TTS_TEMPERATURE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @staticmethod
    def _to_async_database_url(database_url: str) -> str:
        """Normalize PostgreSQL URLs for SQLAlchemy asyncpg usage."""
        if database_url.startswith("postgres://"):
            return "postgresql+asyncpg://" + database_url[len("postgres://"):]
        if database_url.startswith("postgresql://") and "+asyncpg" not in database_url:
            return "postgresql+asyncpg://" + database_url[len("postgresql://"):]
        return database_url

    @staticmethod
    def _to_sync_database_url(database_url: str) -> str:
        """Normalize PostgreSQL URLs for psycopg2 usage."""
        return (
            database_url
            .replace("postgresql+asyncpg://", "postgresql://")
            .replace("postgresql+psycopg2://", "postgresql://")
        )

    @property
    def resolved_database_url(self) -> str:
        """Return DATABASE_URL or build it from POSTGRES_* settings."""
        if self.database_url:
            return self.database_url

        user = quote(self.postgres_user, safe="")
        password = quote(self.postgres_password, safe="")
        db = quote(self.postgres_db, safe="")
        return f"postgresql://{user}:{password}@{self.postgres_host}:{self.postgres_port}/{db}"

    @property
    def async_database_url(self) -> str:
        """Database URL formatted for SQLAlchemy async engine creation."""
        return self._to_async_database_url(self.resolved_database_url)

    @property
    def sync_database_url(self) -> str:
        """Database URL formatted for psycopg2 connections."""
        return self._to_sync_database_url(self.resolved_database_url)

    @property
    def qdrant_client_kwargs(self) -> Dict[str, str]:
        """Return QdrantClient kwargs for server or local filesystem mode."""
        if self.qdrant_url:
            kwargs = {"url": self.qdrant_url}
            if self.qdrant_api_key:
                kwargs["api_key"] = self.qdrant_api_key
            return kwargs

        return {"path": self.qdrant_path}

    @property
    def qdrant_location(self) -> str:
        """Human-readable Qdrant location for logs."""
        return self.qdrant_url or self.qdrant_path

    @property
    def resolved_qdrant_collections(self) -> List[str]:
        """Return de-duplicated collection names to query in Qdrant."""
        names: List[str] = []

        if self.qdrant_collection_name.strip():
            names.append(self.qdrant_collection_name.strip())

        for name in self.qdrant_collection_names.split(","):
            cleaned = name.strip()
            if cleaned:
                names.append(cleaned)

        # Preserve order while removing duplicates
        return list(dict.fromkeys(names))

# Initialize global settings
settings = Settings()
