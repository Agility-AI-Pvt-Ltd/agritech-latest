import os
from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Keys
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")

    # Database Settings
    base_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    db_folder: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage")
    db_path: str = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage")
    qdrant_path: str = Field(
        os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "db_storage", "qdrant"),
        alias="QDRANT_PATH"
    )
    qdrant_collection_name: str = Field("agritech_knowledge", alias="QDRANT_COLLECTION_NAME")
    qdrant_log_enabled: bool = Field(True, alias="QDRANT_LOG_ENABLED")
    qdrant_log_dir: str = Field(
        os.path.join(base_dir, "logs", "qdrant"),
        alias="QDRANT_LOG_DIR"
    )

    # PostgreSQL (Async SQLAlchemy) Settings
    database_url: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/agritech",
        alias="DATABASE_URL"
    )
    database_echo: bool = Field(False, alias="DATABASE_ECHO")
    database_pool_size: int = Field(20, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(40, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, alias="DATABASE_POOL_TIMEOUT")
    database_pool_recycle: int = Field(1800, alias="DATABASE_POOL_RECYCLE")
    database_pool_pre_ping: bool = Field(True, alias="DATABASE_POOL_PRE_PING")
    database_auto_create_tables: bool = Field(True, alias="DATABASE_AUTO_CREATE_TABLES")

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
    llm_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL_NAME")
    embedding_model: str = Field("models/text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")
    sentence_transformer_model: str = Field("all-MiniLM-L6-v2", alias="SENTENCE_TRANSFORMER_MODEL")
    llm_temperature: float = 0.2

    # Retrieval Mode Settings
    retrieval_mode: str = Field("rag", alias="RETRIEVAL_MODE")  # rag | pageindex
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Initialize global settings
settings = Settings()
