import logging
from typing import Any, AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from services.weather import OpenMeteoWeatherProvider, WeatherProvider
from services.vectorstore import QdrantVectorStore, VectorStoreProvider
from services.pageindex import PageIndexProvider
from services.advisory import LangGraphAdvisoryGenerator
from services.advisory_log import AdvisoryLogService
from services.conversation import ConversationService
from services.crop import CropService
from services.speech import SarvamSpeechService
from core.database import db_manager

logger = logging.getLogger(__name__)

# Dependency Injection Container (Singleton Instances)
# We instantiate them here to provide to FastAPI depends
class DependableContainer:
    def __init__(self):
        self.weather_provider: WeatherProvider | None = None
        self.vector_store: VectorStoreProvider | None = None
        self.pageindex_provider: PageIndexProvider | None = None
        self.advisory_generator: LangGraphAdvisoryGenerator | None = None
        self.advisory_log_service: AdvisoryLogService | None = None
        self.conversation_service: ConversationService | None = None
        self.crop_service: CropService | None = None
        self.speech_service: SarvamSpeechService | None = None
        self.chat_llm: Any | None = None
        self.chat_safety_llm: Any | None = None
        self.chat_qdrant_client: Any | None = None

    def get_weather_provider(self) -> WeatherProvider:
        if self.weather_provider is None:
            self.weather_provider = OpenMeteoWeatherProvider()
        return self.weather_provider

    def get_vector_store(self) -> VectorStoreProvider:
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore()
        return self.vector_store

    def get_pageindex_provider(self) -> PageIndexProvider:
        if self.pageindex_provider is None:
            self.pageindex_provider = PageIndexProvider()
        return self.pageindex_provider

    def get_advisory_generator(self) -> LangGraphAdvisoryGenerator:
        if self.advisory_generator is None:
            self.advisory_generator = LangGraphAdvisoryGenerator()
        return self.advisory_generator

    def get_advisory_log_service(self) -> AdvisoryLogService:
        if self.advisory_log_service is None:
            self.advisory_log_service = AdvisoryLogService()
        return self.advisory_log_service

    def get_conversation_service(self) -> ConversationService:
        if self.conversation_service is None:
            self.conversation_service = ConversationService()
        return self.conversation_service

    def get_crop_service(self) -> CropService:
        if self.crop_service is None:
            self.crop_service = CropService()
        return self.crop_service

    def get_speech_service(self) -> SarvamSpeechService:
        if self.speech_service is None:
            self.speech_service = SarvamSpeechService()
        return self.speech_service

    def get_chat_llm(self) -> Any:
        if self.chat_llm is None:
            from pipeline.llm_factory import get_llm
            self.chat_llm = get_llm()
        return self.chat_llm

    def get_chat_safety_llm(self) -> Any | None:
        if self.chat_safety_llm is None:
            from pipeline.llm_factory import get_safety_llm
            self.chat_safety_llm = get_safety_llm()
        return self.chat_safety_llm

    def get_chat_qdrant_client(self) -> Any | None:
        if self.chat_qdrant_client is None:
            from pipeline.llm_factory import get_qdrant_client
            self.chat_qdrant_client = get_qdrant_client()
        return self.chat_qdrant_client

container = DependableContainer()

def get_weather_provider() -> WeatherProvider:
    return container.get_weather_provider()

def get_vector_store() -> VectorStoreProvider:
    return container.get_vector_store()

def get_pageindex_provider() -> PageIndexProvider:
    return container.get_pageindex_provider()

def get_advisory_generator() -> LangGraphAdvisoryGenerator:
    return container.get_advisory_generator()

def get_advisory_log_service() -> AdvisoryLogService:
    return container.get_advisory_log_service()

def get_conversation_service() -> ConversationService:
    return container.get_conversation_service()

def get_crop_service() -> CropService:
    return container.get_crop_service()

def get_speech_service() -> SarvamSpeechService:
    return container.get_speech_service()

def get_chat_llm() -> Any:
    return container.get_chat_llm()

def get_chat_safety_llm() -> Any | None:
    return container.get_chat_safety_llm()

def get_chat_qdrant_client() -> Any | None:
    return container.get_chat_qdrant_client()

def init_chat_resources_on_startup() -> tuple[Any, Any | None, Any | None, Any | None]:
    """Initialize shared chat resources during app startup."""
    llm = get_chat_llm()
    safety_llm = get_chat_safety_llm()
    qdrant = get_chat_qdrant_client()
    embedding_model = None

    try:
        from pipeline.llm_factory import get_embedding_model
        embedding_model = get_embedding_model()
    except Exception as exc:
        logger.warning("[!] Embedding model failed to initialize at startup: %s", exc)

    if safety_llm is None:
        logger.warning("[!] Chat safety LLM failed to initialize at startup")
    else:
        logger.info("[✓] Chat safety LLM initialized at startup")

    if qdrant is None:
        logger.warning("[!] Chat Qdrant client failed to initialize at startup")
    else:
        logger.info("[✓] Chat Qdrant client initialized at startup")

    if embedding_model is None:
        logger.warning("[!] Embedding model failed to initialize at startup")
    else:
        logger.info("[✓] Embedding model initialized at startup")

    return llm, safety_llm, qdrant, embedding_model

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in db_manager.get_session():
        yield session
