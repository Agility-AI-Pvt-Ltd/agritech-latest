from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from services.weather import OpenMeteoWeatherProvider, WeatherProvider
from services.vectorstore import QdrantVectorStore, VectorStoreProvider
from services.pageindex import PageIndexProvider
from services.advisory import LangGraphAdvisoryGenerator
from services.advisory_log import AdvisoryLogService
from services.conversation import ConversationService
from services.crop import CropService
from core.database import db_manager

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

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in db_manager.get_session():
        yield session
