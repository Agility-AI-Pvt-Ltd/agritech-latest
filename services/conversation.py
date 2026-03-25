from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from repositories.conversation_repository import ConversationRepository


class ConversationService:
    def __init__(self, repository: Optional[ConversationRepository] = None):
        self.repository = repository or ConversationRepository()

    async def save_incoming_query(
        self,
        session: AsyncSession,
        user_id: str,
        conversation_id: str,
        query: str,
    ):
        return await self.repository.create_message(
            session=session,
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
        )

    async def save_ai_response(
        self,
        session: AsyncSession,
        record_id: int,
        ai_response: str,
    ):
        return await self.repository.set_ai_response(
            session=session,
            record_id=record_id,
            ai_response=ai_response,
        )
