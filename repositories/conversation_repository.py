from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.conversation import Conversation


class ConversationRepository:
    async def create_message(
        self,
        session: AsyncSession,
        user_id: str,
        conversation_id: str,
        query: str,
    ) -> Conversation:
        record = Conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            query=query,
            ai_response=None,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record

    async def set_ai_response(
        self,
        session: AsyncSession,
        record_id: int,
        ai_response: str,
    ) -> Optional[Conversation]:
        result = await session.execute(select(Conversation).where(Conversation.id == record_id))
        record = result.scalar_one_or_none()
        if record is None:
            return None

        record.ai_response = ai_response
        await session.commit()
        await session.refresh(record)
        return record
