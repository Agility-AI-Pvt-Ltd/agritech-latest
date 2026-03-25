from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from models.advisory_log import AdvisoryLog


class AdvisoryRepository:
    async def create_log(
        self,
        session: AsyncSession,
        user_id: str,
        request_type: str,
        user_query: str,
        sowing_date: date,
        latitude: float,
        longitude: float,
        crop_stage: str,
        retrieval_mode: str,
        weather_current: Dict[str, Any],
        weather_forecast: List[Dict[str, Any]],
        advisory: str,
        question_choice: Optional[str] = None,
    ) -> AdvisoryLog:
        record = AdvisoryLog(
            user_id=user_id,
            request_type=request_type,
            question_choice=question_choice,
            user_query=user_query,
            sowing_date=sowing_date,
            latitude=latitude,
            longitude=longitude,
            crop_stage=crop_stage,
            retrieval_mode=retrieval_mode,
            weather_current=weather_current,
            weather_forecast=weather_forecast,
            advisory=advisory,
        )

        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record
