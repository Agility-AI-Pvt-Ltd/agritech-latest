from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Date, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class AdvisoryLog(Base):
    __tablename__ = "advisory_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    request_type: Mapped[str] = mapped_column(String(20), nullable=False)
    question_choice: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    sowing_date: Mapped[date] = mapped_column(Date, nullable=False)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    crop_stage: Mapped[str] = mapped_column(String(100), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(30), nullable=False)
    weather_current: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    weather_forecast: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    advisory: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
