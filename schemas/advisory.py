from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from core.config import settings

class AdvisoryRequest(BaseModel):
    """Request body for advisory endpoint"""
    user_query: str
    sowing_date: str
    latitude: float = settings.default_latitude
    longitude: float = settings.default_longitude
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class PredefinedQuestionRequest(BaseModel):
    """Request body for predefined question endpoint"""
    choice: str
    sowing_date: str
    latitude: float = settings.default_latitude
    longitude: float = settings.default_longitude
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class AdvisoryResponse(BaseModel):
    """Response model for advisory"""
    user_query: str
    crop_stage: str
    weather_current: Dict[str, Any]
    weather_forecast: List[Dict[str, Any]]
    advisory: str

class QuestionResponse(BaseModel):
    """Response model for questions"""
    questions: Dict[str, str]
