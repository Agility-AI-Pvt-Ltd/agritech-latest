from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import logging
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from schemas.advisory import AdvisoryRequest, PredefinedQuestionRequest, AdvisoryResponse, QuestionResponse
from schemas.chat import ChatRequest, ChatResponse
from schemas.speech import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from core.config import settings

from services.weather import WeatherProvider
from services.vectorstore import VectorStoreProvider
from services.pageindex import PageIndexProvider
from services.advisory import LangGraphAdvisoryGenerator
from services.advisory_log import AdvisoryLogService
from services.conversation import ConversationService
from services.crop import CropService
from services.speech import SarvamSpeechService
from api.dependencies import (
    get_weather_provider,
    get_vector_store,
    get_pageindex_provider,
    get_advisory_generator,
    get_crop_service,
    get_advisory_log_service,
    get_conversation_service,
    get_speech_service,
    get_chat_llm,
    get_chat_safety_llm,
    get_chat_qdrant_client,
    get_db_session,
)
from api.auth import AuthenticatedUser, get_current_user
from api.rate_limit import consume_chat_quota

router = APIRouter(prefix="/api", tags=["advisory"])
logger = logging.getLogger(__name__)

def _get_context(
    query: str,
    stage: str,
    vector_store: VectorStoreProvider,
    pageindex_provider: PageIndexProvider,
) -> str:
    mode = settings.retrieval_mode.strip().lower()

    if mode == "pageindex":
        if not pageindex_provider.is_loaded():
            raise HTTPException(status_code=500, detail="PageIndex data nahi load hua!")

        context = pageindex_provider.search(query)
        if not context:
            raise HTTPException(status_code=500, detail="PageIndex se context nahi mila!")
        return context

    if mode == "hybrid":
        from services.hybrid_retrieval import hybrid_context

        context = hybrid_context(
            query=query,
            stage=stage,
            vector_store=vector_store,
            pageindex_provider=pageindex_provider,
            top_k=settings.hybrid_top_k,
        )
        if not context:
            raise HTTPException(status_code=500, detail="Hybrid retrieval se context nahi mila!")
        return context

    # Default: RAG
    if not vector_store.is_loaded():
        raise HTTPException(status_code=500, detail="Knowledge base nahi load hua!")

    docs = vector_store.search(f"{query}\nSchedule and chemicals for {stage} maize")
    return "\n".join([d.page_content for d in docs])

@router.post("/advisory", response_model=AdvisoryResponse)
async def get_advisory(
    request: AdvisoryRequest,
    weather_provider: WeatherProvider = Depends(get_weather_provider),
    vector_store: VectorStoreProvider = Depends(get_vector_store),
    pageindex_provider: PageIndexProvider = Depends(get_pageindex_provider),
    advisory_generator: LangGraphAdvisoryGenerator = Depends(get_advisory_generator),
    crop_service: CropService = Depends(get_crop_service),
    advisory_log_service: AdvisoryLogService = Depends(get_advisory_log_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Get crop advisory based on custom query with 3-day forecast"""
    try:
        sowing_dt = datetime.strptime(request.sowing_date, "%Y-%m-%d")
        if sowing_dt > datetime.now():
            raise HTTPException(status_code=400, detail="Future date nahi chalegi!")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date format sahi karein (YYYY-MM-DD)")
    
    stage = crop_service.calculate_crop_stage(request.sowing_date)

    user_id = request.user_id or "anonymous"
    conversation_id = request.conversation_id or str(uuid4())
    conversation_row_id = None
    try:
        conversation_row = await conversation_service.save_incoming_query(
            session=db_session,
            user_id=user_id,
            conversation_id=conversation_id,
            query=request.user_query,
        )
        conversation_row_id = conversation_row.id
    except Exception as exc:
        logger.warning("Failed to store incoming conversation query: %s", exc)

    weather_current, weather_forecast = weather_provider.fetch_weather(request.latitude, request.longitude)

    context = _get_context(
        query=request.user_query,
        stage=stage,
        vector_store=vector_store,
        pageindex_provider=pageindex_provider,
    )
    
    advisory = advisory_generator.generate_advisory(
        request.user_query, stage, weather_current, weather_forecast, context
    )

    if conversation_row_id is not None:
        try:
            await conversation_service.save_ai_response(
                session=db_session,
                record_id=conversation_row_id,
                ai_response=advisory,
            )
        except Exception as exc:
            logger.warning("Failed to store AI response in conversation: %s", exc)

    try:
        await advisory_log_service.save_advisory_log(
            session=db_session,
            user_id=user_id,
            request_type="custom",
            user_query=request.user_query,
            sowing_date=sowing_dt.date(),
            latitude=request.latitude,
            longitude=request.longitude,
            crop_stage=stage,
            retrieval_mode=settings.retrieval_mode.strip().lower(),
            weather_current=weather_current,
            weather_forecast=weather_forecast,
            advisory=advisory,
        )
    except Exception as exc:
        logger.warning("Failed to persist advisory log: %s", exc)
    
    return AdvisoryResponse(
        user_query=request.user_query,
        crop_stage=stage,
        weather_current=weather_current,
        weather_forecast=weather_forecast,
        advisory=advisory
    )

@router.post("/advisory/predefined", response_model=AdvisoryResponse)
async def get_predefined_advisory(
    request: PredefinedQuestionRequest,
    weather_provider: WeatherProvider = Depends(get_weather_provider),
    vector_store: VectorStoreProvider = Depends(get_vector_store),
    pageindex_provider: PageIndexProvider = Depends(get_pageindex_provider),
    advisory_generator: LangGraphAdvisoryGenerator = Depends(get_advisory_generator),
    crop_service: CropService = Depends(get_crop_service),
    advisory_log_service: AdvisoryLogService = Depends(get_advisory_log_service),
    conversation_service: ConversationService = Depends(get_conversation_service),
    db_session: AsyncSession = Depends(get_db_session),
):
    """Get advisory using predefined questions with 3-day forecast"""
    
    question = settings.predefined_questions.get(request.choice)
    if not question:
        raise HTTPException(status_code=400, detail=f"Choice '{request.choice}' invalid hai!")
    
    try:
        sowing_dt = datetime.strptime(request.sowing_date, "%Y-%m-%d")
        if sowing_dt > datetime.now():
            raise HTTPException(status_code=400, detail="Future date nahi chalegi!")
    except ValueError:
        raise HTTPException(status_code=400, detail="Date format sahi karein (YYYY-MM-DD)")
    
    stage = crop_service.calculate_crop_stage(request.sowing_date)

    user_id = request.user_id or "anonymous"
    conversation_id = request.conversation_id or str(uuid4())
    conversation_row_id = None
    try:
        conversation_row = await conversation_service.save_incoming_query(
            session=db_session,
            user_id=user_id,
            conversation_id=conversation_id,
            query=question,
        )
        conversation_row_id = conversation_row.id
    except Exception as exc:
        logger.warning("Failed to store incoming conversation query: %s", exc)

    weather_current, weather_forecast = weather_provider.fetch_weather(request.latitude, request.longitude)

    context = _get_context(
        query=question,
        stage=stage,
        vector_store=vector_store,
        pageindex_provider=pageindex_provider,
    )
    
    advisory = advisory_generator.generate_advisory(
        question, stage, weather_current, weather_forecast, context
    )

    if conversation_row_id is not None:
        try:
            await conversation_service.save_ai_response(
                session=db_session,
                record_id=conversation_row_id,
                ai_response=advisory,
            )
        except Exception as exc:
            logger.warning("Failed to store AI response in conversation: %s", exc)

    try:
        await advisory_log_service.save_advisory_log(
            session=db_session,
            user_id=user_id,
            request_type="predefined",
            question_choice=request.choice,
            user_query=question,
            sowing_date=sowing_dt.date(),
            latitude=request.latitude,
            longitude=request.longitude,
            crop_stage=stage,
            retrieval_mode=settings.retrieval_mode.strip().lower(),
            weather_current=weather_current,
            weather_forecast=weather_forecast,
            advisory=advisory,
        )
    except Exception as exc:
        logger.warning("Failed to persist advisory log: %s", exc)
    
    return AdvisoryResponse(
        user_query=question,
        crop_stage=stage,
        weather_current=weather_current,
        weather_forecast=weather_forecast,
        advisory=advisory
    )


    

@router.get("/questions", response_model=QuestionResponse)
async def get_questions():
    """Get all available predefined questions"""
    return QuestionResponse(questions=settings.predefined_questions)

@router.get("/health")
async def health_check(
    vector_store: VectorStoreProvider = Depends(get_vector_store),
    pageindex_provider: PageIndexProvider = Depends(get_pageindex_provider)
):
    """Health check endpoint"""
    mode = settings.retrieval_mode.strip().lower()
    if mode == "pageindex":
        if not pageindex_provider.is_loaded():
            return {"status": "unhealthy", "message": "PageIndex data nahi load hua"}
        return {"status": "healthy", "message": "All systems go!", "mode": mode}

    if mode == "hybrid":
        from services.bm25 import get_bm25_retriever

        vector_loaded = vector_store.is_loaded()
        bm25_loaded = get_bm25_retriever().is_loaded()
        pageindex_loaded = pageindex_provider.is_loaded()
        if not vector_loaded and not bm25_loaded and not pageindex_loaded:
            return {"status": "unhealthy", "message": "No hybrid retrieval source loaded", "mode": mode}
        return {
            "status": "healthy",
            "message": "All systems go!",
            "mode": mode,
            "sources": {
                "rag": vector_loaded,
                "bm25": bm25_loaded,
                "pageindex": pageindex_loaded,
            },
        }

    if not vector_store.is_loaded():
        return {"status": "unhealthy", "message": "Knowledge base nahi load hua", "mode": mode}
    return {"status": "healthy", "message": "All systems go!"}


@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    llm=Depends(get_chat_llm),
    safety_llm=Depends(get_chat_safety_llm),
    qdrant_client=Depends(get_chat_qdrant_client),
) -> ChatResponse:
    """
    Conversational agricultural advisory endpoint powered by LangGraph agent.
    Pass user_id + conversation_id on every call — conversation state is
    persisted in PostgreSQL so context is maintained across requests.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    from pipeline.graph import run

    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client is not initialized. Please check startup logs and QDRANT_PATH.")

    user_id = user["sub"]
    quota = consume_chat_quota(user)

    result = run(
        query=req.query,
        llm=llm,
        safety_llm=safety_llm,
        qdrant_client=qdrant_client,
        conversation_id=req.conversation_id,
        user_id=user_id,
    )

    tools_used = [tc["tool"] for tc in result.get("tool_calls", [])]

    return ChatResponse(
        response=result.get("final_response", ""),
        conversation_id=req.conversation_id,
        user_id=user_id,
        tools_used=tools_used,
        loop_count=result.get("loop_count", 0),
        rate_limit_remaining=quota["remaining"],
        rate_limit_limit=quota["limit"],
    )


@router.post("/stt", response_model=SpeechToTextResponse)
def speech_to_text(
    req: SpeechToTextRequest,
    _user: AuthenticatedUser = Depends(get_current_user),
    speech_service: SarvamSpeechService = Depends(get_speech_service),
) -> SpeechToTextResponse:
    """Thin wrapper around Sarvam speech-to-text."""
    try:
        result = speech_service.speech_to_text(
            audio_base64=req.audio_base64,
            audio_mime_type=req.audio_mime_type,
        )
    except Exception as exc:
        logger.warning("Speech-to-text failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Speech-to-text failed: {exc}") from exc

    return SpeechToTextResponse(
        text=result.transcript,
        language_code=result.language_code,
        request_id=result.request_id,
    )


@router.post("/tts", response_model=TextToSpeechResponse)
def text_to_speech(
    req: TextToSpeechRequest,
    _user: AuthenticatedUser = Depends(get_current_user),
    speech_service: SarvamSpeechService = Depends(get_speech_service),
) -> TextToSpeechResponse:
    """Thin wrapper around Sarvam text-to-speech."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        result = speech_service.text_to_speech(text=req.text)
    except Exception as exc:
        logger.warning("Text-to-speech failed: %s", exc)
        raise HTTPException(status_code=400, detail=f"Text-to-speech failed: {exc}") from exc

    return TextToSpeechResponse(
        audio_base64=result.audio_base64,
        audio_mime_type=result.mime_type,
        request_id=result.request_id,
    )


@router.get("/profile/{user_id}")
def get_user_profile(user_id: str) -> dict:
    """Return the stored user profile for a given user_id (from agent's DB)."""
    import pipeline.database as db
    profile = db.load_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for user_id={user_id!r}")
    return profile
