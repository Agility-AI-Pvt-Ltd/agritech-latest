"""
FastAPI Application Setup
Creates and configures the FastAPI app instance with middleware
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.auth import router as auth_router
from api.routes import router
from core.config import settings

app = FastAPI(
    title="Kisan Mitra - Smart Advisory System",
    description="UP-style Kisan-friendly crop advisory powered by LangGraph & LLM",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.resolved_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(router)

@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Namaste! Kisan Mitra advisory system ready!",
        "docs": "/docs",
        "api_endpoints": {
            "chat":                "POST /api/chat",
            "advisory":            "POST /api/advisory",
            "predefined_advisory": "POST /api/advisory/predefined",
            "questions":           "GET  /api/questions",
            "health":              "GET  /api/health",
            "profile":             "GET  /api/profile/{user_id}",
        }
    }
