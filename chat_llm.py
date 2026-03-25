"""
chat_llm.py  –  LLM + Qdrant factory for the /api/chat agent.

Extracted from temp/main.py so it does not conflict with v1's main.py.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

_embedding_model = None

def get_embedding_model():
    """Return a singleton SentenceTransformer encoder."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        print(f"[*] Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def get_llm():
    """Return whichever LLM is configured via env vars."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_LARGE_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )

    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=os.getenv("NVIDIA_LARGE_MODEL", "meta/llama-3.1-70b-instruct"),
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.3,
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("LLM_LARGE_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    )


def get_qdrant_client():
    """Return an initialized QdrantClient pointing at local db_storage."""
    try:
        from qdrant_client import QdrantClient
        from core.config import settings
        qdrant_path = os.getenv("QDRANT_PATH", settings.qdrant_path)
        os.makedirs(qdrant_path, exist_ok=True)
        client = QdrantClient(path=qdrant_path)
        print(f"[*] Qdrant loaded from: {qdrant_path}")
        return client
    except Exception as e:
        print(f"[!] Qdrant init failed: {e} — RAG search will be unavailable.")
        return None
