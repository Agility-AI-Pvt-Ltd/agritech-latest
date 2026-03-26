"""
pipeline/llm_factory.py  –  LLM + Qdrant + Embedding model factory.

Provides singleton-safe getters for the three shared expensive objects
used by the /api/chat pipeline.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from core.config import settings

load_dotenv()

_embedding_model = None
_safety_llm = None


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
    """Return whichever LLM is configured via env vars.

    Supported providers (LLM_PROVIDER env var): google | nvidia | openai (default).
    """
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


_qdrant_client = None


def get_qdrant_client():
    """Return an initialized QdrantClient pointing at local db_storage."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        # Try to reuse the single backend vectorstore to avoid file locking issues
        try:
            from api.dependencies import get_vector_store
            store = get_vector_store()
            if store and store.is_loaded():
                _qdrant_client = store.get_client()
                return _qdrant_client
        except ImportError:
            pass

        from qdrant_client import QdrantClient
        from core.config import settings
        qdrant_path = os.getenv("QDRANT_PATH", settings.qdrant_path)
        os.makedirs(qdrant_path, exist_ok=True)
        _qdrant_client = QdrantClient(path=qdrant_path)
        print(f"[*] Qdrant loaded from: {qdrant_path}")
        return _qdrant_client
    except Exception as e:
        print(f"[!] Qdrant init failed: {e} — RAG search will be unavailable.")
        return None


def get_safety_llm():
    """Return a singleton classifier LLM for the pre-agent safety gate."""
    global _safety_llm
    if _safety_llm is not None:
        return _safety_llm

    provider = settings.safety_llm_provider.strip().lower()

    try:
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _safety_llm = ChatGoogleGenerativeAI(
                model=settings.safety_llm_model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=settings.safety_llm_temperature,
            )
            return _safety_llm

        if provider == "nvidia":
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            _safety_llm = ChatNVIDIA(
                model=settings.safety_llm_model,
                api_key=os.getenv("NVIDIA_API_KEY"),
                temperature=settings.safety_llm_temperature,
            )
            return _safety_llm

        from langchain_openai import ChatOpenAI
        _safety_llm = ChatOpenAI(
            model=settings.safety_llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=settings.safety_llm_temperature,
        )
        return _safety_llm
    except Exception as e:
        print(f"[!] Safety LLM init failed: {e}")
        return None
