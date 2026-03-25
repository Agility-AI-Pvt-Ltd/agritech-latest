import uvicorn
from contextlib import asynccontextmanager
from app import app
from api.dependencies import get_vector_store, get_pageindex_provider
from api.routes import init_chat_resources_on_startup
from core.config import settings
from core.database import db_manager

# ==========================================
# STARTUP EVENT
# ==========================================


@asynccontextmanager
async def lifespan(_app):
    """Run startup/shutdown lifecycle hooks."""
    print("\n" + "="*50)
    print("    KISAN MITRA - STARTING UP")
    print("="*50)
    
    mode = settings.retrieval_mode.strip().lower()
    print(f"[*] Retrieval mode: {mode}")

    await db_manager.init(auto_create_tables=settings.database_auto_create_tables)
    print("[✓] PostgreSQL engine initialized with async pool")

    # Initialise agent DB tables (user_profiles, conversation_states)
    try:
        import db as agent_db
        agent_db.init_db()
    except Exception as _e:
        print(f"[!] Agent DB init warning: {_e}")

    if mode == "pageindex":
        pageindex_provider = get_pageindex_provider()
        if pageindex_provider.is_loaded():
            print("[✓] PageIndex loaded successfully!")
        else:
            print("[!] WARNING: PageIndex failed to load!")
    else:
        # Default: RAG (Qdrant)
        vector_store = get_vector_store()
        if vector_store.is_loaded():
            collections = ", ".join(settings.resolved_qdrant_collections)
            print(f"[✓] Qdrant Knowledge Base loaded (checked collections: {collections})")
        else:
            print("[!] WARNING: Qdrant knowledge base not loaded. Run scripts/ingest.py or scripts/ingestion_docling.py first.")

        # Initialize chat agent resources at startup so tool calls never run with None client
        _llm, chat_qdrant = init_chat_resources_on_startup()
        if chat_qdrant is not None:
            print("[✓] Chat Qdrant client initialized.")
        else:
            print("[!] WARNING: Chat Qdrant client failed to initialize.")
    
    print("="*50 + "\n")
    yield

    await db_manager.dispose()
    print("[✓] PostgreSQL engine disposed")

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
