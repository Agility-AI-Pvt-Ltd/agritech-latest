# Kisan Mitra

Kisan Mitra is an agritech advisory backend for maize workflows. It combines a FastAPI API, a LangGraph-powered chat agent, Qdrant-based retrieval, PostgreSQL-backed persistence, Redis caching, weather lookup, and optional speech endpoints.

The repository currently supports two main product flows:

- `POST /api/advisory` and `POST /api/advisory/predefined` for direct advisory generation from sowing date, weather, and retrieved context.
- `POST /api/chat` for multi-turn conversational advisory with memory, safety screening, tool use, sowing-date capture, and maize FAQ routing.

## Features

- LangGraph chat agent with tool-calling loops
- Safety gate before the chat agent runs
- Persistent user profile and conversation state
- Redis-backed active chat state caching
- Qdrant retrieval over multiple maize manuals
- Stage-aware maize FAQ retrieval from `data/maize_knowledge_tree.json`
- Weather lookup via Open-Meteo
- Speech-to-text and text-to-speech wrappers via Sarvam
- Local JSON logging for tool calls and chat sessions

## Project Layout

- `main.py`: app startup, database init, retrieval checks, chat resource warmup
- `app.py`: FastAPI app and root endpoint
- `api/`: API routes and dependency wiring
- `services/`: non-chat advisory services, weather, speech, vector search
- `pipeline/`: chat agent graph, prompts, guards, tools, and chat persistence
- `schemas/`: request and response models
- `scripts/`: ingestion scripts
- `data/`: source PDFs and maize FAQ knowledge tree
- `sandbox/`: local test scripts and experimental UI
- `docs/`: architecture notes and diagram

## Requirements

- Python 3.11+
- PostgreSQL
- Redis
- Local filesystem access for Qdrant storage

## Environment

Create a `.env` file in the repo root.

```env
# Main chat model
LLM_PROVIDER=openai
LLM_LARGE_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_key

# Optional alternative providers
# GOOGLE_API_KEY=your_google_key
# NVIDIA_API_KEY=your_nvidia_key

# Safety classifier model
SAFETY_LLM_PROVIDER=openai
SAFETY_LLM_MODEL=gpt-4o-mini
SAFETY_LLM_TEMPERATURE=0.0

# Embeddings and retrieval
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=agritech_knowledge
QDRANT_COLLECTION_NAMES=spring_corn_fertilizers_db,maize_production_manual_db,spring_corn_pest_and_diseases_db,spring_corn_pop_db,farmerbook_db
MAIZE_FAQ_COLLECTION_NAME=maize_faq_db
MAIZE_FAQ_TREE_PATH=./data/maize_knowledge_tree.json
RETRIEVAL_MODE=hybrid
HYBRID_TOP_K=8
BM25_TOP_K=6
BM25_MARKDOWN_DIR=./data/markdowns
BM25_CHUNK_WORDS=260

# Databases
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=agritech
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
REDIS_URL=redis://redis:6379/0
DATABASE_AUTO_CREATE_TABLES=true

# Chat safety
CHAT_SAFETY_ENABLED=true
CHAT_SAFETY_FAIL_CLOSED=true
CHAT_SECURITY_BLOCKING_ENABLED=true
CHAT_LOW_INFO_BLOCKING_ENABLED=true
CHAT_SAFETY_MODEL_CLASSIFICATION_ENABLED=true

# Optional speech
SARVAM_API_KEY=your_sarvam_key
SARVAM_BASE_URL=https://api.sarvam.ai
```

Notes:

- `DATABASE_URL` is optional. If omitted, the app builds it from `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, and `POSTGRES_PORT`.
- The same `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` keys can be passed to the official Postgres Docker image to create the user/database automatically.
- Chat and direct advisory generation default to `LLM_PROVIDER=openai`.
- `GOOGLE_API_KEY` is optional and only needed if you explicitly switch a provider to Google.

## Install

```bash
uv sync
```

For the optional Docling ingestion script:

```bash
uv sync --extra docling
```

## Ingest Data

For deployment, create the PageIndex tree if missing and create Qdrant embeddings from every markdown file under `data/markdowns`:

```bash
uv run python scripts/deploy_ingest.py
```

To force PageIndex tree regeneration:

```bash
uv run python scripts/deploy_ingest.py --rebuild-pageindex
```

To also ingest the maize FAQ knowledge tree:

```bash
uv run python scripts/deploy_ingest.py --include-faq
```

Optional Docling PDF ingestion path:

```bash
uv sync --extra docling
uv run python scripts/ingestion_docling.py
```

Expected markdown source files:

- `data/markdowns/fertilizer-materials-and-computation.md`
- `data/markdowns/MAIZE PRODUCTION MANUAL.md`
- `data/markdowns/Management_Pests_Diseases_Manual.md`
- `data/markdowns/Spring Sweet Corn (zaid Maize) – Package Of Practices (pop) _ Uttar Pradesh.md`
- `data/markdowns/farmerbook.md`

## Run

Start the API:

```bash
python main.py
```

The server runs on `http://localhost:8000` and exposes Swagger docs at `http://localhost:8000/docs`.

Optional local UI:

```bash
streamlit run sandbox/streamlit_app.py
```

## API Endpoints

- `GET /`: root summary
- `GET /api/health`: retrieval health check
- `GET /api/questions`: predefined advisory prompts
- `POST /api/advisory`: direct advisory for a custom query
- `POST /api/advisory/predefined`: advisory for a predefined question choice
- `POST /api/chat`: multi-turn agentic advisory chat
- `GET /api/profile/{user_id}`: stored user profile from the chat pipeline
- `POST /api/stt`: speech-to-text
- `POST /api/tts`: text-to-speech

## Example Requests

Chat:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "farmer-1",
    "conversation_id": "conv-1",
    "query": "मेरी मक्का की फसल में कीड़े लग रहे हैं, क्या करूं?"
  }'
```

Direct advisory:

```bash
curl -X POST http://localhost:8000/api/advisory \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What farm operations should I do today?",
    "sowing_date": "2026-03-01",
    "latitude": 26.8,
    "longitude": 80.9,
    "user_id": "farmer-1",
    "conversation_id": "conv-2"
  }'
```

Speech-to-text:

```bash
curl -X POST http://localhost:8000/api/stt \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "BASE64_AUDIO_HERE",
    "audio_mime_type": "audio/wav"
  }'
```

## Chat Pipeline Notes

- The chat graph runs a safety node before the main agent loop.
- The agent can call tools for weather, datetime, web search, RAG, BigHaat, and maize FAQ retrieval.
- For stage-dependent maize queries, the agent can ask for sowing date, persist it, derive crop stage, and then resume the original user intent.
- Chat state is cached in Redis and also persisted in PostgreSQL through `pipeline/database/`.

## Observability

- Tool calls are logged under `logs/tool_calls/`
- Chat session summaries and LLM call logs are written under `logs/chat_sessions/`
- Qdrant and PageIndex logging paths are configurable in `core/config.py`

## Testing

There is no single formal test suite entrypoint yet, but the repo includes local scripts under `sandbox/`, for example:

```bash
python sandbox/run_test_cases.py
python sandbox/test_chat.py
python sandbox/test_tools.py
```

These are better treated as developer smoke tests than as a stable CI suite.

## Architecture Notes

The codebase currently has two persistence layers in parallel:

- `core/database.py` plus `models/` and `repositories/` for the async SQLAlchemy-backed advisory endpoints
- `pipeline/database/` for the chat agent's user profile and conversation-state storage

For a quick architecture walkthrough, see `docs/codebase_architecture.md`.
