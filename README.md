# 🌾 Kisan Mitra: Agentic Agritech Advisory

Kisan Mitra is a production-ready, agentic RAG (Retrieval-Augmented Generation) system designed to provide expert agricultural advisory to Indian farmers. It specializes in **Spring Corn (Zaid Maize)** cultivation in Uttar Pradesh, leveraging real-time weather data, deep agricultural knowledge bases, and persistent user profiling.

---

## 🚀 Recent Updates

- **🛡️ Safety Gate Before `/api/chat`**: Prompt injection, secret-exfiltration attempts, malware-style prompts, and low-information garbage inputs are blocked before the main agent loop.
- **🧠 Persistent Chat Memory**: Conversation state is cached in Redis and stored in PostgreSQL, including user location, sowing date, and pending clarification state.
- **🧩 Modular Chat API Schemas**: Chat request/response models are separated into `schemas/chat.py` so the route layer stays cleaner and easier to scale.
- **🔌 Consistent FastAPI Dependency Injection**: `/api/chat` now uses the same dependency-injection pattern as the advisory endpoints, with chat resources initialized through `api/dependencies.py`.
- **🌱 Sowing Date + Crop Stage Memory**: When the farmer gives a maize sowing date, the agent stores it, resolves the current crop stage, and reuses it in later turns.
- **📚 Maize FAQ Knowledge Base**: Added a maize FAQ knowledge source backed by Qdrant. It supports:
  - direct exact-match lookup from `knowledge_tree`
  - semantic search over `rag_entries.search_text`
  - stage-filtered retrieval when crop stage is known
  - all-stage FAQ search when crop stage is not yet known
- **🧭 Deterministic FAQ Router**: Maize FAQ/advisory queries are routed deterministically. For maize questions that need stage awareness, the agent asks for sowing date first, stores crop stage, then uses FAQ retrieval. When `rag_search` is called and crop stage is known, FAQ is called first and its answer is intended to lead the final response.
- **📝 Local Execution Tracing**: Tool inputs/outputs and LLM call metadata are written locally under `logs/tool_calls/` and `logs/chat_sessions/` for debugging and auditability.
- **🔁 Resume Pending Intent**: If the agent asks for sowing date first, the next reply saves the date and continues answering the original unresolved maize question in the same turn.

---

## 🌟 Core Features

- **Agentic RAG**: Powered by **LangGraph**, the agent can reason, use multiple tools, and loop until it finds a high-quality, actionable answer.
- **Multi-Source Knowledge**: Ingests multiple PDF manuals (Fertilizers, Pests, Production, POP) using **IBM Docling** for advanced semantic hierarchical chunking.
- **Stage-Aware Maize FAQ Retrieval**:
  - FAQ vectors are built from [`data/maize_knowledge_tree.json`](./data/maize_knowledge_tree.json)
  - `search_text` is embedded into a dedicated Qdrant FAQ collection
  - if `crop_stage` is known, the agent hard-filters by `crop_stage`, then semantically searches within that stage
  - if `crop_stage` is unknown, the FAQ tool can still search across all maize FAQ entries
  - exact stage + exact question matches use a zero-LLM-cost direct lookup path
- **Optimized Multi-Collection RAG**:
  - `rag_search` generates 4 targeted sub-queries for POP, fertilizer, pest, and production manuals
  - the 4 sub-queries are embedded together in one batch
  - the 4 collection lookups are then executed concurrently with a thread pool for lower latency on the current multi-collection layout
- **Persistent Memory**:
  - **PostgreSQL**: Long-term storage for user profiles (crops, farm size, location) and full conversation history.
  - **Redis**: Ultra-fast caching of the active `AgentState` for seamless multi-turn chat.
- **Smart Profiling**: Automatically extracts and remembers user facts (like name, farm details) as they chat to personalize future advice.
- **Maize State Tracking**: The chat pipeline persists maize sowing date and resolved crop stage in:
  - Redis active state
  - PostgreSQL conversation state
  - PostgreSQL user profile
- **Real-time Tools**:
  - `rag_search`: Semantic search across domain-specific agricultural vector collections. If crop stage is known, the agent calls FAQ first and then RAG for supporting context.
  - `faq_search_by_crop_stage`: Maize FAQ lookup (exact match first, then semantic search). Uses stage filtering when crop stage is known, otherwise it can search across all FAQ stages.
  - `set_crop_stage`: Resolve the current maize crop stage from sowing date and persist it in chat state.
  - `bighaat_search`: Search India's largest agri-platform for product prices, links, and blog advice.
  - `get_weather`: Current weather and 3-day forecast using exact latitude/longitude.
  - `geocode_location`: Resolve village/city names into exact coordinates before weather lookup.
  - `web_search`: Fallback for the latest 7-day news via DuckDuckGo and Google News.
  - `get_current_datetime`: Contextual awareness of date, time, and farming season.

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Orchestration**: LangGraph & LangChain
- **API Framework**: FastAPI
- **Vector Database**: Qdrant (Local persistent storage)
- **Primary Database**: PostgreSQL (Profiles & History)
- **Cache**: Redis (Active state storage)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **UI**: Streamlit (Premium dark-themed interface)

---

## ⚙️ Setup & Installation

### 1. Requirements
Ensure you have **Python 3.11+**, **PostgreSQL**, and **Redis** installed.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
# AI & LLM
GOOGLE_API_KEY=your_gemini_api_key
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=openai
LLM_LARGE_MODEL=gpt-4o-mini

# Databases
DATABASE_URL=postgresql://user:pass@localhost:5432/agritech
REDIS_URL=redis://localhost:6379/0

# Storage & Paths
QDRANT_PATH=./db_storage/qdrant
RETRIEVAL_MODE=rag
MAIZE_FAQ_COLLECTION_NAME=maize_faq_db
MAIZE_FAQ_TREE_PATH=./data/maize_knowledge_tree.json

# Safety
CHAT_SAFETY_ENABLED=true
CHAT_SECURITY_BLOCKING_ENABLED=true
CHAT_LOW_INFO_BLOCKING_ENABLED=true
CHAT_SAFETY_MODEL_CLASSIFICATION_ENABLED=true
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Application

### Start the Backend (FastAPI)
```bash
python main.py
```
*The API is available at `http://localhost:8000`. Docs at `/docs`.*

### Ingest Knowledge Bases
PDF manuals:
```bash
python scripts/ingestion_docling.py
```

Maize FAQ knowledge tree:
```bash
python scripts/ingest_maize_faq.py
```

### Start the Frontend (Streamlit)
```bash
streamlit run sandbox/streamlit_app.py
```

### Run Tests (Sandbox)
```bash
python sandbox/run_test_cases.py
```

---

## 🏗️ Architecture Directory Structure

- `schemas/`: Pydantic request/response models shared across API endpoints.
- `pipeline/`: Core agent logic.
  - `graph.py`: LangGraph definition & state management.
  - `agent.py`: Main agent node, deterministic prechecks, and clarification resume logic.
  - `agent_guards.py`: Deterministic routing for sowing-date capture, datetime, and maize FAQ pre-routing.
  - `tools/`: Central tool dispatcher & individual integrations.
  - `prompts/`: Core system and conditional prompts.
- `api/`: FastAPI routes plus dependency providers for advisory and chat endpoints.
- `data/maize_knowledge_tree.json`: Stage-wise maize FAQ tree plus `rag_entries` for FAQ vector ingestion.
- `scripts/ingest_maize_faq.py`: FAQ ingestion pipeline for the maize knowledge tree.
- `sandbox/`: Test cases, playground scripts, and the Streamlit frontend.
- `core/`: Database configurations and settings.

---

## 🔎 Observability

- **Tool call logs**: Every tool invocation is logged as one JSON file under `logs/tool_calls/`, including tool name, input params, output payload, status, `conversation_id`, `user_id`, and `call_id`.
- **LLM call logs**: Per-conversation LLM metadata is written under `logs/chat_sessions/` as `<conversation_id>.summary.json` and `<conversation_id>.llm_calls.jsonl`.
- **LangSmith**: LangSmith tracing is not wired into the project yet. Current tracing is local-file based.

---

## 🌾 Usage Tips
- **Hinglish/Hindi**: The agent prefers Hindi (Devanagari) but understands Hinglish and English.
- **Weather Flow**: The weather tool requires exact coordinates. If the user location is not known, the agent first asks for location, calls `geocode_location`, then calls `get_weather`.
- **Product Search**: Ask "खाद कहाँ से खरीदूँ?" or "मक्के के बीज की कीमत क्या है?" to trigger the BigHaat tool.
- **Maize Advisory Flow**: For stage-dependent maize advice, the agent should ask for sowing date, not crop stage. Once sowing date is provided, it stores the date, derives crop stage with `set_crop_stage`, and then answers the original question.
- **FAQ Retrieval Logic**:
  - maize FAQ/advisory questions can be routed to the FAQ tool deterministically
  - the FAQ tool receives the original `query`, optional `crop_stage`, and `top_k`
  - the FAQ semantic search translates the query into English before embedding
  - exact question matches are handled directly from the JSON tree
  - when crop stage is known and `rag_search` is used, FAQ is called first and should lead the final answer, with broader RAG/manual context added after it
