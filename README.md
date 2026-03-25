# 🌾 Kisan Mitra: Agentic Agritech Advisory (v8)

Kisan Mitra is a production-ready, agentic RAG (Retrieval-Augmented Generation) system designed to provide expert agricultural advisory to Indian farmers. It specializes in **Spring Corn (Zaid Maize)** cultivation in Uttar Pradesh, leveraging real-time weather data, deep agricultural knowledge bases, and persistent user profiling.

---

## 🚀 Recent Updates (v8)

- **🌦️ Address-Aware Weather**: The `get_weather` tool now supports direct location names (e.g., "Meerut", "Sitapur"). It automatically resolves village/city names to exact coordinates using an integrated geocoding fallback.
- **🛠️ Robust BigHaat Integration**: Refactored the `bighaat_search` tool to use a robust Google News RSS backend. This eliminates 404 errors, handles Hindi queries perfectly, and extracts live pricing from search results.
- **⚡ Performance Singleton**: The embedding model loading is now a singleton pattern, ensuring `SentenceTransformer` is loaded once at startup for high-speed RAG searches.
- **📅 Fresh News Filter**: Web search results (DuckDuckGo + Google News) are strictly filtered for the **last 7 days** to ensure time-sensitive advisory.
- **🛡️ Clean Repo**: Updated `.gitignore` to exclude local `logs/` and temporary artifacts.

---

## 🌟 Core Features

- **Agentic RAG**: Powered by **LangGraph**, the agent can reason, use multiple tools, and loop until it finds a high-quality, actionable answer.
- **Multi-Source Knowledge**: Ingests multiple PDF manuals (Fertilizers, Pests, Production, POP) using **IBM Docling** for advanced semantic hierarchical chunking.
- **Persistent Memory**:
  - **PostgreSQL**: Long-term storage for user profiles (crops, farm size, location) and full conversation history.
  - **Redis**: Ultra-fast caching of the active `AgentState` for seamless multi-turn chat.
- **Smart Profiling**: Automatically extracts and remembers user facts (like name, farm details) as they chat to personalize future advice.
- **Real-time Tools**:
  - `rag_search`: Semantic search across domain-specific agricultural vector collections.
  - `bighaat_search`: Search India's largest agri-platform for product prices, links, and blog advice.
  - `get_weather`: Current weather and 3-day forecast (supports coordinates OR city/village names).
  - `geocode_location`: Resolving village/city names to exact coordinates.
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

# Databases
DATABASE_URL=postgresql://user:pass@localhost:5432/agritech
REDIS_URL=redis://localhost:6379/0

# Storage & Paths
QDRANT_PATH=./db_storage/qdrant
RETRIEVAL_MODE=rag
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

### Start the Frontend (Streamlit)
```bash
streamlit run streamlit_app.py
```

---

## 🌾 Usage Tips
- **Hinglish/Hindi**: The agent prefers Hindi (Devanagari) but understands Hinglish and English.
- **Location Context**: You can ask "मेरठ का मौसम कैसा है?" and the agent will automatically resolve the location for the weather forecast.
- **Product Search**: Ask "खाद कहाँ से खरीदूँ?" or "मक्के के बीज की कीमत क्या है?" to trigger the BigHaat tool.
