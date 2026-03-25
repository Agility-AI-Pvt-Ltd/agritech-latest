# 🌾 Kisan Mitra: Agentic Agritech Advisory

Kisan Mitra is a production-ready, agentic RAG (Retrieval-Augmented Generation) system designed to provide expert agricultural advisory to Indian farmers. It specialized in **Spring Corn (Zaid Maize)** cultivation in Uttar Pradesh, leveraging real-time weather data, deep agricultural knowledge bases, and persistent user profiling.

---

## 🚀 Features

- **Agentic RAG**: Powered by **LangGraph**, the agent can reason, use tools, and loop until it finds a high-quality answer for the farmer.
- **Multi-Source Knowledge**: Ingests multiple PDF manuals (Fertilizers, Pests, Production, POP) using **IBM Docling** for advanced semantic hierarchical chunking.
- **Persistent Memory**:
  - **PostgreSQL**: Long-term storage for user profiles (crops, farm size, location) and full conversation ledgers.
  - **Redis**: Ultra-fast caching of the current agent state for seamless multi-turn chat.
- **Smart Profiling**: Automatically extracts and remembers user facts (like name, farm details) as they chat.
- **Real-time Tools**:
  - `rag_search`: Semantic search across domain-specific vector collections.
  - `get_weather`: 3-day forecasts via Open-Meteo.
  - `geocode_location`: Resolving village/city names to exact coordinates.
  - `web_search`: Fallback for latest general info via DuckDuckGo.
- **Premium UI**: A sleek, dark-green Streamlit interface optimized for chat and advisory.

---

## 🛠️ Tech Stack

- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Orchestration**: LangGraph & LangChain
- **API Framework**: FastAPI
- **Vector Database**: Qdrant (Local storage)
- **Primary Database**: PostgreSQL (State & Profiles)
- **Cache**: Redis
- **Document Parsing**: IBM Docling
- **UI**: Streamlit

---

## 📁 Project Structure

```text
├── agent.py            # Main LangGraph agent node & logic
├── graph.py            # LangGraph workflow definition & persistence
├── tools.py            # Tool implementations (RAG, Weather, etc.)
├── db.py               # PostgreSQL + Redis storage layers
├── main.py             # FastAPI entry point & lifecycle
├── streamlit_app.py    # Streamlit Chat UI
├── services/           # Business logic (Advisory, Weather, VectorStore)
├── scripts/            # Ingestion & maintenance scripts
├── models/             # SQLAlchemy ORM models
├── api/                # FastAPI routes and dependencies
└── core/               # Configuration and global settings
```

---

## ⚙️ Setup & Installation

### 1. Requirements
Ensure you have **Python 3.11+**, **PostgreSQL**, and **Redis** installed.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key

# Databases
DATABASE_URL=postgresql://user:pass@localhost:5432/agritech
REDIS_URL=redis://localhost:6379/0

# Storage
QDRANT_PATH=./db_storage/qdrant
RETRIEVAL_MODE=rag
```

### 3. Install Dependencies
```bash
uv pip install -r requirements.txt
# OR
pip install -r requirements.txt
```

### 4. Knowledge Ingestion
Place your agricultural PDFs in the `data/` folder and run the ingestion pipeline:
```bash
python scripts/ingestion_docling.py
```

---

## 🏃 Running the Application

### Start the Backend (FastAPI)
```bash
python main.py
```
The API will be available at `http://localhost:8000`. You can view the interactive docs at `/docs`.

### Start the Frontend (Streamlit)
In a new terminal:
```bash
streamlit run streamlit_app.py
```

---

## 📧 Usage
- **Chat**: Simply type your query in Hindi or English.
- **Profile**: The agent "learns" about you (e.g., your name, location) and stores it in the sidebar-linked Profile.
- **Tools**: Watch the sidebar or logs to see the agent call RAG search or Weather tools automatically.
