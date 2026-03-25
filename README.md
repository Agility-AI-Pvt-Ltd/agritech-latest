# 🌾 Kisan Mitra - Smart Advisory System

Ek systematic FastAPI-based agricultural advisory system Uttar Pradesh ke farmers ke liye.
# architecture link
[Figma file](https://www.figma.com/board/kJYq8lWkBVbbDTV2zVqWoR/AgriTech-AI-Architecture_v1?node-id=0-1&t=o2ifrowuTASSFvJN-1)

## 📁 Project Structure

```
advisory ai/
├── main.py           # 🚀 FastAPI app entry point
├── router.py         # 🛣️  Saare API endpoints/routes
├── functions.py      # 🔧 Saare helper functions
├── cli.py            # 💻 CLI interface (optional)
├── app.py            # ❌ Old file (ab use na karein)
└── db_storage/       # 📚 Vector database
    └── chroma.sqlite3
```

## 🎯 File Descriptions

### **main.py**
- FastAPI application setup
- CORS middleware configuration
- Startup events (database loading)
- Server initialization
- Root endpoint

**Chalane ke liye:**
```bash
python main.py
```
Server start hoga `http://localhost:8000` par

### **router.py**
- **Saare API endpoints:**
  - `POST /api/advisory` - Custom query ke liye advisory
  - `POST /api/advisory/predefined` - Predefined questions se advisory
  - `GET /api/questions` - Saare available questions dekhen
  - `GET /api/health` - Health check
  - `GET /` - Welcome message

- Request/Response models (Pydantic)
- Input validation

### **functions.py**
- `load_db()` - Chroma database load karna
- `fetch_weather()` - Open-Meteo API se weather data
- `calculate_crop_stage()` - Days se crop stage nikalna
- `generate_advisory()` - LLM se advisory generate karna
- `get_predefined_question()` - Pre-defined questions
- Constants aur utilities

### **cli.py**
- Command line interface
- Original interactive flow maintain kiya hai
- FastAPI ke bina use kar sakte ho

## 🚀 Usage

### Option 1: FastAPI Server (Recommended)

```bash
# Install dependencies
pip install fastapi uvicorn langgraph langchain-openai langchain-community chroma-db requests

# Run server
python main.py

# API Documentation
# Browser mein jaao: http://localhost:8000/docs
```

**API Call Example:**
```bash
curl -X POST "http://localhost:8000/api/advisory" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Kya mujhe aaj Sinchai karni chahiye?",
    "sowing_date": "2025-01-15",
    "latitude": 26.8,
    "longitude": 80.9
  }'
```

**Predefined Question Example:**
```bash
curl -X POST "http://localhost:8000/api/advisory/predefined" \
  -H "Content-Type: application/json" \
  -d '{
    "choice": "1",
    "sowing_date": "2025-01-15"
  }'
```

### Option 2: CLI Interface

```bash
python cli.py
```

Interactive prompt mein:
1. Bijai ki tarikh daalein
2. Latitude/Longitude (optional)
3. Question choose karein
4. Advisory le lo! 🎉

## 📊 API Endpoints Details

### 1️⃣ Custom Advisory
```
POST /api/advisory
```
**Request:**
```json
{
  "user_query": "string",
  "sowing_date": "YYYY-MM-DD",
  "latitude": 26.8,
  "longitude": 80.9
}
```

**Response:**
```json
{
  "user_query": "aapka sawal",
  "crop_stage": "Establishment/Seedling",
  "weather": {
    "temp": 28.5,
    "humidity": 65,
    "rain_sum": 0.5
  },
  "advisory": "⚡ MUKHYA SALAH: ..."
}
```

### 2️⃣ Predefined Advisory
```
POST /api/advisory/predefined
```
**Choices:**
- `"1"` - Today/Tomorrow operations
- `"2"` - Weather risks
- `"3"` - Fertilizer/Pesticide timing
- `"4"` - Disease symptoms
- `"5"` - Common diseases
- `"6"` - Heat protection
- `"7"` - Best practices

### 3️⃣ Get Questions
```
GET /api/questions
```
Saare available questions aur unke codes

### 4️⃣ Health Check
```
GET /api/health
```
System status check karna

## 🔧 Configuration

**main.py mein change kar sakte ho:**
```python
# Server settings
host="0.0.0.0"  # Kaunse address par listen kare
port=8000       # Port number
reload=True     # Auto-reload file changes par
```

**CORS settings (Production ke liye):**
```python
allow_origins=["http://localhost:3000"]  # Specific domains
```

## ⚙️ Dependencies

```
fastapi==0.104.1
uvicorn==0.24.0
langgraph==0.0.x
langchain-openai==0.x
langchain-community==0.x
chroma-db==0.x
requests==2.31.0
pydantic==2.x
```

**Install sab:**
```bash
pip install -r requirements.txt
```

## 🗂️ Database Setup

Pehle knowledge base ingest karna padega:
```bash
python test\ advisory\ ai/ingest.py
```

Phir `db_storage/` folder mein `chroma.sqlite3` ban jayega.

## 📝 Example Workflow

```
1. python main.py             → Server start
2. Browser: localhost:8000/docs → Docs dekhen
3. Try out POST /api/advisory/predefined
   {
     "choice": "3",
     "sowing_date": "2025-01-01"
   }
4. Response mil jayega! 🎉
```

## 🛠️ Development Tips

1. **Changes karte ho to auto-reload hoga** (reload=True)
2. **Logs dekho terminal mein**
3. **API tests kar sakte ho `/docs` se**
4. **Database loading time laga sakta hai startup par**

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| Knowledge base nahi mila | `ingest.py` pehle run karo |
| OpenAI API error | API key check karo functions.py mein |
| Port 8000 busy hai | `port=8001` change karo main.py mein |
| CORS error | CORS middleware check karo main.py mein |

## 📚 Next Steps

- [ ] Database se more farming info add karo
- [ ] Multiple crops support add karo
- [ ] User authentication add karo
- [ ] Database integration (MongoDB/PostgreSQL)
- [ ] Frontend bana lo (React/Vue)
- [ ] Mobile app bana lo

---

**Kisan Mitra - Farming Ka Digital Guide! 🌾💚**
