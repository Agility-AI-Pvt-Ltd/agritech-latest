"""
tools.py  –  All tools the agent can call.

Each tool is a plain Python function returning a dict.
The TOOLS list contains LangChain-style tool schemas for the LLM.
"""
from __future__ import annotations

import json
import os
import traceback
import uuid
import requests
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from llm_logging import log_llm_call


# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (passed to LLM as bind_tools / function_calling)
# ──────────────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Search the agricultural knowledge base (Qdrant vector store) for "
            "information about crops, pests, diseases, fertilizers, and farming "
            "best practices. Use this first for agri-related questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the knowledge base.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 4).",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "geocode_location",
        "description": (
            "Get the exact latitude and longitude for a given city, village, "
            "or address. Call this tool FIRST to resolve a user's address "
            "into coordinates before calling get_weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "The name of the city, village, or address (e.g., 'Sitapur, Uttar Pradesh').",
                },
            },
            "required": ["address"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Fetch current weather and 3-day forecast for a location using "
            "latitude and longitude. CRITICAL: You MUST use exact coordinates "
            "returned by the geocode_location tool or known from the user's profile. "
            "DO NOT GUESS OR APPROXIMATE coordinates yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location.",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location.",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo for recent or general information "
            "not found in the knowledge base. Use as a fallback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": (
            "Return the current date, day of the week, time, and farming-season "
            "context (e.g. 'Zaid / Spring season'). Use this when the user asks "
            "about today's date, day, time, or what agricultural season it is."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timezone_offset_hours": {
                    "type": "number",
                    "description": "UTC offset in hours (default 5.5 for IST).",
                    "default": 5.5,
                },
            },
            "required": [],
        },
    },
]


def _json_safe(value: Any) -> Any:
    """Best-effort conversion for JSON logging."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_tool_log(
    tool_name: str,
    params: Dict[str, Any],
    result: Dict[str, Any],
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    call_id: str | None = None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    """Write one JSON file per tool call under ./logs/tool_calls."""
    try:
        project_root = os.path.abspath(os.path.dirname(__file__))
        logs_dir = os.path.join(project_root, "logs", "tool_calls")
        os.makedirs(logs_dir, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        suffix = uuid.uuid4().hex[:8]
        file_name = f"{ts}_{tool_name}_{suffix}.json"
        file_path = os.path.join(logs_dir, file_name)

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "status": status,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "call_id": call_id,
            "input": _json_safe(params),
            "output": _json_safe(result),
            "error": error,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as log_err:
        print(f"[!] Tool logging failed for {tool_name}: {log_err}")


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

def execute_get_current_datetime(timezone_offset_hours: float = 5.5) -> Dict[str, Any]:
    """Return current date/time in the given timezone (default IST)."""
    try:
        tz = timezone(timedelta(hours=timezone_offset_hours))
        now = datetime.now(tz)
        month = now.month
        # Simple agricultural season mapper for India
        if month in (2, 3, 4, 5):      season = "Zaid / Spring (Zaid Maize season)"
        elif month in (6, 7, 8, 9, 10): season = "Kharif (Monsoon season)"
        else:                            season = "Rabi (Winter season)"
        return {
            "date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": f"UTC{'+' if timezone_offset_hours >= 0 else ''}{timezone_offset_hours}",
            "farming_season": season,
            "datetime_iso": now.isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}

def generate_sub_queries(
    query: str,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    from chat_llm import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    import json
    
    try:
        llm = get_llm()
        system_prompt = """You are an agricultural assistant. Given a user query about farming, and optionally recent conversation context, generate 4 specific sub-questions to search our vector database.
The databases contain English documents. Therefore, YOU MUST TRANSLATE THE QUERIES INTO ENGLISH before returning them, even if the user query is in Hindi or another language.

The databases are:
1. pop_query: Deals with Package of Practices (POP) like sowing, harvesting, seeds, spacing, yield, and water management.
2. fertilizer_query: Deals with fertilizers, nutrient computation, soil health, and application methods.
3. pest_query: Deals with pests, diseases, weeds, and their management strategies.
4. production_query: Deals with overall maize production, cultivation guidelines, crop management, and agronomy.

Return ONLY a valid JSON object with these exactly 4 keys ("pop_query", "fertilizer_query", "pest_query", "production_query"), mapping to the strictly ENGLISH generated questions. Do not include markdown formatting or extra text."""
        
        context_str = ""
        if chat_history:
            context_str = "Recent Chat History:\n" + "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history]) + "\n\n"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context_str}Current Query to expand: {query}")
        ]
        
        msg = llm.invoke(messages)
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.generate_sub_queries",
            request={
                "query": query,
                "chat_history_count": len(chat_history or []),
            },
            response={
                "has_content": bool(getattr(msg, "content", "")),
            },
        )
        
        content = msg.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        return json.loads(content)
    except Exception as exc:
        log_llm_call(
            conversation_id=conversation_id,
            user_id=user_id,
            source="tool.generate_sub_queries.error",
            request={
                "query": query,
                "chat_history_count": len(chat_history or []),
            },
            error=str(exc),
        )
        # Fallback to original query
        return {
            "pop_query": query,
            "fertilizer_query": query,
            "pest_query": query,
            "production_query": query
        }

def execute_rag_search(
    query: str,
    top_k: int = 2,
    qdrant_client=None,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> Dict[str, Any]:
    """Run similarity search across all 4 collections using query regeneration."""
    if qdrant_client is None:
        return {"error": "Qdrant client not initialized", "chunks": []}

    try:
        from chat_llm import get_embedding_model
        encoder = get_embedding_model()
        
        sub_queries = generate_sub_queries(
            query,
            chat_history,
            conversation_id=conversation_id,
            user_id=user_id,
        )
        collections = {
            "pop_query": "spring_corn_pop_db",
            "fertilizer_query": "spring_corn_fertilizers_db",
            "pest_query": "spring_corn_pest_and_diseases_db",
            "production_query": "maize_production_manual_db"
        }
        
        chunks = []
        for key, coll_name in collections.items():
            sub_q = sub_queries.get(key, query)
            query_vector = encoder.encode(sub_q).tolist()
            
            try:
                response = qdrant_client.query_points(
                    collection_name=coll_name,
                    query=query_vector,
                    limit=top_k,
                )
                print(response)
                for hit in response.points:
                    payload = hit.payload or {}
                    chunks.append({
                        "collection": coll_name,
                        "sub_query": sub_q,
                        "score": round(float(hit.score), 4),
                        "content": payload.get("page_content", ""),
                        "metadata": payload.get("metadata", {}),
                    })
            except Exception as coll_e:
                print(f"[!] RAG Search Warning: {coll_e}")

        return {"query": query, "sub_queries": sub_queries, "chunks": chunks}

    except Exception as e:
        return {"error": str(e), "chunks": []}


def execute_get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch weather from Open-Meteo (no API key required)."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}"
            f"&current_weather=true"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_max&timezone=auto"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        cw = data.get("current_weather", {})
        daily = data.get("daily", {})

        forecast = []
        times = daily.get("time", [])
        for i in range(1, 4):
            if i < len(times):
                forecast.append({
                    "date": times[i],
                    "temp_max": daily.get("temperature_2m_max", [None] * 5)[i],
                    "temp_min": daily.get("temperature_2m_min", [None] * 5)[i],
                    "rain_mm": daily.get("precipitation_sum", [0] * 5)[i],
                    "humidity": daily.get("relative_humidity_2m_max", [None] * 5)[i],
                })

        return {
            "current": {
                "temperature": cw.get("temperature"),
                "wind_speed": cw.get("windspeed"),
                "weather_code": cw.get("weathercode"),
            },
            "forecast_3day": forecast,
        }
    except Exception as e:
        return {"error": str(e)}


def execute_web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """DuckDuckGo instant answer search (no key required)."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()

        results: List[Dict] = []
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            results.append({
                "title": data.get("Heading", "Result"),
                "snippet": abstract,
                "url": data.get("AbstractURL", ""),
            })

        for item in data.get("RelatedTopics") or []:
            if len(results) >= max_results:
                break
            if isinstance(item, dict) and "Text" in item:
                results.append({
                    "title": "Related",
                    "snippet": item["Text"],
                    "url": item.get("FirstURL", ""),
                })

        return {"query": query, "results": results[:max_results]}
    except Exception as e:
        return {"error": str(e), "results": []}


def execute_geocode_location(address: str) -> Dict[str, Any]:
    """Resolve an address to latitude and longitude using Geopy."""
    from geopy.geocoders import Nominatim
    try:
        geolocator = Nominatim(user_agent="agritech_ai_assistant")
        location = geolocator.geocode(address)
        if location:
            return {
                "address": address,
                "resolved_address": location.address,
                "latitude": location.latitude,
                "longitude": location.longitude
            }
        return {"error": f"Could not resolve the address: '{address}'. Please ask the user for a more specific city or village name."}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def dispatch_tool(
    tool_name: str,
    params: Dict[str, Any],
    qdrant_client=None,
    chat_history: list | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    call_id: str | None = None,
) -> Dict[str, Any]:
    """Route a tool call to its implementation."""
    result: Dict[str, Any]
    status = "ok"
    err_msg: str | None = None

    try:
        if tool_name == "rag_search":
            result = execute_rag_search(
                qdrant_client=qdrant_client,
                chat_history=chat_history,
                conversation_id=conversation_id,
                user_id=user_id,
                **params,
            )
        elif tool_name == "get_weather":
            result = execute_get_weather(**params)
        elif tool_name == "geocode_location":
            result = execute_geocode_location(**params)
        elif tool_name == "web_search":
            result = execute_web_search(**params)
        elif tool_name == "get_current_datetime":
            result = execute_get_current_datetime(**params)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        status = "error"
        err_msg = str(exc)
        result = {
            "error": err_msg,
            "tool": tool_name,
            "traceback": traceback.format_exc(),
        }

    _write_tool_log(
        tool_name,
        params,
        result,
        conversation_id=conversation_id,
        user_id=user_id,
        call_id=call_id,
        status=status,
        error=err_msg,
    )
    return result
