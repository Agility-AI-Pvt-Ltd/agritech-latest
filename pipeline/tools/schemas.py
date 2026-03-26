"""
pipeline/tools/schemas.py  –  Tool JSON schemas for LLM function calling.

Each entry follows the OpenAI/LangChain tool schema format.
Add new tools by appending to this list.
"""

TOOLS = [
    {
        "name": "bighaat_search",
        "description": (
            "Search BigHaat (India's largest agri platform) for: "
            "(1) agricultural products like seeds, pesticides, fungicides, fertilizers with prices, "
            "(2) Kisan Vedika blog articles on crop management, pest control, schemes. "
            "Use this when farmer asks about buying inputs OR needs practical crop advice with product recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term e.g. 'maize fall armyworm insecticide' or 'wheat rust fungicide'",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["products", "blogs", "both"],
                    "description": "What to search: 'products' for buying, 'blogs' for advice, 'both' for all.",
                    "default": "both",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results per category (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
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
            "Fetch current weather and 3-day forecast using exact latitude and longitude only. "
            "If coordinates are unknown, first ask the user for their address or village/city name, "
            "then call geocode_location, and only then call get_weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location (e.g. 26.8467).",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location (e.g. 80.9462).",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for recent or general information not found in the "
            "knowledge base. Always prioritize results from Uttar Pradesh or India."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
                "state": {
                    "type": "string",
                    "description": "Preferred state scope (default: Uttar Pradesh).",
                    "default": "Uttar Pradesh",
                },
                "country": {
                    "type": "string",
                    "description": "Preferred country scope (default: India).",
                    "default": "India",
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
