import os
import json
from tools import execute_rag_search, execute_get_weather, execute_web_search
from main import get_qdrant_client

def test_weather():
    print("Testing get_weather...")
    res = execute_get_weather(28.7, 77.1)
    print("Weather Result:", json.dumps(res, indent=2))
    assert "error" not in res, f"Weather failed: {res.get('error')}"

def test_web_search():
    print("Testing web_search...")
    res = execute_web_search("Agriculture in India")
    print("Web Search Result:", json.dumps(res, indent=2))
    assert "error" not in res, f"Web search failed: {res.get('error')}"

def test_rag_search():
    print("Testing rag_search...")
    client = get_qdrant_client()
    res = execute_rag_search("fertilizers for corn", 2, client)
    print("RAG Result:", json.dumps(res, indent=2))
    assert "error" not in res, f"RAG search failed: {res.get('error')}"

if __name__ == "__main__":
    test_weather()
    test_web_search()
    test_rag_search()
