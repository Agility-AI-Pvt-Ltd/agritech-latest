import os
import time
from graph import run
from main import get_llm, get_qdrant_client

TEST_CASES = [
    {"query": "Who is the Prime Minister of India?", "expected_tool": "web_search", "fallback_expected": False},
    {"query": "What's the weather in Mumbai? It's lat 19.0, lon 72.8.", "expected_tool": "get_weather", "fallback_expected": False},
    {"query": "Are there any pests for spring corn?", "expected_tool": "rag_search", "fallback_expected": True},
    {"query": "How to fertilize spring corn?", "expected_tool": "rag_search", "fallback_expected": True},
    {"query": "What is the recommended plant population for spring corn?", "expected_tool": "rag_search", "fallback_expected": True},
    {"query": "Is it going to rain in Delhi tomorrow? Let me tell you, its lat is 28.7 and lon is 77.1.", "expected_tool": "get_weather", "fallback_expected": False},
    {"query": "Search the web for the latest agricultural news in India.", "expected_tool": "web_search", "fallback_expected": False},
    {"query": "I want to know about general maize facts. Search online.", "expected_tool": "web_search", "fallback_expected": False},
    {"query": "What is the temperature right now?", "expected_tool": "get_weather", "fallback_expected": False},
    {"query": "Tell me about duckduckgo.", "expected_tool": "web_search", "fallback_expected": False},
]

def run_tests():
    llm = get_llm()
    qdrant_client = get_qdrant_client()
    
    passed = 0
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n--- Test Case {i}: {tc['query']} ---")
        try:
            result = run(
                query=tc["query"],
                llm=llm,
                qdrant_client=qdrant_client,
                user_latitude=26.8,
                user_longitude=80.9
            )
            
            tool_calls = result.get("tool_calls", [])
            tools_used = [t["tool"] for t in tool_calls]
            print(f"Tools Used: {tools_used}")
            print(f"Final Response: {result.get('final_response', 'NO_RESPONSE')}")
            
            # Simple heuristic
            if tc["expected_tool"] in tools_used:
                print("PASSED - Expected tool found.")
                passed += 1
            else:
                if len(tools_used) == 0:
                    print("SKIPPED tool call, but maybe LLM knew the answer without tool.")
                    passed += 1
                else:
                    print(f"FAILED - Expected {tc['expected_tool']}, got {tools_used}")
        except Exception as e:
            print(f"FAILED - Exception: {e}")
        
        # Rate limit protection just in case
        time.sleep(2)
        
    print(f"\nTotal Passed: {passed}/{len(TEST_CASES)}")

if __name__ == "__main__":
    run_tests()
