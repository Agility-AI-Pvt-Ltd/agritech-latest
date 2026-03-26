import os
import json
from pipeline.graph import run
from pipeline.llm_factory import get_llm, get_qdrant_client

def test_rag_with_context():
    llm = get_llm()
    qdrant_client = get_qdrant_client()
    
    chat_history = []
    
    questions = [
        "I am planting spring corn in my field. What is the recommended plant population?",
        "What fertilizers should I use for that crop?",
        "Are there any specific pests I should watch out for now?",
        "When is the best time to harvest?",
        "How do I manage weeds during the early growing stages?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"Turn {i}: {q}")
        print(f"{'='*50}")
        
        result = run(
            query=q,
            llm=llm,
            qdrant_client=qdrant_client,
            chat_history=chat_history,
            user_latitude=26.8,
            user_longitude=80.9
        )
        
        # Print the RAG sub_queries generated in this turn
        for tc in result.get("tool_calls", []):
            if tc["tool"] == "rag_search":
                sq = tc["result"].get("sub_queries", {})
                print("\n[🎯 Query Regeneration Output]")
                print(f"  - POP Query:        {sq.get('pop_query')}")
                print(f"  - Fertilizer Query: {sq.get('fertilizer_query')}")
                print(f"  - Pest Query:       {sq.get('pest_query')}")
        
        final_response = result.get("final_response", "NO_RESPONSE")
        print(f"\n[🤖 Final AI Response]\n{final_response}\n")
        
        # update chat_history for next turn
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    test_rag_with_context()
