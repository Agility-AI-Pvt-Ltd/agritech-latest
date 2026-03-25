import os
from graph import run
from main import get_llm, get_qdrant_client

def test_location_memory():
    llm = get_llm()
    qdrant_client = get_qdrant_client()
    
    chat_history = []
    
    # Notice we do NOT pass user_latitude or user_longitude into `run`.
    # The agent must rely purely on the user's stated location in the conversation.
    questions = [
        "Hi, I am reaching out from Patna, Bihar.",
        "What is the current weather summary?",
        "Do I need to irrigate my spring corn crop based on this week's weather?"
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"Turn {i}: {q}")
        print(f"{'='*50}")
        
        result = run(
            query=q,
            llm=llm,
            qdrant_client=qdrant_client,
            chat_history=chat_history
        )
        
        # Print tools used
        tools_used = [tc["tool"] for tc in result.get("tool_calls", [])]
        print(f"Tools Used this turn: {tools_used}")
        for tc in result.get("tool_calls", []):
            if tc["tool"] == "get_weather":
                print(f"  -> Weather coordinates used: lat={tc['params'].get('latitude')}, lon={tc['params'].get('longitude')}")
        
        final_response = result.get("final_response", "NO_RESPONSE")
        print(f"\n[🤖 Final AI Response]\n{final_response}\n")
        
        # update chat_history for next turn
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": final_response})

if __name__ == "__main__":
    test_location_memory()
