import httpx
import uuid
import time
import sys

API_URL = "http://localhost:8000/api/chat"
USER_ID = f"test_memory_user_{uuid.uuid4().hex[:6]}"
CONV_ID = f"test_memory_conv_{uuid.uuid4().hex[:6]}"

# Simulate a conversation of 6 messages. 
# Window is 10 messages (5 user+assistant turns).
# By turn 6, the first turn should be summarized.

MESSAGES = [
    "Hi, my name is Bob and my favorite tractor color is neon green. I farm in Punjab.", # Turn 1
    "What is the weather usually like in Punjab during summer?", # Turn 2
    "I'm thinking of planting wheat next season.", # Turn 3
    "Should I use urea or DAP for the wheat?", # Turn 4
    "How much water does wheat need in the first month?", # Turn 5
    "By the way, do you remember my name and my favorite tractor color from our first message?", # Turn 6
]

def main():
    print(f"--- Starting Memory Test ---")
    print(f"User ID: {USER_ID}")
    print(f"Conv ID: {CONV_ID}\n")

    with httpx.Client(timeout=60.0) as client:
        for i, msg in enumerate(MESSAGES, 1):
            print(f"Turn {i} - User: {msg}")
            
            payload = {
                "user_id": USER_ID,
                "conversation_id": CONV_ID,
                "query": msg
            }
            
            try:
                resp = client.post(API_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                print(f"Turn {i} - Agent: {data.get('response', '')}\n")
            except Exception as e:
                print(f"Error on Turn {i}: {e}")
                if isinstance(e, httpx.HTTPError) and e.response:
                    print(e.response.text)
                sys.exit(1)

            time.sleep(1) # small pause between requests

    print("--- Test Complete ---")

if __name__ == "__main__":
    main()
