import asyncio
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from db import save_state, load_state, _redis_client, init_db

def test_redis_caching():
    init_db()
    print("=== Testing Redis Dual-Persistence Architecture ===")
    
    # 1. Generate unique session
    user_id = "test_user"
    conv_id = f"test_conv_{uuid.uuid4().hex[:8]}"
    
    # 2. Construct simulated state with LangChain message objects
    mock_state = {
        "user_id": user_id,
        "chat_history": [
            {"role": "user", "content": "Hello!"},
            {"role": "agent", "content": "Hi there!"}
        ],
        "conversation_summary": "User said hello.",
        "user_location": "Varanasi, UP",
        "messages": [
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!")
        ]
    }
    
    print(f"[*] Saving state for {user_id} / {conv_id}...")
    save_state(conv_id, mock_state, user_id=user_id)
    
    # 3. Verify Redis has the key
    redis_key = f"agri:state:{conv_id}"
    print(f"[*] Checking Redis for key: {redis_key}")
    exists = _redis_client.exists(redis_key)
    if exists:
        print("[SUCCESS] Key exists in Redis!")
    else:
        print("[FAILED] Key missing from Redis!")
        return

    # 4. Load state (Should hit Redis)
    print("\n[*] Loading state...")
    loaded_state = load_state(conv_id)
    
    print("\n[*] Verifying data integrity:")
    print(f"    - chat_history count: {len(loaded_state['chat_history'])}")
    print(f"    - summary: {loaded_state['conversation_summary']}")
    
    # 5. Verify LangChain objects deserialized correctly
    messages = loaded_state.get("messages", [])
    print(f"    - messages count: {len(messages)}")
    if len(messages) == 2 and isinstance(messages[0], HumanMessage) and isinstance(messages[1], AIMessage):
        print("    - [SUCCESS] Messages successfully deserialized to LangChain objects!")
    else:
        print("    - [FAILED] Messages failed to reconstruct properly!")
        print(messages)
    
    # 6. Fallback test: Delete from Redis, load from Postgres
    print("\n[*] Simulating Cache Expiration (deleting from Redis)...")
    _redis_client.delete(redis_key)
    
    print("[*] Loading state again (Should hit Postgres)...")
    fallback_state = load_state(conv_id)
    if fallback_state:
        print("    - [SUCCESS] State successfully recovered from permanent Postgres ledger!")
        print(f"    - chat_history count: {len(fallback_state['chat_history'])}")
    else:
        print("    - [FAILED] Postgres fallback failed!")

if __name__ == "__main__":
    test_redis_caching()
