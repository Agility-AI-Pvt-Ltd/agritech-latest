"""
test_db_and_datetime.py - Integration test for:
  1. get_current_datetime tool
  2. PostgreSQL state persistence across 3 simulated "stateless REST" calls
  3. Location memory loaded back from DB on 2nd and 3rd call
"""
import uuid
from graph import run
from main import get_llm, get_qdrant_client
import db

def separator(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def run_test():
    llm = get_llm()
    qdrant_client = get_qdrant_client()

    # --- Test 1: get_current_datetime tool ---
    separator("TEST 1: Datetime Tool")
    result = run(
        query="What is today's date, day, and which farming season is it right now?",
        llm=llm,
        qdrant_client=qdrant_client
    )
    tools_used = [tc["tool"] for tc in result.get("tool_calls", [])]
    print(f"Tools Used: {tools_used}")
    for tc in result.get("tool_calls", []):
        if tc["tool"] == "get_current_datetime":
            print(f"Datetime Result: {tc['result']}")
    print(f"\nResponse: {result.get('final_response')}")

    # --- Test 2: PostgreSQL state persistence ---
    separator("TEST 2: PostgreSQL State Persistence")

    # Initialize DB table
    db.init_db()

    # Generate a unique conversation ID (simulates a user session)
    conv_id = f"test-session-{uuid.uuid4().hex[:8]}"
    print(f"Conversation ID: {conv_id}")

    # ── Simulated REST Call 1: User gives location ──────────────────────────
    print("\n[REST Call 1] User says where they are...")
    r1 = run(
        query="My name is Raman and I am farming in Varanasi, Uttar Pradesh.",
        llm=llm,
        qdrant_client=qdrant_client,
        conversation_id=conv_id
    )
    print(f"Response: {r1.get('final_response', '')[:200]}")

    # ── Simulated REST Call 2: NEW request — no chat_history passed ─────────
    print("\n[REST Call 2] New stateless request (no chat_history in call)...")
    r2 = run(
        query="What is the current weather at my location?",
        llm=llm,
        qdrant_client=qdrant_client,
        conversation_id=conv_id   # same conv_id → state auto-loaded from DB
    )
    tools_r2 = [tc["tool"] for tc in r2.get("tool_calls", [])]
    print(f"Tools Used: {tools_r2}")
    for tc in r2.get("tool_calls", []):
        if tc["tool"] == "get_weather":
            print(f"  -> Used: lat={tc['params'].get('latitude')}, lon={tc['params'].get('longitude')}")
    print(f"Response: {r2.get('final_response', '')[:300]}")

    # ── Simulated REST Call 3: Reference earlier context ────────────────────
    print("\n[REST Call 3] Asking about earlier topic...")
    r3 = run(
        query="Based on the weather you just told me, should I irrigate today?",
        llm=llm,
        qdrant_client=qdrant_client,
        conversation_id=conv_id
    )
    print(f"Response: {r3.get('final_response', '')[:400]}")

    # Cleanup test record
    db.delete_state(conv_id)
    print(f"\n[DB] Cleaned up test conversation {conv_id}")

if __name__ == "__main__":
    run_test()
