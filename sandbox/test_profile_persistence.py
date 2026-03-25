"""
test_profile_persistence.py — Integration test for:
  1. User mentions name, farm size, crops, soil type across 3 REST-like calls
  2. user_profiles table is updated automatically after each turn
  3. On 4th call (no chat_history passed), profile is loaded and referenced in response
"""
import uuid
from graph import run
from main import get_llm, get_qdrant_client
import db

def sep(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

def run_test():
    llm = get_llm()
    qdrant = get_qdrant_client()

    db.init_db()

    user_id = f"user-{uuid.uuid4().hex[:6]}"
    conv_id = f"conv-{uuid.uuid4().hex[:6]}"
    print(f"user_id={user_id}   conv_id={conv_id}")

    turns = [
        "Hi! My name is Suresh and I farm in Gorakhpur, Uttar Pradesh.",
        "I have 8 acres of land with sandy loam soil and I mainly grow Spring Corn and Wheat.",
        "I also rely on drip irrigation and I have a borewell on site.",
        # Simulated new stateless REST call - no chat_history passed, no location
        "What fertilizer schedule is best for my crops?",
    ]

    for i, q in enumerate(turns, 1):
        sep(f"Turn {i}: {q[:60]}")
        result = run(
            query=q,
            llm=llm,
            qdrant_client=qdrant,
            conversation_id=conv_id,
            user_id=user_id,
        )
        print(f"Tools: {[tc['tool'] for tc in result.get('tool_calls', [])]}")
        print(f"Response: {result.get('final_response', '')[:300]}")

        profile = db.load_user_profile(user_id)
        print(f"\n[DB] Profile after turn {i}:")
        if profile:
            for k, v in profile.items():
                if k not in ("user_id",) and v:
                    print(f"    {k}: {v}")
        else:
            print("    (no profile yet)")

    # Cleanup
    db.delete_state(conv_id)
    try:
        conn = db._get_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_profiles WHERE user_id = %s;", (user_id,))
        conn.close()
    except Exception as e:
        print(f"Cleanup error: {e}")
    print(f"\n[DONE] Cleaned up test data for user={user_id}, conv={conv_id}")

if __name__ == "__main__":
    run_test()
