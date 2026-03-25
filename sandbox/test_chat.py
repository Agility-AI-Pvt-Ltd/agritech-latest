import httpx
import sys

# Change to false to test internal run() graph directly instead of API
TEST_API = False

if not TEST_API:
    from graph import run
    from main import get_llm, get_qdrant_client
    llm = get_llm()
    qdrant = get_qdrant_client()

conv_id = "conv_test_final_01"
u_id = "test_user_krishna"

prompts = [
    "hii",
    "kaise ho tum?",
    "aaj ka masuam bta do",
    "Sitapur uttar pradesh",
    "mera naam Krishna kumar hai",
    "mai kha rhta hun"
]

for p in prompts:
    print(f"\n[{u_id}]: {p}")
    if TEST_API:
        resp = httpx.post(
            "http://127.0.0.1:8000/api/chat",
            json={
                "message": p,
                "conversation_id": conv_id,
                "user_id": u_id
            },
            timeout=120.0
        )
        print(f"[Bot]: {resp.json().get('response')}")
    else:
        state = run(
            query=p,
            llm=llm,
            qdrant_client=qdrant,
            conversation_id=conv_id,
            user_id=u_id
        )
        print(f"[Bot]: {state.get('final_response')}")
