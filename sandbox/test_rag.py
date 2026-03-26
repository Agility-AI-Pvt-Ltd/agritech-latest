import httpx
from pipeline.graph import run
from pipeline.llm_factory import get_llm

llm = get_llm()

# Mock Qdrant Client to avoid file locks with the running Streamlit app
class MockResponse:
    def __init__(self):
        self.points = []

class MockQdrant:
    def query_points(self, *args, **kwargs):
        print(f"    [QDRANT MOCK SEARCH] -> Collection: {kwargs.get('collection_name')} | Query Vector Len: {len(kwargs.get('query', []))}")
        return MockResponse()

conv_id = "conv_test_rag_01"
u_id = "test_user_krishna"

query = "mujhe kitne matra me urea khad dalna chhaiye ?"
print(f"\n[USER]: {query}\n" + "-"*40)

state = run(
    query=query,
    llm=llm,
    qdrant_client=MockQdrant(),  # Inject mock to test retrieval looping
    conversation_id=conv_id,
    user_id=u_id
)

print("\n--- TEST RESULTS ---")
print(f"Total Loops: {state.get('loop_count', 0)}")
print(f"Needs More Info (Last loop): {state.get('needs_more_info', False)}")

tool_results = state.get("tool_result")
print(f"Last Tool Result Sub-queries: {tool_results.get('sub_queries', {}) if tool_results else 'None'}")

tools = state.get("tool_calls", [])
print(f"Total Tool Executions: {len(tools)}")
for idx, t in enumerate(tools):
    print(f"  {idx+1}. Tool: {t['tool']} | Params: {t['params']}")

print("\n[BOT]:", state.get("final_response"))
