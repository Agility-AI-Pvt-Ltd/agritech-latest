import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import execute_bighaat_search
import json

def test_queries():
    queries = [
        "maize seeds",         # English
        "fungicide",           # English 
        "फंगीसाइड",             # Hindi
        "corn fertilizer",      # English
    ]
    
    for q in queries:
        print(f"\n--- Testing query: {q} ---")
        try:
            res = execute_bighaat_search(q, search_type="both", max_results=3)
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error testing {q}: {e}")

if __name__ == "__main__":
    test_queries()
