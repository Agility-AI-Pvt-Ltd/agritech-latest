import os
import sys

def main():
    try:
        from qdrant_client import QdrantClient
        # Connecting to the real database since the Streamlit lock is removed
        qdrant = QdrantClient(path="/Users/krishnakumar/Downloads/merged/db_storage")
    except Exception as e:
        print(f"\n[!] Cannot connect to Qdrant copy: {e}")
        sys.exit(1)

    from pipeline.tools.rag import execute_rag_search
    query = "Does urea fertilizer affect pest resistance in spring maize?"
    print(f"\n[QUERY] -> {query}")
    
    # We pass chat_history=None and qdrant_client to our testing environment
    res = execute_rag_search(query, top_k=2, qdrant_client=qdrant)
    
    print("\n--- TRANSLATED SUB-QUERIES ENGINES ---")
    sub = res.get("sub_queries", {})
    for k, v in sub.items():
        print(f"  {k}: {v}")
        
    print("\n--- RETRIEVED CHUNKS ---")
    chunks = res.get("chunks", [])
    if not chunks:
        print("  <No results found>")
    for idx, c in enumerate(chunks):
        print(f"\n[Match {idx+1} | Score: {c['score']} | {c['collection']}]")
        print(f"  Snippet: {c['content'][:300]}...")

if __name__ == "__main__":
    main()
