import os
import sys

# Add project root to sys.path to allow importing from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from tools import execute_rag_search


def main():
    try:
        from qdrant_client import QdrantClient
        # Connecting to the real database since the Streamlit lock is removed
        qdrant = QdrantClient(path="/Users/krishnakumar/Downloads/merged/db_storage")
    except Exception as e:
        print(f"\n[!] Cannot connect to Qdrant: {e}")
        sys.exit(1)


    query = "Does urea fertilizer affect pest resistance in spring maize?"
    print(f"\n[QUERY] -> {query}")

    encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings
    
    try:
        # response = execute_rag_search(query, qdrant_client=qdrant)
        # print(response)
        hits = qdrant.query_points(
        collection_name="spring_corn_pest_and_diseases_db",
        query= encoder.encode(query).tolist(),
        limit=3
        )
        print(hits)
    except Exception as e:
        print(f"\n[!] Cannot connect to Qdrant: {e}")
        sys.exit(1)

    # collection/spring_corn_fertilizers_db
    # /collection/spring_corn_pest_and_diseases_db  
    # /collection/spring_corn_pop_db

    for hit in hits.points:
        print(hit.payload, "score:", hit.score)
  


if __name__ == "__main__":
    main()
