import argparse
import os
import sys
import time
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import settings


DEFAULT_QUERIES = [
    "How should I prepare land for maize sowing?",
    "What fertilizer dose should I apply to maize crop?",
    "My maize crop is showing disease symptoms, what should I do?",
    "How often should maize be irrigated in summer?",
]


def _extract_hits(result: Any) -> list[Any]:
    if result is None:
        return []
    if isinstance(result, list):
        return result

    points = getattr(result, "points", None)
    if isinstance(points, list):
        return points

    inner_result = getattr(result, "result", None)
    if isinstance(inner_result, list):
        return inner_result

    inner_points = getattr(inner_result, "points", None)
    if isinstance(inner_points, list):
        return inner_points

    return []


def _search_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
) -> tuple[list[Any], float]:
    start = time.perf_counter()
    hits: list[Any] = []

    if hasattr(client, "search"):
        try:
            result = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            hits = _extract_hits(result)
        except Exception:
            hits = []

    if not hits and hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = _extract_hits(result)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return hits, elapsed_ms


def _preview_text(hit: Any) -> str:
    payload = getattr(hit, "payload", None) or {}
    text = str(payload.get("page_content", "")).strip()
    text = " ".join(text.split())
    return text[:140] + ("..." if len(text) > 140 else "")


def benchmark_queries(queries: list[str], top_k: int) -> None:
    client = QdrantClient(path=settings.qdrant_path)
    encoder = SentenceTransformer(settings.sentence_transformer_model)
    configured_collections = settings.resolved_qdrant_collections
    collections = [
        name for name in configured_collections
        if client.collection_exists(name)
    ]

    print(f"Qdrant path: {settings.qdrant_path}")
    print(f"Embedding model: {settings.sentence_transformer_model}")
    print(f"Configured collections: {', '.join(configured_collections)}")
    print(f"Existing collections: {', '.join(collections) if collections else '[none found]'}")

    if not collections:
        print("No configured Qdrant collections were found on disk.")
        return

    grand_total_ms = 0.0

    for idx, query in enumerate(queries, start=1):
        print("\n" + "=" * 90)
        print(f"Query {idx}: {query}")

        embed_start = time.perf_counter()
        query_vector = encoder.encode(query, normalize_embeddings=True).tolist()
        embedding_ms = (time.perf_counter() - embed_start) * 1000.0

        query_total_start = time.perf_counter()
        total_hits = 0

        print(f"Embedding time: {embedding_ms:.2f} ms")
        for collection_name in collections:
            hits, elapsed_ms = _search_collection(
                client,
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
            total_hits += len(hits)
            print(f"- {collection_name}: {elapsed_ms:.2f} ms | hits={len(hits)}")

            if hits:
                top_hit = hits[0]
                score = float(getattr(top_hit, "score", 0.0))
                print(f"  top score: {score:.4f}")
                print(f"  preview: {_preview_text(top_hit)}")

        query_total_ms = (time.perf_counter() - query_total_start) * 1000.0
        grand_total_ms += query_total_ms + embedding_ms
        print(f"Total retrieval time: {query_total_ms:.2f} ms | total hits={total_hits}")

    print("\n" + "=" * 90)
    print(f"Completed {len(queries)} synchronous queries.")
    print(f"Grand total time including embeddings: {grand_total_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark synchronous Qdrant retrieval for 4 queries.")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Query to benchmark. Pass up to 4 times. Defaults to 4 built-in sample queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top results to request from each collection.",
    )
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    queries = queries[:4]
    if len(queries) < 4 and not args.queries:
        queries = DEFAULT_QUERIES

    benchmark_queries(queries, top_k=args.top_k)


if __name__ == "__main__":
    main()
