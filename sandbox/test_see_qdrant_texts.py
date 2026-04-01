import argparse
import os
import sys
from typing import Any

from qdrant_client import QdrantClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import settings


def _get_text_from_payload(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""

    for key in ("page_content", "text", "search_text", "recommendation", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _get_source_from_payload(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("collection", "source", "filename", "subtopic", "crop_stage"):
            value = metadata.get(key)
            if value:
                return str(value)

    for key in ("source", "category"):
        value = payload.get(key)
        if value:
            return str(value)

    return ""


def print_collection_points(client: QdrantClient, collection_name: str, limit: int) -> None:
    if not client.collection_exists(collection_name):
        print(f"[SKIP] Collection not found: {collection_name}")
        return

    points, next_offset = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    print(f"\n=== Collection: {collection_name} | Points fetched: {len(points)} ===")
    if next_offset is not None:
        print(f"Next offset: {next_offset}")

    for point in points:
        payload = point.payload if isinstance(point.payload, dict) else {}
        text = _get_text_from_payload(payload)
        source = _get_source_from_payload(payload)

        print(f"ID: {point.id}")
        if source:
            print(f"Source: {source}")
        if "metadata" in payload:
            print(f"Metadata: {payload.get('metadata')}")
        print("Text:")
        print(text or "[no text field found in payload]")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect payload text stored in local Qdrant collections.")
    parser.add_argument(
        "--collection",
        action="append",
        dest="collections",
        help="Collection name to inspect. Can be passed multiple times. Defaults to all configured collections.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of points to fetch per collection.",
    )
    args = parser.parse_args()

    client = QdrantClient(path=settings.qdrant_path)
    collections = args.collections or list(dict.fromkeys(settings.resolved_qdrant_collections + [settings.maize_faq_collection_name]))

    print(f"Qdrant path: {settings.qdrant_path}")
    for collection_name in collections:
        print_collection_points(client, collection_name, args.limit)


if __name__ == "__main__":
    main()

_COLLECTION_MAP = {
    "pop_query":        "spring_corn_pop_db",
    "fertilizer_query": "spring_corn_fertilizers_db",
    "pest_query":       "spring_corn_pest_and_diseases_db",
    "production_query": "maize_production_manual_db",
}
