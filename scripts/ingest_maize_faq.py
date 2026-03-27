"""
Ingest maize FAQ rag_entries from data/maize_knowledge_tree.json into Qdrant.

Embeds the `search_text` field exactly as described in the JSON metadata.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from core.config import settings
from pipeline.llm_factory import get_embedding_model

BATCH_SIZE = 128


def _load_entries() -> List[Dict[str, Any]]:
    with open(settings.maize_faq_tree_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("rag_entries") or [])


def _create_collection(client: QdrantClient, vector_size: int) -> None:
    collection_name = settings.maize_faq_collection_name
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def _upload_batches(client: QdrantClient, entries: List[Dict[str, Any]]) -> None:
    encoder = get_embedding_model()
    total = len(entries)
    for start in range(0, total, BATCH_SIZE):
        batch = entries[start:start + BATCH_SIZE]
        texts = [entry["search_text"] for entry in batch]
        vectors = encoder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        points: List[qdrant_models.PointStruct] = []
        for idx, (entry, vector) in enumerate(zip(batch, vectors), start=start):
            metadata = {
                "entry_id": entry.get("id"),
                "crop_stage": entry.get("crop_stage"),
                "subtopic": entry.get("subtopic"),
                "category": entry.get("category"),
            }
            points.append(
                qdrant_models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload={
                        "question": entry.get("question", ""),
                        "recommendation": entry.get("recommendation", ""),
                        "category": entry.get("category", ""),
                        "search_text": entry.get("search_text", ""),
                        "metadata": metadata,
                    },
                )
            )

        client.upsert(
            collection_name=settings.maize_faq_collection_name,
            points=points,
            wait=True,
        )


def main() -> None:
    entries = _load_entries()
    if not entries:
        print("[!] No rag_entries found in maize knowledge tree.")
        return

    client = QdrantClient(path=settings.qdrant_path)
    encoder = get_embedding_model()
    sample_vector = encoder.encode("vector-size-check", normalize_embeddings=True)

    _create_collection(client, vector_size=len(sample_vector))
    _upload_batches(client, entries)
    print(
        f"[✓] Ingested {len(entries)} maize FAQ entries into "
        f"'{settings.maize_faq_collection_name}'."
    )


if __name__ == "__main__":
    main()
