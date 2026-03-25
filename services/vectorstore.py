import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.config import settings

class VectorStoreProvider(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search the knowledge base."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if vector store is initialized."""
        pass

class QdrantVectorStore(VectorStoreProvider):
    def __init__(self):
        self._client: QdrantClient | None = None
        self._encoder: SentenceTransformer | None = None
        self._initialize()

    def _initialize(self):
        try:
            os.makedirs(settings.qdrant_path, exist_ok=True)
            self._client = QdrantClient(path=settings.qdrant_path)
            self._encoder = SentenceTransformer(
                settings.sentence_transformer_model
            )
            print("[*] Qdrant vector store initialized.")
        except Exception as e:
            self._client = None
            self._encoder = None
            print(f"[!] Error initializing Qdrant: {e}")

    def get_client(self) -> QdrantClient | None:
        """Expose initialized Qdrant client for shared reuse."""
        return self._client

    @staticmethod
    def _build_document(payload: Dict[str, Any]) -> Document:
        page_content = str(payload.get("page_content", ""))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def _extract_hits(result: Any) -> List[Any]:
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

    def _search_points(self, query_vector: List[float], k: int) -> List[Any]:
        if not self._client:
            return []

        all_hits: List[Any] = []
        collection_names = settings.resolved_qdrant_collections

        for collection_name in collection_names:
            hits: List[Any] = []

            # qdrant-client compatibility:
            # - older versions: client.search(...)
            # - newer versions: client.query_points(...)
            if hasattr(self._client, "search"):
                try:
                    result = self._client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        limit=k,
                        with_payload=True,
                        with_vectors=False,
                    )
                    hits = self._extract_hits(result)
                except Exception:
                    hits = []

            if not hits and hasattr(self._client, "query_points"):
                try:
                    result = self._client.query_points(
                        collection_name=collection_name,
                        query=query_vector,
                        limit=k,
                        with_payload=True,
                        with_vectors=False,
                    )
                    hits = self._extract_hits(result)
                except Exception:
                    hits = []

            for hit in hits:
                payload = getattr(hit, "payload", None)
                if not isinstance(payload, dict):
                    payload = {}
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata.setdefault("collection", collection_name)
                payload["metadata"] = metadata
                hit.payload = payload

            all_hits.extend(hits)

        all_hits.sort(key=lambda h: float(getattr(h, "score", 0.0)), reverse=True)
        return all_hits[:k]

    def _write_search_log(self, query: str, hits: List[Any]) -> None:
        if not settings.qdrant_log_enabled:
            return

        try:
            os.makedirs(settings.qdrant_log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = os.path.join(settings.qdrant_log_dir, f"qdrant_query_{ts}.json")

            serialized_hits: List[Dict[str, Any]] = []
            for rank, hit in enumerate(hits, start=1):
                payload = getattr(hit, "payload", None)
                payload = payload if isinstance(payload, dict) else {}
                point_id = getattr(hit, "id", "")
                score = getattr(hit, "score", 0.0)
                serialized_hits.append(
                    {
                        "rank": rank,
                        "point_id": str(point_id),
                        "score": float(score),
                        "metadata": payload.get("metadata", {}),
                        "page_content": payload.get("page_content", ""),
                    }
                )

            data = {
                "timestamp": datetime.now().isoformat(),
                "collection_names": settings.resolved_qdrant_collections,
                "query": query,
                "result_count": len(serialized_hits),
                "results": serialized_hits,
            }

            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[*] Qdrant log written: {log_path}")
        except Exception as e:
            print(f"[!] Failed to write Qdrant log: {e}")

    def search(self, query: str, k: int = 3) -> List[Document]:
        if not self._client or not self._encoder:
            return []

        if not self.is_loaded():
            return []

        query_vector = self._encoder.encode(query, normalize_embeddings=True).tolist()
        hits = self._search_points(query_vector=query_vector, k=k)

        self._write_search_log(query=query, hits=hits)
        documents: List[Document] = []
        for hit in hits:
            payload = getattr(hit, "payload", None)
            payload = payload if isinstance(payload, dict) else {}
            documents.append(self._build_document(payload))
        return documents

    def is_loaded(self) -> bool:
        if not self._client:
            return False

        for collection_name in settings.resolved_qdrant_collections:
            if not self._client.collection_exists(collection_name):
                continue

            try:
                collection_info = self._client.get_collection(collection_name)
                points_count = collection_info.points_count or 0
                if points_count > 0:
                    return True
            except Exception:
                continue

        return False
