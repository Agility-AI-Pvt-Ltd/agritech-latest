from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document


class PageIndexProvider:
    """Optional tree-based PageIndex retriever."""

    def __init__(self) -> None:
        self._loaded = False
        self._docs: List[Document] = []
        try:
            from core.config import settings
            if settings.retrieval_mode.strip().lower() in {"pageindex", "hybrid"}:
                self._try_load(settings)
        except Exception:
            pass

    def _try_load(self, settings) -> None:
        """Try to load the PageIndex tree from disk."""
        import json
        import os

        tree_path = settings.pageindex_tree_path
        if not os.path.exists(tree_path):
            print(f"[PageIndex] Tree file not found: {tree_path}")
            return
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                self._tree = json.load(f)
            self._pdf_path = settings.pageindex_pdf_path
            self._max_nodes = settings.pageindex_max_nodes
            self._docs = self._flatten_tree(self._tree)
            self._loaded = bool(self._docs)
            print(f"[PageIndex] Loaded {len(self._docs)} nodes from {tree_path}")
        except Exception as e:
            print(f"[PageIndex] Failed to load: {e}")

    def _flatten_tree(self, node: Any, path: str = "root") -> List[Document]:
        docs: List[Document] = []

        if isinstance(node, dict):
            text_parts: List[str] = []
            children: List[Any] = []
            for key, value in node.items():
                next_path = f"{path}.{key}"
                if isinstance(value, (dict, list)):
                    children.append((next_path, value))
                    continue
                if isinstance(value, str) and value.strip():
                    text_parts.append(f"{key}: {value.strip()}")
                elif isinstance(value, (int, float, bool)):
                    text_parts.append(f"{key}: {value}")

            if text_parts:
                docs.append(
                    Document(
                        page_content="\n".join(text_parts),
                        metadata={
                            "retrieval_source": "pageindex",
                            "pageindex_path": path,
                        },
                    )
                )

            for child_path, child in children:
                docs.extend(self._flatten_tree(child, child_path))
            return docs

        if isinstance(node, list):
            for idx, item in enumerate(node):
                docs.extend(self._flatten_tree(item, f"{path}[{idx}]"))
            return docs

        if isinstance(node, str) and node.strip():
            docs.append(
                Document(
                    page_content=node.strip(),
                    metadata={
                        "retrieval_source": "pageindex",
                        "pageindex_path": path,
                    },
                )
            )
        return docs

    def is_loaded(self) -> bool:
        return self._loaded

    def search_documents(self, query: str, k: int | None = None) -> List[Document]:
        if not self._loaded:
            return []

        from core.config import settings
        from services.bm25 import rank_documents

        return rank_documents(query, self._docs, top_k=k or settings.pageindex_max_nodes)

    def search(self, query: str) -> str:
        """Return page-index context for the query."""
        docs = self.search_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)
