"""
services/pageindex.py  –  PageIndex retrieval provider (stub).

This is an optional retrieval mode (RETRIEVAL_MODE=pageindex).
The default mode is 'rag' (Qdrant). This stub allows the server to
start cleanly when pageindex data is not present.
"""
from __future__ import annotations


class PageIndexProvider:
    """
    Optional tree-based page-index retriever.
    Only active when RETRIEVAL_MODE=pageindex in .env.
    Currently a stub — returns empty results when not loaded.
    """

    def __init__(self) -> None:
        self._loaded = False
        # Attempt to load only if config points to pageindex mode
        try:
            from core.config import settings
            if settings.retrieval_mode.strip().lower() == "pageindex":
                self._try_load(settings)
        except Exception:
            pass

    def _try_load(self, settings) -> None:
        """Try to load the PageIndex tree from disk."""
        import json, os
        tree_path = settings.pageindex_tree_path
        if not os.path.exists(tree_path):
            print(f"[PageIndex] Tree file not found: {tree_path}")
            return
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                self._tree = json.load(f)
            self._pdf_path = settings.pageindex_pdf_path
            self._max_nodes = settings.pageindex_max_nodes
            self._loaded = True
            print(f"[PageIndex] Loaded tree from {tree_path}")
        except Exception as e:
            print(f"[PageIndex] Failed to load: {e}")

    def is_loaded(self) -> bool:
        return self._loaded

    def search(self, query: str) -> str:
        """Return page-index context for the query (stub returns empty string)."""
        if not self._loaded:
            return ""
        # Real implementation would traverse the tree and extract PDF text.
        # Left as stub — extend here if pageindex mode is needed.
        return ""
