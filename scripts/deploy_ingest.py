"""Deployment ingestion entrypoint.

Run after Postgres/Redis/Qdrant are up:

    uv run python scripts/deploy_ingest.py
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qdrant_client import QdrantClient

from core.config import settings


def _check_qdrant() -> None:
    client = QdrantClient(**settings.qdrant_client_kwargs)
    client.get_collections()
    print(f"[✓] Qdrant reachable: {settings.qdrant_location}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest deployment knowledge data into Qdrant.")
    parser.add_argument(
        "--include-faq",
        action="store_true",
        help="Also ingest maize FAQ entries from MAIZE_FAQ_TREE_PATH when present.",
    )
    parser.add_argument(
        "--rebuild-pageindex",
        action="store_true",
        help="Rebuild PAGEINDEX_TREE_PATH even if it already exists.",
    )
    args = parser.parse_args()

    from scripts.build_pageindex import write_pageindex_tree
    from scripts import ingest_markdown

    jobs = ingest_markdown.discover_markdown_jobs()
    if not jobs:
        raise RuntimeError(f"No markdown files found in {ingest_markdown.MARKDOWN_DIR}")

    print("[*] Deployment ingestion preflight")
    print(f"    Markdown dir: {ingest_markdown.MARKDOWN_DIR}")
    print(f"    Markdown files: {len(jobs)}")
    for job in jobs:
        print(f"    - {job['filename']} -> {job['collection']}")

    write_pageindex_tree(
        markdown_dir=settings.bm25_markdown_dir,
        output_path=settings.pageindex_tree_path,
        overwrite=args.rebuild_pageindex,
    )

    _check_qdrant()
    ingest_markdown.main()

    if args.include_faq:
        if not os.path.exists(settings.maize_faq_tree_path):
            print(f"[!] FAQ tree not found, skipping: {settings.maize_faq_tree_path}")
            return
        from scripts.ingest_maize_faq import main as ingest_faq

        ingest_faq()

    print("[✓] Deployment ingestion complete")


if __name__ == "__main__":
    main()
