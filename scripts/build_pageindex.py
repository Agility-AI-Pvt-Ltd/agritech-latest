"""Build a PageIndex JSON tree from deployment markdown files."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import settings

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _normalize_heading(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).strip()


def _summarize(text: str, *, max_chars: int = 700) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def _sections_from_markdown(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    heading_path: List[str] = []
    buffer: List[str] = []
    sections: List[Dict[str, Any]] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        sections.append(
            {
                "heading_path": " > ".join(heading_path),
                "text": _summarize(body),
                "word_count": len(body.split()),
            }
        )

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            flush()
            level = len(match.group(1))
            heading_path = heading_path[: level - 1]
            heading_path.append(_normalize_heading(match.group(2)))
            continue
        buffer.append(line)

    flush()
    return sections


def build_pageindex_tree(markdown_dir: str | Path, *, source: str = "markdown") -> Dict[str, Any]:
    root = Path(markdown_dir)
    documents: List[Dict[str, Any]] = []

    for markdown_path in sorted(root.glob("*.md")):
        sections = _sections_from_markdown(markdown_path)
        documents.append(
            {
                "filename": markdown_path.name,
                "source_path": str(markdown_path),
                "source": source,
                "section_count": len(sections),
                "sections": sections,
            }
        )

    return {
        "index_type": "markdown_pageindex",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "markdown_dir": str(root),
        "document_count": len(documents),
        "documents": documents,
    }


def write_pageindex_tree(*, markdown_dir: str | Path, output_path: str | Path, overwrite: bool = False) -> Path:
    output = Path(output_path)
    if output.exists() and not overwrite:
        print(f"[PageIndex] Existing tree found, keeping: {output}")
        return output

    tree = build_pageindex_tree(markdown_dir)
    if not tree["documents"]:
        raise RuntimeError(f"No markdown files found in {markdown_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[PageIndex] Wrote tree: {output} ({tree['document_count']} documents)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex JSON from markdown files.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if PAGEINDEX_TREE_PATH already exists.")
    parser.add_argument("--markdown-dir", default=settings.bm25_markdown_dir)
    parser.add_argument("--output", default=settings.pageindex_tree_path)
    args = parser.parse_args()

    write_pageindex_tree(
        markdown_dir=args.markdown_dir,
        output_path=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
