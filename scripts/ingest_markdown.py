"""Markdown-based multi-manual ingestion into Qdrant."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from tqdm.auto import tqdm

BATCH_SIZE = 128
MAX_WORDS = 320
OVERLAP_WORDS = 50
MIN_WORDS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from pipeline.llm_factory import get_embedding_model

MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdowns"

KNOWN_MARKDOWN_COLLECTIONS = {
    "fertilizer-materials-and-computation.md": (
        "spring_corn_fertilizers_db",
        "fertilizer materials and computation",
    ),
    "MAIZE PRODUCTION MANUAL.md": (
        "maize_production_manual_db",
        "maize production manual",
    ),
    "Management_Pests_Diseases_Manual.md": (
        "spring_corn_pest_and_diseases_db",
        "pest and diseases of maize",
    ),
    "Spring Sweet Corn (zaid Maize) – Package Of Practices (pop) _ Uttar Pradesh.md": (
        "spring_corn_pop_db",
        "package of practices for maize",
    ),
}

IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _safe_collection_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{cleaned}_db" if cleaned else "markdown_manual_db"


def discover_markdown_jobs() -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    for markdown_path in sorted(MARKDOWN_DIR.glob("*.md")):
        collection, metadata_source = KNOWN_MARKDOWN_COLLECTIONS.get(
            markdown_path.name,
            (_safe_collection_name(markdown_path.name), markdown_path.stem),
        )
        jobs.append(
            {
                "filename": markdown_path.name,
                "collection": collection,
                "metadata_source": metadata_source,
            }
        )
    return jobs


def clean_markdown(md: str) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.DOTALL)
    md = md.replace("\r\n", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def extract_images(text: str) -> Tuple[str, List[Dict[str, str]]]:
    images: List[Dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        alt = match.group(1).strip()
        src = match.group(2).strip()
        images.append({"alt": alt, "src": src})
        return f"\nImage described: {alt}\n" if alt else "\n"

    cleaned = IMAGE_PATTERN.sub(repl, text)
    return cleaned, images


def split_by_headings(md: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current_heading_path: List[str] = []
    current_lines: List[str] = []

    for line in md.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append(
                    {
                        "heading_path": current_heading_path.copy(),
                        "content": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []

            level = len(match.group(1))
            heading = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            current_heading_path = current_heading_path[: level - 1]
            current_heading_path.append(heading)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading_path": current_heading_path.copy(),
                "content": "\n".join(current_lines).strip(),
            }
        )

    return sections


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and stripped.count("|") >= 2


def convert_table_to_sentences(table_lines: List[str]) -> Tuple[str, str]:
    raw_table = "\n".join(table_lines)
    cleaned = [line.strip() for line in table_lines if line.strip()]
    if len(cleaned) < 2:
        return raw_table, raw_table

    headers = [header.strip() for header in cleaned[0].strip("|").split("|")]
    sentences: List[str] = []

    for row in cleaned[2:]:
        cols = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        parts = [f"{header}: {cell}" for header, cell in zip(headers, cols) if cell]
        if parts:
            sentences.append(". ".join(parts) + ".")

    return raw_table, "\n".join(sentences)


def _convert_html_tables(text: str) -> Tuple[str, List[str]]:
    raw_tables: List[str] = []

    def repl(match: re.Match[str]) -> str:
        raw_table = match.group(0)
        raw_tables.append(raw_table)

        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", raw_table, flags=re.DOTALL | re.IGNORECASE)
        if not rows:
            return "\n"

        parsed_rows: List[List[str]] = []
        for row in rows:
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.DOTALL | re.IGNORECASE)
            normalized_cells = [
                re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip()
                for cell in cells
            ]
            if normalized_cells:
                parsed_rows.append(normalized_cells)

        if len(parsed_rows) < 2:
            return "\n"

        headers = parsed_rows[0]
        sentences: List[str] = []
        for row in parsed_rows[1:]:
            if len(row) != len(headers):
                continue
            parts = [f"{header}: {value}" for header, value in zip(headers, row) if value]
            if parts:
                sentences.append(". ".join(parts) + ".")
        return "\n" + "\n".join(sentences) + "\n"

    updated_text = re.sub(
        r"<table.*?>.*?</table>",
        repl,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return updated_text, raw_tables


def transform_tables(text: str) -> Tuple[str, List[str]]:
    text, html_tables = _convert_html_tables(text)

    lines = text.splitlines()
    result_lines: List[str] = []
    raw_tables = list(html_tables)
    i = 0

    while i < len(lines):
        if is_table_line(lines[i]):
            table_block: List[str] = []
            while i < len(lines) and is_table_line(lines[i]):
                table_block.append(lines[i])
                i += 1

            raw_table, sentence_version = convert_table_to_sentences(table_block)
            raw_tables.append(raw_table)
            if sentence_version.strip():
                result_lines.append(sentence_version)
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines), raw_tables


def semantic_split(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []

    for para in paragraphs:
        candidate = "\n\n".join(current + [para])
        if len(candidate.split()) <= MAX_WORDS:
            current.append(para)
            continue

        if current:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            overlap_words = chunk.split()[-OVERLAP_WORDS:]
            overlap_text = " ".join(overlap_words)
            current = [overlap_text, para] if overlap_text else [para]
            continue

        words = para.split()
        start = 0
        while start < len(words):
            end = start + MAX_WORDS
            chunks.append(" ".join(words[start:end]))
            start += max(1, MAX_WORDS - OVERLAP_WORDS)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def build_embedding_text(heading_path: List[str], chunk_text: str) -> str:
    heading = " > ".join(heading_path)
    if heading:
        return f"Heading: {heading}\n\n{chunk_text}"
    return chunk_text


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def process_markdown(md_text: str, *, filename: str, metadata_source: str) -> List[Document]:
    md_text = clean_markdown(md_text)
    sections = split_by_headings(md_text)
    final_docs: List[Document] = []
    final_index = 0

    for section_idx, section in enumerate(sections):
        heading_path = section["heading_path"]
        content = section["content"]
        if not content.strip():
            continue

        content, images = extract_images(content)
        content_for_embedding, raw_tables = transform_tables(content)
        if len(content_for_embedding.split()) < MIN_WORDS:
            continue

        semantic_chunks = semantic_split(content_for_embedding)
        for chunk_idx, chunk in enumerate(semantic_chunks):
            if len(chunk.split()) < MIN_WORDS:
                continue

            embedding_text = build_embedding_text(heading_path, chunk)
            metadata = {
                "source": metadata_source,
                "document_type": "agricultural_manual_markdown",
                "filename": filename,
                "section_index": section_idx,
                "chunk_index": chunk_idx,
                "final_chunk_index": final_index,
                "heading_path": " > ".join(heading_path),
                "images_count": len(images),
                "tables_count": len(raw_tables),
            }

            payload_text = embedding_text.strip()
            final_docs.append(
                Document(
                    page_content=payload_text,
                    metadata={
                        **metadata,
                        "raw_text": content.strip(),
                        "raw_tables": raw_tables,
                        "images": images,
                    },
                )
            )
            final_index += 1

    return final_docs


def _create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name):
        print(f"[*] Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    print(f"[✓] Created Qdrant collection: {collection_name}")


def _upload_in_batches(
    client: QdrantClient,
    collection_name: str,
    all_docs: List[Document],
    encoder: Any,
) -> None:
    if not all_docs:
        return

    total = len(all_docs)
    for start_idx in tqdm(range(0, total, BATCH_SIZE), desc=f"Uploading to {collection_name}", unit="batch"):
        batch_docs = all_docs[start_idx : start_idx + BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_vectors = encoder.encode(
            batch_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        points: List[qdrant_models.PointStruct] = []
        for idx, (doc, vector) in enumerate(zip(batch_docs, batch_vectors), start=start_idx):
            doc_metadata = dict(doc.metadata or {})
            payload = {
                "page_content": doc.page_content,
                "metadata": _serialize_metadata(doc_metadata),
                "raw_text": doc_metadata.get("raw_text", ""),
                "raw_tables": doc_metadata.get("raw_tables", []),
                "images": doc_metadata.get("images", []),
            }
            points.append(
                qdrant_models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )


def main() -> None:
    print("\n" + "=" * 80)
    print("MARKDOWN MULTI-MANUAL QDRANT INGESTION")
    print("=" * 80)

    encoder = get_embedding_model()
    qdrant_location = settings.qdrant_location
    if not settings.qdrant_url:
        os.makedirs(settings.qdrant_path, exist_ok=True)
    qdrant_client = QdrantClient(**settings.qdrant_client_kwargs)

    sample_vector = encoder.encode("vector-size-check", normalize_embeddings=True)
    vector_size = len(sample_vector)

    jobs = discover_markdown_jobs()
    if not jobs:
        raise RuntimeError(f"No markdown files found in {MARKDOWN_DIR}")

    print(f"[*] Markdown files discovered: {len(jobs)}")
    for job in jobs:
        print("\n" + "-" * 60)
        print(f"[*] Processing: {job['metadata_source'].upper()}")

        markdown_path = MARKDOWN_DIR / job["filename"]
        if not markdown_path.exists():
            print(f"[!] Markdown file not found: {markdown_path}")
            continue

        md_text = markdown_path.read_text(encoding="utf-8")
        all_docs = process_markdown(
            md_text,
            filename=job["filename"],
            metadata_source=job["metadata_source"],
        )
        if not all_docs:
            print(f"[!] No chunks produced for {job['filename']}")
            continue

        print(f"[✓] Generated {len(all_docs)} chunks from {job['filename']}")
        _create_qdrant_collection(qdrant_client, job["collection"], vector_size=vector_size)
        _upload_in_batches(qdrant_client, job["collection"], all_docs, encoder)
        print(f"[✓] Successfully ingested {job['collection']}")

    print("\n" + "=" * 80)
    print("[✓] FULL MARKDOWN INGESTION COMPLETE")
    print(f"    Qdrant: {qdrant_location}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
