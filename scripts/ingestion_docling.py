"""
Multi-Manual Vectorstore Ingestion via Docling 
(Advanced Document Parsing & Semantic Hierarchical Chunking)
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import Any, Dict, List
from tqdm.auto import tqdm

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.config import settings

BATCH_SIZE = 128
MIN_CHARS = 120
MERGE_BELOW_CHARS = 260
TARGET_CHUNK_CHARS = 900
MAX_CHUNK_CHARS = 1400
OVERLAP_CHARS = 140


@dataclass
class PreparedChunk:
    text: str
    headings: List[str]
    metadata: Dict[str, Any]


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to a serializable format."""
    serialized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_chunk_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<![.\n:;])\n(?=[a-z0-9(])", " ", text, flags=re.IGNORECASE)

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Drop likely page-number lines and common OCR artifacts.
        if re.fullmatch(r"(page\s*)?\d{1,4}", line.lower()):
            continue
        if re.fullmatch(r"[-_=~•.]{3,}", line):
            continue
        if len(line) <= 3 and not re.search(r"[A-Za-z]", line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_noise_lines(chunks: List[PreparedChunk]) -> List[PreparedChunk]:
    line_frequency: Dict[str, int] = {}
    for chunk in chunks:
        seen_in_chunk = set()
        for line in chunk.text.splitlines():
            normalized = line.strip().lower()
            if not normalized or len(normalized) < 4 or len(normalized) > 80:
                continue
            seen_in_chunk.add(normalized)
        for line in seen_in_chunk:
            line_frequency[line] = line_frequency.get(line, 0) + 1

    repeated_lines = {
        line
        for line, count in line_frequency.items()
        if count >= 3 and not re.search(r"\d+\s*(kg|g|ml|l|days?)\b", line)
    }

    if not repeated_lines:
        return chunks

    cleaned_chunks: List[PreparedChunk] = []
    for chunk in chunks:
        kept_lines = [
            line
            for line in chunk.text.splitlines()
            if line.strip().lower() not in repeated_lines
        ]
        updated_text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
        if updated_text:
            chunk.text = updated_text
            cleaned_chunks.append(chunk)
    return cleaned_chunks


def _looks_like_low_information_chunk(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) < MIN_CHARS:
        return True

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) == 1:
        line = lines[0]
        if len(line.split()) <= 6 and not re.search(r"[.!?]", line):
            return True

    alpha_chars = sum(ch.isalpha() for ch in stripped)
    if alpha_chars < 40:
        return True

    punctuation_density = len(re.findall(r"[|_]{2,}|\.{4,}", stripped))
    if punctuation_density > 2:
        return True

    return False


def _compose_chunk_text(headings: List[str], text: str) -> str:
    cleaned_headings = [_normalize_heading_text(h) for h in headings if _normalize_heading_text(h)]
    if not cleaned_headings:
        return text

    heading_path = " > ".join(cleaned_headings)
    body = text.strip()
    if body.lower().startswith(heading_path.lower()):
        return body
    return f"{heading_path}\n\n{body}"


def _extract_chunk_metadata(chunk: Any, metadata_source: str, filename: str, chunk_index: int) -> Dict[str, Any]:
    headings = list(getattr(getattr(chunk, "meta", None), "headings", None) or [])
    doc_items = list(getattr(getattr(chunk, "meta", None), "doc_items", None) or [])
    page_numbers: List[int] = []

    for item in doc_items:
        prov = getattr(item, "prov", None) or []
        for entry in prov:
            page_no = getattr(entry, "page_no", None)
            if isinstance(page_no, int):
                page_numbers.append(page_no)

    unique_pages = sorted(set(page_numbers))
    return {
        "source": metadata_source,
        "document_type": "agricultural_manual",
        "filename": filename,
        "chunk_index": chunk_index,
        "headings": headings,
        "section_path": " > ".join(_normalize_heading_text(h) for h in headings if _normalize_heading_text(h)),
        "page_start": unique_pages[0] if unique_pages else None,
        "page_end": unique_pages[-1] if unique_pages else None,
    }


def _merge_small_chunks(chunks: List[PreparedChunk]) -> List[PreparedChunk]:
    merged: List[PreparedChunk] = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            same_heading = prev.headings == chunk.headings
            prev_short = len(prev.text) < MERGE_BELOW_CHARS
            cur_short = len(chunk.text) < MERGE_BELOW_CHARS
            combined_len = len(prev.text) + len(chunk.text)
            if same_heading and (prev_short or cur_short) and combined_len <= MAX_CHUNK_CHARS:
                prev.text = f"{prev.text}\n\n{chunk.text}".strip()
                prev.metadata["page_end"] = chunk.metadata.get("page_end") or prev.metadata.get("page_end")
                continue
        merged.append(chunk)
    return merged


def _split_large_chunk(chunk: PreparedChunk) -> List[PreparedChunk]:
    text = chunk.text.strip()
    if len(text) <= MAX_CHUNK_CHARS:
        return [chunk]

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    windows: List[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= TARGET_CHUNK_CHARS:
            current = candidate
            continue

        if current:
            windows.append(current.strip())

        if len(paragraph) <= MAX_CHUNK_CHARS:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + TARGET_CHUNK_CHARS, len(paragraph))
            split_at = paragraph.rfind(" ", start, end)
            if split_at <= start + 200:
                split_at = end
            windows.append(paragraph[start:split_at].strip())
            if split_at >= len(paragraph):
                break
            start = max(split_at - OVERLAP_CHARS, start + 1)
        current = ""

    if current:
        windows.append(current.strip())

    split_chunks: List[PreparedChunk] = []
    total_parts = len(windows)
    for part_idx, window in enumerate(windows, start=1):
        part_metadata = dict(chunk.metadata)
        part_metadata["chunk_part"] = part_idx
        part_metadata["chunk_parts_total"] = total_parts
        split_chunks.append(
            PreparedChunk(
                text=window,
                headings=list(chunk.headings),
                metadata=part_metadata,
            )
        )
    return split_chunks


def _prepare_docling_chunks(docling_chunks: List[Any], metadata_source: str, filename: str) -> List[Document]:
    prepared_chunks: List[PreparedChunk] = []

    for idx, chunk in enumerate(docling_chunks):
        normalized_text = _normalize_chunk_text(getattr(chunk, "text", ""))
        metadata = _extract_chunk_metadata(chunk, metadata_source, filename, idx)
        headings = list(metadata.get("headings") or [])

        if not normalized_text:
            continue

        prepared_chunks.append(
            PreparedChunk(
                text=normalized_text,
                headings=headings,
                metadata=metadata,
            )
        )

    prepared_chunks = _remove_repeated_noise_lines(prepared_chunks)
    prepared_chunks = [chunk for chunk in prepared_chunks if not _looks_like_low_information_chunk(chunk.text)]
    prepared_chunks = _merge_small_chunks(prepared_chunks)

    final_chunks: List[PreparedChunk] = []
    for chunk in prepared_chunks:
        final_chunks.extend(_split_large_chunk(chunk))

    documents: List[Document] = []
    for idx, chunk in enumerate(final_chunks):
        metadata = dict(chunk.metadata)
        metadata["final_chunk_index"] = idx
        chunk_text = _compose_chunk_text(chunk.headings, chunk.text)
        if _looks_like_low_information_chunk(chunk_text):
            continue
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents


def _load_and_chunk_docling(pdf_path: str, metadata_source: str) -> List[Document]:
    """Load and semantically chunk PDF using IBM Docling."""
    if not os.path.exists(pdf_path):
        print(f"[!] PDF file not found: {pdf_path}")
        return []

    try:
        print(f"[*] Starting Docling deep parse on {os.path.basename(pdf_path)}...")
        
        # 1. Parse Document natively retaining tables, layouts, and OCR
        converter = DocumentConverter()
        doc_result = converter.convert(pdf_path)
        
        # 2. Use Docling's advanced semantic hierarchical chunker
        chunker = HierarchicalChunker()
        docling_chunks = list(chunker.chunk(doc_result.document))

        # 3. Normalize and reshape chunks for retrieval quality.
        langchain_docs = _prepare_docling_chunks(
            docling_chunks,
            metadata_source=metadata_source,
            filename=os.path.basename(pdf_path),
        )
            
        print(f"[✓] Docling extracted and generated {len(langchain_docs)} semantic chunks.")
        return langchain_docs

    except Exception as exc:
        print(f"[!] Failed to parse PDF using Docling: {exc}")
        return []


def _create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create or recreate a Qdrant collection."""
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
    encoder: SentenceTransformer
) -> None:
    """Upload documents to Qdrant in batches."""
    if not all_docs:
        return
        
    total = len(all_docs)
    batch_starts = list(range(0, total, BATCH_SIZE))
    
    for start_idx in tqdm(batch_starts, desc=f"Uploading to {collection_name}", unit="batch"):
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_docs = all_docs[start_idx:end_idx]
        batch_texts = [doc.page_content for doc in batch_docs]
        
        # Encode batch texts
        batch_vectors = encoder.encode(
            batch_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # Create Qdrant points
        points: List[qdrant_models.PointStruct] = []
        for idx, (doc, vector) in enumerate(zip(batch_docs, batch_vectors), start=start_idx):
            payload = {
                "page_content": doc.page_content,
                "metadata": _serialize_metadata(doc.metadata),
            }
            points.append(
                qdrant_models.PointStruct(
                    id=idx,
                    vector=vector,
                    payload=payload,
                )
            )

        # Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )


def main() -> None:
    """Main Docling ingestion pipeline for multiple manuals."""
    print("\n" + "=" * 80)
    print("🌱 DOCLING SEMANTIC VECTORSTORE INGESTION")
    print("=" * 80)

    # Initialize models and DB Client
    print("\n[*] Initializing embedding model and Qdrant...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    if not settings.qdrant_url:
        os.makedirs(settings.qdrant_path, exist_ok=True)
    qdrant_client = QdrantClient(**settings.qdrant_client_kwargs)
    
    sample_vector = encoder.encode("vector-size-check", normalize_embeddings=True)
    vector_size = len(sample_vector)

    # Define documents configuration
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    
    pdf_jobs = [
        {
            "filename": "fertilizer-materials-and-computation (1).pdf",
            "collection": "spring_corn_fertilizers_db",
            "metadata_source": "fertilizer materials and computation"
        },
        {
            "filename": "U82ManIitaProductionNothomNodev (1).pdf",
            "collection": "maize_production_manual_db",
            "metadata_source": "maize production manual"
        },
        {
            "filename": "Management_Pests_Diseases_Manual (1).pdf",
            "collection": "spring_corn_pest_and_diseases_db",
            "metadata_source": "pest and diseases of maize"
        },
        {
            "filename": "Spring Sweet Corn (zaid Maize) – Package Of Practices (pop) _ Uttar Pradesh (1) (1).pdf",
            "collection": "spring_corn_pop_db",
            "metadata_source": "package of practices for maize"
        }
    ]

    for job in pdf_jobs:
        print("\n" + "-" * 60)
        print(f"[*] Processing: {job['metadata_source'].upper()}")
        
        pdf_path = os.path.join(base_path, job["filename"])
        
        # Load and verify (Docling chunks natively based on doc structure!)
        all_docs = _load_and_chunk_docling(pdf_path, job["metadata_source"])
        if not all_docs:
            print(f"[!] Skipping {job['collection']} due to missing file or parsing error.")
            continue

        # Recreate DB
        _create_qdrant_collection(qdrant_client, job["collection"], vector_size=vector_size)

        # Upload
        print(f"[*] Starting vector upload to {job['collection']}...")
        _upload_in_batches(qdrant_client, job["collection"], all_docs, encoder)
        print(f"[✓] Successfully ingested {job['collection']}.")

    # Summary
    print("\n" + "=" * 80)
    print(f"[✓] FULL DOCLING INGESTION COMPLETE")
    print(f"    Qdrant: {settings.qdrant_location}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
