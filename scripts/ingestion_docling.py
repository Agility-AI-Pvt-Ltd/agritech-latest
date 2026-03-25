"""
Multi-Manual Vectorstore Ingestion via Docling 
(Advanced Document Parsing & Semantic Hierarchical Chunking)
"""

import os
import sys
from typing import Any, Dict, List
from tqdm.auto import tqdm

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

BATCH_SIZE = 128

def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to a serializable format."""
    serialized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


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

        # 3. Convert to Langchain Documents
        langchain_docs = []
        for c in docling_chunks:
            # We map docling chunks to standard Langchain docs for Qdrant
            doc = Document(
                page_content=c.text,
                metadata={
                    "source": metadata_source,
                    "document_type": "agricultural_manual",
                    "filename": os.path.basename(pdf_path)
                }
            )
            langchain_docs.append(doc)
            
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
    
    qdrant_path = os.getenv(
        "QDRANT_PATH",
        "/Users/krishnakumar/Downloads/merged/db_storage"
    )
    os.makedirs(qdrant_path, exist_ok=True)
    qdrant_client = QdrantClient(path=qdrant_path)
    
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
    print(f"    Qdrant Path: {qdrant_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
