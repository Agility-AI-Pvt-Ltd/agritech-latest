import pandas as pd
import os
import sys
from typing import Any, Dict, List
from tqdm.auto import tqdm

# Ensure the parent directory is in the path to import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from core.config import settings

BATCH_SIZE = 128


def _discover_pdfs(data_dir: str) -> List[str]:
    pdf_paths: List[str] = []
    for root, _dirs, files in os.walk(data_dir):
        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file_name))
    return sorted(pdf_paths)


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def _load_excel_documents(excel_path: str) -> List[Document]:
    if not os.path.exists(excel_path):
        print(f"[!] Excel file not found: {excel_path} (continuing with PDFs)")
        return []

    try:
        df = pd.read_excel(excel_path)
        df["text_content"] = (
            "Question: "
            + df["पूछे जाने वाले प्रश्न-"].fillna("").astype(str)
            + "\nSuggestion: "
            + df["सुझाव"].fillna("").astype(str)
        )
        excel_loader = DataFrameLoader(df, page_content_column="text_content")
        docs = excel_loader.load()
        print(f"[✓] Loaded Excel rows: {len(docs)} from {excel_path}")
        return docs
    except Exception as exc:
        print(f"[!] Failed to load Excel: {exc}")
        return []


def _load_pdf_documents(pdf_files: List[str]) -> List[Document]:
    pdf_docs: List[Document] = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs", unit="file"):
        try:
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            pdf_docs.extend(docs)
            print(f"[✓] Loaded PDF: {pdf_file} (pages: {len(docs)})")
        except Exception as exc:
            print(f"[!] Failed to load PDF {pdf_file}: {exc}")
    return pdf_docs


def _split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "।", ".", " ", ""],
    )
    return text_splitter.split_documents(documents)


def _create_qdrant_collection(client: QdrantClient, vector_size: int) -> None:
    if client.collection_exists(settings.qdrant_collection_name):
        client.delete_collection(collection_name=settings.qdrant_collection_name)

    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def _upload_in_batches(client: QdrantClient, all_docs: List[Document], encoder: SentenceTransformer) -> None:
    total = len(all_docs)
    batch_starts = list(range(0, total, BATCH_SIZE))
    for start_idx in tqdm(batch_starts, desc="Uploading vectors", unit="batch"):
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_docs = all_docs[start_idx:end_idx]
        batch_texts = [doc.page_content for doc in batch_docs]
        batch_vectors = encoder.encode(
            batch_texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

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

        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        tqdm.write(f"[✓] Uploaded chunks: {start_idx + 1}-{end_idx} / {total}")


def main() -> None:
    print("\n" + "=" * 72)
    print("🚜 AGRITECH INGESTION PIPELINE (QDRANT)")
    print("=" * 72)

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    excel_path = os.path.join(base_path, "Maize package of practice .xlsx")

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Data folder not found: {base_path}")

    pdf_files = _discover_pdfs(base_path)
    print(f"[*] PDFs discovered in data folder: {len(pdf_files)}")

    excel_docs = _load_excel_documents(excel_path)
    pdf_docs = _load_pdf_documents(pdf_files)

    source_docs = excel_docs + pdf_docs
    if not source_docs:
        raise RuntimeError("No source documents found (Excel/PDF).")

    all_docs = _split_documents(source_docs)
    print(f"[*] Total chunks generated: {len(all_docs)}")

    encoder = SentenceTransformer(settings.sentence_transformer_model)

    os.makedirs(settings.qdrant_path, exist_ok=True)
    qdrant_client = QdrantClient(path=settings.qdrant_path)

    sample_vector = encoder.encode("vector-size-check", normalize_embeddings=True)
    _create_qdrant_collection(qdrant_client, vector_size=len(sample_vector))
    _upload_in_batches(qdrant_client, all_docs, encoder)

    print(
        f"[✓] Ingestion complete. Collection '{settings.qdrant_collection_name}' now contains {len(all_docs)} chunks."
    )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()


