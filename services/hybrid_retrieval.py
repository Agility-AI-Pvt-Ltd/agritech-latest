from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document

from core.config import settings


def _dedupe_key(text: str, metadata: dict) -> str:
    source = metadata.get("source_path") or metadata.get("filename") or metadata.get("collection") or ""
    heading = metadata.get("heading_path") or metadata.get("pageindex_path") or ""
    compact_text = " ".join((text or "").split())[:240]
    return f"{source}|{heading}|{compact_text}"


def reciprocal_rank_fuse_documents(
    ranked_sources: Dict[str, List[Document]],
    *,
    top_k: int,
    source_weights: Dict[str, float] | None = None,
) -> List[Document]:
    """Merge ranked result lists with reciprocal-rank fusion."""
    weights = source_weights or {}
    fused: dict[str, dict] = {}

    for source_name, docs in ranked_sources.items():
        weight = weights.get(source_name, 1.0)
        for rank, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            key = _dedupe_key(doc.page_content, metadata)
            entry = fused.setdefault(
                key,
                {
                    "score": 0.0,
                    "content": doc.page_content,
                    "metadata": metadata,
                    "sources": [],
                },
            )
            entry["score"] += weight / (60 + rank)
            entry["sources"].append(source_name)

    ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    results: List[Document] = []
    for rank, item in enumerate(ordered[:top_k], start=1):
        metadata = dict(item["metadata"])
        metadata["hybrid_rank"] = rank
        metadata["hybrid_score"] = round(float(item["score"]), 6)
        metadata["retrieval_sources"] = sorted(set(item["sources"]))
        results.append(Document(page_content=item["content"], metadata=metadata))
    return results


def document_to_chunk(doc: Document, *, default_collection: str, sub_query: str) -> dict:
    metadata = dict(doc.metadata or {})
    collection = (
        metadata.get("collection")
        or metadata.get("retrieval_source")
        or default_collection
    )
    score = (
        metadata.get("hybrid_score")
        or metadata.get("retrieval_score")
        or metadata.get("bm25_score")
        or 0.0
    )
    return {
        "collection": collection,
        "sub_query": sub_query,
        "score": score,
        "content": doc.page_content,
        "metadata": metadata,
    }


def chunks_to_documents(chunks: List[dict], *, source_name: str) -> List[Document]:
    docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.get("metadata") or {})
        metadata.setdefault("collection", chunk.get("collection", source_name))
        metadata.setdefault("retrieval_source", source_name)
        metadata["retrieval_score"] = chunk.get("score", 0.0)
        metadata["retrieval_rank"] = idx + 1
        docs.append(Document(page_content=chunk.get("content", ""), metadata=metadata))
    return docs


def hybrid_context(
    *,
    query: str,
    stage: str | None,
    vector_store,
    pageindex_provider,
    top_k: int | None = None,
) -> str:
    from services.bm25 import get_bm25_retriever

    limit = top_k or settings.hybrid_top_k
    search_query = f"{query}\nCrop stage: {stage} maize" if stage else query
    sources: Dict[str, List[Document]] = {}

    if vector_store and vector_store.is_loaded():
        sources["rag"] = vector_store.search(search_query, k=limit)

    if pageindex_provider and pageindex_provider.is_loaded():
        sources["pageindex"] = pageindex_provider.search_documents(search_query, k=settings.pageindex_max_nodes)

    bm25 = get_bm25_retriever()
    if bm25.is_loaded():
        sources["bm25"] = bm25.search(search_query, top_k=settings.bm25_top_k)

    merged = reciprocal_rank_fuse_documents(
        sources,
        top_k=limit,
        source_weights={"rag": 1.0, "bm25": 0.9, "pageindex": 0.8},
    )
    return "\n\n".join(doc.page_content for doc in merged)
