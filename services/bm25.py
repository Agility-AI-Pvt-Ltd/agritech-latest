from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from core.config import settings

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def _chunk_words(words: List[str], *, max_words: int, overlap: int = 40) -> Iterable[str]:
    if not words:
        return

    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if len(chunk) < 20:
            continue
        yield " ".join(chunk)


def _normalize_heading(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).strip()


def _documents_from_markdown(path: Path, *, max_words: int) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    docs: List[Document] = []
    heading_path: List[str] = []
    buffer: List[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index
        section_text = "\n".join(buffer).strip()
        buffer.clear()
        if not section_text:
            return

        words = section_text.split()
        for chunk_index, chunk in enumerate(_chunk_words(words, max_words=max_words)):
            heading = " > ".join(heading_path)
            page_content = f"Heading: {heading}\n\n{chunk}" if heading else chunk
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "retrieval_source": "bm25_markdown",
                        "filename": path.name,
                        "source_path": str(path),
                        "heading_path": heading,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    },
                )
            )
        section_index += 1

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
    return docs


def rank_documents(query: str, documents: List[Document], *, top_k: int) -> List[Document]:
    """Rank an arbitrary document list with BM25 and return copied Documents."""
    if not query.strip() or not documents:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    tokenized_docs = [_tokenize(doc.page_content) for doc in documents]
    total_docs = len(tokenized_docs)
    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(total_docs, 1)
    doc_freq: dict[str, int] = {}

    for tokens in tokenized_docs:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    k1 = 1.5
    b = 0.75
    scored: List[tuple[float, int]] = []
    for idx, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue

        term_counts: dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        score = 0.0
        doc_len = len(tokens)
        for token in query_tokens:
            tf = term_counts.get(token, 0)
            if tf == 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
            score += idf * ((tf * (k1 + 1)) / denom)

        if score > 0:
            scored.append((score, idx))

    scored.sort(reverse=True)
    ranked: List[Document] = []
    for rank, (score, idx) in enumerate(scored[:top_k], start=1):
        doc = documents[idx]
        metadata = dict(doc.metadata or {})
        metadata["bm25_score"] = round(score, 4)
        metadata["bm25_rank"] = rank
        ranked.append(Document(page_content=doc.page_content, metadata=metadata))
    return ranked


@dataclass
class BM25MarkdownRetriever:
    markdown_dir: str
    chunk_words: int

    def __post_init__(self) -> None:
        self._docs: List[Document] = []
        self._fingerprint: tuple[tuple[str, float, int], ...] = ()

    def _current_fingerprint(self) -> tuple[tuple[str, float, int], ...]:
        root = Path(self.markdown_dir)
        if not root.exists():
            return ()
        return tuple(
            sorted(
                (str(path), path.stat().st_mtime, path.stat().st_size)
                for path in root.glob("*.md")
                if path.is_file()
            )
        )

    def _load_if_needed(self) -> None:
        fingerprint = self._current_fingerprint()
        if fingerprint == self._fingerprint:
            return

        docs: List[Document] = []
        for file_path, _mtime, _size in fingerprint:
            docs.extend(_documents_from_markdown(Path(file_path), max_words=self.chunk_words))

        self._docs = docs
        self._fingerprint = fingerprint
        print(f"[*] BM25 markdown index loaded: {len(self._docs)} chunks from {self.markdown_dir}")

    def is_loaded(self) -> bool:
        self._load_if_needed()
        return bool(self._docs)

    def search(self, query: str, *, top_k: int | None = None) -> List[Document]:
        self._load_if_needed()
        return rank_documents(query, self._docs, top_k=top_k or settings.bm25_top_k)


_bm25_retriever: BM25MarkdownRetriever | None = None


def get_bm25_retriever() -> BM25MarkdownRetriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        _bm25_retriever = BM25MarkdownRetriever(
            markdown_dir=settings.bm25_markdown_dir,
            chunk_words=settings.bm25_chunk_words,
        )
    return _bm25_retriever
