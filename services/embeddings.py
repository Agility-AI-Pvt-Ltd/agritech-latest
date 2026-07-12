"""Embedding adapters with a small SentenceTransformer-compatible surface."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


class EmbeddingVector(list):
    def tolist(self) -> List[float]:
        return list(self)


class EmbeddingMatrix(list):
    def tolist(self) -> List[List[float]]:
        return [list(vector) for vector in self]


def _normalize(vector: Sequence[float]) -> EmbeddingVector:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not norm:
        return EmbeddingVector(float(value) for value in vector)
    return EmbeddingVector(float(value) / norm for value in vector)


class OpenAIEmbeddingEncoder:
    """Adapter exposing `.encode()` for existing ingestion/retrieval code."""

    def __init__(self, *, model: str, api_key: str | None = None):
        from langchain_openai import OpenAIEmbeddings

        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self.model_name = model
        self._embeddings = OpenAIEmbeddings(**kwargs)

    def encode(
        self,
        texts: str | Iterable[str],
        *,
        normalize_embeddings: bool = False,
        **_: object,
    ) -> EmbeddingVector | EmbeddingMatrix:
        if isinstance(texts, str):
            vector = self._embeddings.embed_query(texts)
            return _normalize(vector) if normalize_embeddings else EmbeddingVector(vector)

        vectors = self._embeddings.embed_documents(list(texts))
        if normalize_embeddings:
            return EmbeddingMatrix(_normalize(vector) for vector in vectors)
        return EmbeddingMatrix(EmbeddingVector(vector) for vector in vectors)
