from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from fastapi import HTTPException

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from .faiss_service import store_embeddings


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_text_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="sentence-transformers is not installed")
    return SentenceTransformer(model_name)


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if overlap >= max_chars:
        overlap = max_chars // 5  # keep overlap smaller than window
    chunks: List[str] = []
    n = len(text)
    start = 0
    step = max(1, max_chars - overlap)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step
    return chunks


def embed_texts(texts: Iterable[str], model=None) -> np.ndarray:
    texts_list = [t for t in texts if t]
    if not texts_list:
        return np.zeros((0, 384), dtype="float32")
    if model is None:
        model = load_text_embedding_model()
    vectors = model.encode(texts_list, convert_to_numpy=True, normalize_embeddings=False)
    return vectors.astype("float32", copy=False)


def index_text_corpus(text: str, partition: str) -> List[int]:
    chunks = chunk_text(text)
    if not chunks:
        return []
    vectors = embed_texts(chunks)
    ids = store_embeddings(partition, vectors, payloads=chunks)
    return ids


