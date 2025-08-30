from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "faiss_indices"


def ensure_dir(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def _index_path(partition: str) -> Path:
    ensure_dir(BASE_DIR)
    return BASE_DIR / f"{partition}.index"


def _meta_path(partition: str) -> Path:
    ensure_dir(BASE_DIR)
    return BASE_DIR / f"{partition}_meta.jsonl"


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array [n, d]")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def load_index(partition: str) -> faiss.Index | None:
    path = _index_path(partition)
    if not path.exists():
        return None
    index = faiss.read_index(str(path))
    return index


def get_or_create_index(partition: str, dimension: int) -> faiss.Index:
    index = load_index(partition)
    if index is None:
        # Cosine similarity via inner product after normalization
        base = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap2(base)
        save_index(partition, index)
    else:
        if index.d != dimension:
            raise HTTPException(status_code=400, detail=f"Dimension mismatch for partition '{partition}' (have {index.d}, need {dimension})")
    return index


def save_index(partition: str, index: faiss.Index) -> None:
    path = _index_path(partition)
    faiss.write_index(index, str(path))
    try:
        print(f"FAISS index saved: {path}")
    except Exception:
        pass


def _read_meta_map(partition: str) -> Dict[int, str]:
    meta_file = _meta_path(partition)
    if not meta_file.exists():
        return {}
    id_to_payload: Dict[int, str] = {}
    with meta_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                id_to_payload[int(obj["id"])] = str(obj["payload"])
            except Exception:
                continue
    return id_to_payload


def _append_meta(partition: str, ids: Iterable[int], payloads: Iterable[str]) -> None:
    meta_file = _meta_path(partition)
    with meta_file.open("a", encoding="utf-8") as f:
        for _id, payload in zip(ids, payloads):
            f.write(json.dumps({"id": int(_id), "payload": payload}, ensure_ascii=False) + "\n")


def _current_count(partition: str) -> int:
    meta_file = _meta_path(partition)
    if not meta_file.exists():
        return 0
    # Count lines efficiently
    count = 0
    with meta_file.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def store_embeddings(partition: str, embeddings: np.ndarray, payloads: List[str]) -> List[int]:
    if embeddings.shape[0] != len(payloads):
        raise ValueError("Number of embeddings and payloads must match")

    embeddings = _normalize_embeddings(embeddings.astype("float32", copy=False))

    index = get_or_create_index(partition, dimension=embeddings.shape[1])

    start_id = _current_count(partition)
    ids = np.arange(start_id, start_id + embeddings.shape[0], dtype="int64")

    # Add with explicit IDs
    if not isinstance(index, faiss.IndexIDMap2):
        index = faiss.IndexIDMap2(index)
    index.add_with_ids(embeddings, ids)

    save_index(partition, index)
    _append_meta(partition, ids.tolist(), payloads)

    return ids.tolist()


def search_embeddings(partition: str, query_embeddings: np.ndarray, top_k: int = 5) -> List[List[Tuple[int, float, str]]]:
    index = load_index(partition)
    if index is None or index.ntotal == 0:
        raise HTTPException(status_code=404, detail=f"No index found for partition '{partition}'")

    query_embeddings = _normalize_embeddings(query_embeddings.astype("float32", copy=False))
    scores, idxs = index.search(query_embeddings, top_k)

    id_to_payload = _read_meta_map(partition)

    results: List[List[Tuple[int, float, str]]] = []
    for row_scores, row_ids in zip(scores, idxs):
        triples: List[Tuple[int, float, str]] = []
        for id_val, score in zip(row_ids, row_scores):
            if int(id_val) == -1:
                continue
            payload = id_to_payload.get(int(id_val), "")
            triples.append((int(id_val), float(score), payload))
        results.append(triples)
    return results


