from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from fastapi import HTTPException

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception as exc:  # pragma: no cover - optional dependency
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

from .faiss_service import store_embeddings


DEFAULT_IMAGE_MODEL = "openai/clip-vit-base-patch32"


def load_image_embedding_model(model_name: str = DEFAULT_IMAGE_MODEL):
    if CLIPModel is None or CLIPProcessor is None:
        raise HTTPException(status_code=500, detail="transformers is not installed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


def read_images(image_paths: Iterable[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception:
            continue
    return images


def embed_images(image_paths: Iterable[str], model=None, processor=None, device: str = "cpu") -> np.ndarray:
    paths = [p for p in image_paths]
    if not paths:
        return np.zeros((0, 512), dtype="float32")

    if model is None or processor is None:
        model, processor, device = load_image_embedding_model()

    images = read_images(paths)
    if not images:
        return np.zeros((0, 512), dtype="float32")

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        vectors = outputs.cpu().numpy()
    return vectors.astype("float32", copy=False)


def index_image_paths(image_paths: List[str], partition: str) -> List[int]:
    if not image_paths:
        return []
    vectors = embed_images(image_paths)
    ids = store_embeddings(partition, vectors, payloads=image_paths)
    return ids


def embed_text_queries(queries: Iterable[str], model=None, processor=None, device: str = "cpu") -> np.ndarray:
    texts = [q for q in queries if q]
    if not texts:
        return np.zeros((0, 512), dtype="float32")

    if model is None or processor is None:
        model, processor, device = load_image_embedding_model()

    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        vectors = outputs.cpu().numpy()
    return vectors.astype("float32", copy=False)


