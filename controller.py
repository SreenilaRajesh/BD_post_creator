from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse
from gradio.routes import mount_gradio_app
from pydantic import BaseModel, HttpUrl

from services.parse_document import parse_pdf_to_files
from services.parse_web_url import parse_web_url_to_files
from services.text_indexer import index_text_corpus, embed_texts
from services.image_indexer import index_image_paths, embed_images, embed_text_queries
from services.faiss_service import search_embeddings


app = FastAPI(title="Content Parser API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------


class ParseWebUrlRequest(BaseModel):
    url: HttpUrl


class ParseWebUrlResponse(BaseModel):
    text_file: str
    image_files: List[str]


class ParseDocumentRequest(BaseModel):
    file_path: str


class ParseDocumentResponse(BaseModel):
    text_file: str
    image_files: List[str]


# -----------------------------
# Endpoints
# -----------------------------


@app.post("/parse_web_url", response_model=ParseWebUrlResponse)
def parse_web_url(payload: ParseWebUrlRequest) -> ParseWebUrlResponse:
    text_file, image_files = parse_web_url_to_files(str(payload.url))

    # Index into FAISS partitions
    try:
        text_content = Path(text_file).read_text(encoding="utf-8", errors="ignore")
        text_ids = index_text_corpus(text_content, partition="web_url_text")
        image_ids = index_image_paths(image_files, partition="web_url_images")
        print(f"Indexed web_url_text: {len(text_ids)} items; web_url_images: {len(image_ids)} items")
    except Exception as exc:
        print(f"Indexing error (web): {exc}")

    return ParseWebUrlResponse(text_file=text_file, image_files=image_files)


@app.post("/parse_document", response_model=ParseDocumentResponse)
def parse_document(payload: ParseDocumentRequest) -> ParseDocumentResponse:
    text_file, image_files = parse_pdf_to_files(payload.file_path)

    # Index into FAISS partitions
    try:
        text_content = Path(text_file).read_text(encoding="utf-8", errors="ignore")
        text_ids = index_text_corpus(text_content, partition="doc_text")
        image_ids = index_image_paths(image_files, partition="doc_images")
        print(f"Indexed doc_text: {len(text_ids)} items; doc_images: {len(image_ids)} items")
    except Exception as exc:
        print(f"Indexing error (doc): {exc}")

    return ParseDocumentResponse(text_file=text_file, image_files=image_files)


# -----------------------------
# Search APIs
# -----------------------------


ALLOWED_TEXT_PARTITIONS = {"web_url_text", "doc_text"}
ALLOWED_IMAGE_PARTITIONS = {"web_url_images", "doc_images"}


class SearchTextRequest(BaseModel):
    query: str
    partition: str = "web_url_text"
    top_k: int = 5


class SearchImageRequest(BaseModel):
    query: str
    partition: str = "web_url_images"
    top_k: int = 5


class SearchResultItem(BaseModel):
    id: int
    score: float
    payload: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


@app.post("/search_text", response_model=SearchResponse)
def search_text(payload: SearchTextRequest) -> SearchResponse:
    if payload.partition not in ALLOWED_TEXT_PARTITIONS:
        raise HTTPException(status_code=400, detail="Invalid text partition")

    query_vec = embed_texts([payload.query])
    results_rows = search_embeddings(payload.partition, query_vec, top_k=payload.top_k)
    # Single query -> one row
    row = results_rows[0] if results_rows else []
    items: List[SearchResultItem] = [
        SearchResultItem(id=_id, score=score, payload=payload_str)
        for _id, score, payload_str in row
    ]
    return SearchResponse(results=items)


@app.post("/search_images", response_model=SearchResponse)
def search_images(payload: SearchImageRequest) -> SearchResponse:
    if payload.partition not in ALLOWED_IMAGE_PARTITIONS:
        raise HTTPException(status_code=400, detail="Invalid image partition")

    query_vec = embed_text_queries([payload.query])
    results_rows = search_embeddings(payload.partition, query_vec, top_k=payload.top_k)
    row = results_rows[0] if results_rows else []
    items: List[SearchResultItem] = [
        SearchResultItem(id=_id, score=score, payload=payload_str)
        for _id, score, payload_str in row
    ]
    return SearchResponse(results=items)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Use /parse_web_url or /parse_document"}


# -----------------------------
# Mount Gradio UI
# -----------------------------

try:
    from ui.gradio_ui import build_ui

    demo = build_ui()
    app = mount_gradio_app(app, demo, path="/ui")
except Exception:
    pass


