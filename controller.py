from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

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
from services.image_generation import generate_image
from services.generate_text import generate_linkedin_post


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
        SESSION_TEXT_PARTITIONS.add("web_url_text")
        SESSION_IMAGE_PARTITIONS.add("web_url_images")
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
        SESSION_TEXT_PARTITIONS.add("doc_text")
        SESSION_IMAGE_PARTITIONS.add("doc_images")
    except Exception as exc:
        print(f"Indexing error (doc): {exc}")

    return ParseDocumentResponse(text_file=text_file, image_files=image_files)


# -----------------------------
# Search APIs
# -----------------------------


ALLOWED_TEXT_PARTITIONS = {"web_url_text", "doc_text"}
ALLOWED_IMAGE_PARTITIONS = {"web_url_images", "doc_images"}

# Track which text partitions were indexed in this server session
SESSION_TEXT_PARTITIONS: set[str] = set()
SESSION_IMAGE_PARTITIONS: set[str] = set()


class SearchTextRequest(BaseModel):
    query: str
    partition: Optional[str] = None
    top_k: int = 5


class SearchImageRequest(BaseModel):
    query: str
    partition: Optional[str] = None
    top_k: int = 5


class SearchResultItem(BaseModel):
    id: int
    score: float
    payload: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


@app.post("/search_text", response_model=SearchResponse)
def search_text(payload: SearchTextRequest) -> SearchResponse:
    query_vec = embed_texts([payload.query])

    if payload.partition:
        if payload.partition not in ALLOWED_TEXT_PARTITIONS:
            raise HTTPException(status_code=400, detail="Invalid text partition")
        results_rows = search_embeddings(payload.partition, query_vec, top_k=payload.top_k)
        row = results_rows[0] if results_rows else []
        items: List[SearchResultItem] = [
            SearchResultItem(id=_id, score=score, payload=payload_str)
            for _id, score, payload_str in row
        ]
        return SearchResponse(results=items)

    if not SESSION_TEXT_PARTITIONS:
        raise HTTPException(status_code=400, detail="No indexed content in this session")

    aggregated: List[tuple[float, int, str]] = []
    for part in SESSION_TEXT_PARTITIONS:
        try:
            rows = search_embeddings(part, query_vec, top_k=payload.top_k)
            row = rows[0] if rows else []
            for _id, score, payload_str in row:
                aggregated.append((float(score), int(_id), payload_str))
        except Exception:
            continue

    aggregated.sort(key=lambda x: x[0], reverse=True)
    aggregated = aggregated[: payload.top_k]

    items = [
        SearchResultItem(id=_id, score=score, payload=payload_str)
        for score, _id, payload_str in aggregated
    ]
    return SearchResponse(results=items)


# -----------------------------
# Image Generation API
# -----------------------------


class GenerateImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int | None = None


class GenerateImageResponse(BaseModel):
    image_path: str


@app.post("/generate_image", response_model=GenerateImageResponse)
def generate_image_api(payload: GenerateImageRequest) -> GenerateImageResponse:
    try:
        out_path = generate_image(
            prompt=payload.prompt,
            num_inference_steps=payload.num_inference_steps,
            guidance_scale=payload.guidance_scale,
            seed=payload.seed,
        )
        return GenerateImageResponse(image_path=out_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}")


# -----------------------------
# Text Generation API (Ollama)
# -----------------------------


class GenerateTextRequest(BaseModel):
    prompt: str
    max_tokens: int = 400


class GenerateTextResponse(BaseModel):
    content: str


@app.post("/generate_text", response_model=GenerateTextResponse)
def generate_text_api(payload: GenerateTextRequest) -> GenerateTextResponse:
    try:
        content = generate_linkedin_post(prompt=payload.prompt, max_tokens=payload.max_tokens)
        return GenerateTextResponse(content=content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}")


class StoreImageRequest(BaseModel):
    file_path: str


class StoreImageResponse(BaseModel):
    saved_path: str


@app.post("/store_uploaded_image", response_model=StoreImageResponse)
def store_uploaded_image(payload: StoreImageRequest) -> StoreImageResponse:
    src = Path(payload.file_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=400, detail="File does not exist or is not a file")

    dest_dir = Path(__file__).resolve().parent / "uploaded_images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return StoreImageResponse(saved_path=str(dest.resolve()))


@app.post("/search_images", response_model=SearchResponse)
def search_images(payload: SearchImageRequest) -> SearchResponse:
    query_vec = embed_text_queries([payload.query])

    if payload.partition:
        if payload.partition not in ALLOWED_IMAGE_PARTITIONS:
            raise HTTPException(status_code=400, detail="Invalid image partition")
        results_rows = search_embeddings(payload.partition, query_vec, top_k=payload.top_k)
        row = results_rows[0] if results_rows else []
        items: List[SearchResultItem] = [
            SearchResultItem(id=_id, score=score, payload=payload_str)
            for _id, score, payload_str in row
        ]
        return SearchResponse(results=items)

    # If not provided, require session-based image partitions (latest indexed during this process)
    if not SESSION_IMAGE_PARTITIONS:
        raise HTTPException(status_code=400, detail="No indexed images in this session")

    aggregated: List[tuple[float, int, str]] = []
    for part in SESSION_IMAGE_PARTITIONS:
        try:
            rows = search_embeddings(part, query_vec, top_k=payload.top_k)
            row = rows[0] if rows else []
            for _id, score, payload_str in row:
                aggregated.append((float(score), int(_id), payload_str))
        except Exception:
            continue

    aggregated.sort(key=lambda x: x[0], reverse=True)
    aggregated = aggregated[: payload.top_k]

    items = [
        SearchResultItem(id=_id, score=score, payload=payload_str)
        for score, _id, payload_str in aggregated
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
    # Mount Gradio UI under a stable internal path
    app = mount_gradio_app(app, demo, path="/ui-frame/")

    @app.get("/ui", include_in_schema=False)
    def _ui_host_page() -> HTMLResponse:
        html = (
            "<html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>"
            "<title>AIPost UI</title></head>"
            "<body style='margin:0;padding:0;height:100vh;overflow:hidden;'>"
            "<iframe src='/ui-frame/' style='width:100%;height:100%;border:0;'></iframe>"
            "</body></html>"
        )
        return HTMLResponse(content=html)

    @app.get("/ui/", include_in_schema=False)
    def _ui_host_page_slash() -> HTMLResponse:
        return _ui_host_page()  # same content

    try:
        # Debug: list routes on startup to verify mounts are present
        routes = [getattr(r, "path", str(r)) for r in app.router.routes]
        print("Mounted routes:", routes)
    except Exception:
        pass
except Exception as exc:
    print(f"Failed to mount Gradio UI: {exc}")


