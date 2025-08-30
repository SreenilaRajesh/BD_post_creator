from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import pdfplumber
from fastapi import HTTPException


def ensure_dir(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def write_text_file(output_path: Path, content: str) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(content, encoding="utf-8")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
        combined = "\n\n".join(pages_text)
        return combined.strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF text: {exc}")


def extract_pdf_images(pdf_path: Path, images_dir: Path) -> List[str]:
    ensure_dir(images_dir)
    saved_images: List[str] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            image_counter = 0
            for page_index in range(len(doc)):
                page = doc[page_index]
                for img in page.get_images(full=True):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    try:
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        image_counter += 1
                        out_path = images_dir / f"image_{image_counter:03d}.jpg"
                        pix.save(str(out_path))
                        saved_images.append(str(out_path.resolve()))
                    finally:
                        pix = None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF images: {exc}")

    return saved_images


def parse_pdf_to_files(source_file_path: str) -> Tuple[str, List[str]]:
    source_path = Path(source_file_path).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=400, detail="File does not exist or is not a file")

    temp_dir = Path("input_documents") / timestamp_slug()
    ensure_dir(temp_dir)
    copied_path = temp_dir / source_path.name
    shutil.copy2(source_path, copied_path)

    doc_output_dir = Path("doc_inputs") / timestamp_slug()
    text_file = doc_output_dir / "content.txt"
    images_dir = doc_output_dir / "images"
    ensure_dir(doc_output_dir)
    ensure_dir(images_dir)

    text_content = extract_pdf_text(copied_path)
    write_text_file(text_file, text_content)

    image_files = extract_pdf_images(copied_path, images_dir)

    return str(text_file.resolve()), image_files


