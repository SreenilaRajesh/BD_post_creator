from __future__ import annotations

import io
import time
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from PIL import Image
from fastapi import HTTPException
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


def ensure_dir(directory_path: Path) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)


def write_text_file(output_path: Path, content: str) -> None:
    ensure_dir(output_path.parent)
    output_path.write_text(content, encoding="utf-8")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def fetch_webpage(url: str, timeout_seconds: int = 20) -> str:
    try:
        response = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}")


def extract_text_and_image_urls(html_text: str, base_url: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html_text, "lxml")

    for tag_name in ["script", "style", "noscript"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text = soup.get_text(separator=" ")
    text = normalize_whitespace(text)

    image_urls: List[str] = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        try:
            abs_url = requests.compat.urljoin(base_url, src)
            image_urls.append(abs_url)
        except Exception:
            continue

    seen = set()
    deduped: List[str] = []
    for u in image_urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return text, deduped


def download_and_convert_images(image_urls: List[str], target_dir: Path, max_images: int = 20) -> List[str]:
    ensure_dir(target_dir)
    saved_files: List[str] = []

    for index, image_url in enumerate(image_urls[:max_images], start=1):
        try:
            resp = requests.get(image_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            image_bytes = io.BytesIO(resp.content)
            with Image.open(image_bytes) as img:
                rgb_img = img.convert("RGB")
                out_path = target_dir / f"image_{index:03d}.jpg"
                rgb_img.save(out_path, format="JPEG", quality=90)
                saved_files.append(str(out_path.resolve()))
        except Exception:
            continue

    return saved_files


def websearch_top_url(query: str, num_results: int = 5) -> Optional[str]:
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.results(query, max_results=num_results)
        for item in results:
            link = item.get("link")
            if isinstance(link, str) and link.startswith("http"):
                return link
    except Exception:
        return None
    return None


def parse_web_url_to_files(url: str) -> Tuple[str, List[str]]:
    base_dir = Path("web_inputs") / timestamp_slug()
    text_dir = base_dir
    images_dir = base_dir / "images"
    ensure_dir(text_dir)
    ensure_dir(images_dir)

    selected_url = url
    top = websearch_top_url(url)
    if top:
        selected_url = top

    html = fetch_webpage(selected_url)
    text, image_urls = extract_text_and_image_urls(html, base_url=url)

    text_file = text_dir / "content.txt"
    write_text_file(text_file, text)

    image_files = download_and_convert_images(image_urls, images_dir)

    return str(text_file.resolve()), image_files


