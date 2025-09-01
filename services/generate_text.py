from __future__ import annotations

import json
from typing import Optional

import requests
from fastapi import HTTPException

from config import OLLAMA_HOST, OLLAMA_MODEL


def generate_linkedin_post(prompt: str, max_tokens: int = 400) -> str:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    url = f"{OLLAMA_HOST}/api/generate"
    body = {
        "model": OLLAMA_MODEL,
        "prompt": (
            "You are a helpful assistant. Write a concise, professional LinkedIn post based on the topic below.\n"
            "- Tone: informative, friendly, and actionable.\n"
            "- Include a short hook, 2-4 bullet insights, and a call to engage.\n"
            "- Avoid hashtags except 1-2 relevant ones at the end.\n\n"
            f"Topic: {prompt.strip()}\n\nPost:"
        ),
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        },
        "stream": False,
    }

    try:
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response") or data.get("text") or ""
        text = text.strip()
        if not text:
            raise ValueError("Empty response from model")
        return text
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}")


