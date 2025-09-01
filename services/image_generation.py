from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionPipeline
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    StableDiffusionPipeline = None  # type: ignore


_PIPELINE = None


def _images_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "generated_images"
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_sd_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5"):
    global _PIPELINE
    if StableDiffusionPipeline is None or torch is None:
        raise HTTPException(status_code=500, detail="diffusers/torch not installed")

    if _PIPELINE is not None:
        return _PIPELINE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if (device == "cuda") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    _PIPELINE = pipe
    return _PIPELINE


def generate_image(prompt: str, num_inference_steps: int = 25, guidance_scale: float = 7.5, seed: Optional[int] = None) -> str:
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    pipe = load_sd_pipeline()

    generator = None
    if seed is not None and torch is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))

    images = pipe(
        prompt=prompt.strip(),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    ).images

    if not images:
        raise HTTPException(status_code=500, detail="Image generation failed")

    image: Image.Image = images[0]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = _images_dir() / f"gen_{ts}.png"
    image.save(out_path)
    return str(out_path.resolve())


