from __future__ import annotations

import requests
import shutil
import time
from pathlib import Path
import zipfile
import gradio as gr


def _show():
    return gr.update(visible=True)


def _hide():
    return gr.update(visible=False)


def _step_nav(show_step1: bool, show_step2: bool, message: str = ""):
    return gr.update(visible=show_step1), gr.update(visible=show_step2), message


API_BASE = "http://127.0.0.1:5454"


def on_next(url_value, upload_value):
    messages = []
    try:
        has_url = bool(url_value and str(url_value).strip())
        file_path = None
        if upload_value is not None:
            try:
                file_path = getattr(upload_value, "name", None) or str(upload_value)
            except Exception:
                file_path = None

        # 1) If web url is present
        if has_url:
            try:
                r = requests.post(f"{API_BASE}/parse_web_url", json={"url": str(url_value).strip()}, timeout=120)
                if r.ok:
                    messages.append("Parsed web URL successfully.")
                else:
                    messages.append(f"Web URL parse failed: {r.status_code}")
            except Exception as exc:
                messages.append(f"Web URL parse error: {exc}")

        # 2) If document is present
        if file_path:
            try:
                r = requests.post(f"{API_BASE}/parse_document", json={"file_path": file_path}, timeout=180)
                if r.ok:
                    messages.append("Parsed document successfully.")
                else:
                    messages.append(f"Document parse failed: {r.status_code}")
            except Exception as exc:
                messages.append(f"Document parse error: {exc}")

        # 4) If none
        if not has_url and not file_path:
            messages.append("No inputs provided. Skipping API calls.")

    except Exception as exc:
        messages.append(f"Unexpected error: {exc}")

    return _step_nav(False, True, "\n".join(messages))


def on_skip():
    return _step_nav(False, True, "Skipped inputs.")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AIPost") as demo:
        gr.Markdown("# **AIPost**")
        gr.Markdown("## Build your content with AI")

        start_btn = gr.Button("Start Building!", variant="primary")

        with gr.Group(visible=False) as modal:
            images_state = gr.State([])
            text_state = gr.State("")
            with gr.Column(visible=True) as step1:
                gr.Markdown("### Step 1: Do you have any source")
                url = gr.Textbox(label="Web URL", placeholder="https://example.com")
                upload = gr.File(label="Upload PDF", file_count="single", file_types=[".pdf"])
                with gr.Row():
                    step1_back_btn = gr.Button("Back")
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next", variant="primary")

            with gr.Column(visible=False) as step2:
                gr.Markdown("### Step 2: Text Generation")
                results_md = gr.Markdown("")

                text_option = gr.Radio(
                    label="Choose an option",
                    choices=[
                        "Generate from Prompt",
                        "Use Text from Source Content",
                    ],
                    value="Generate from Prompt",
                )

                with gr.Column(visible=True) as text_prompt_col:
                    text_prompt_tb = gr.Textbox(
                        label="Prompt or topic",
                        placeholder="Enter a topic or prompt...",
                    )

                with gr.Row():
                    step2_back_btn = gr.Button("Back")
                    step2_skip_btn = gr.Button("Skip")
                    step2_next_btn = gr.Button("Next", variant="primary")

            with gr.Column(visible=False) as step3:
                gr.Markdown("### Step 3: Image Upload / Generate")
                img_results_md = gr.Markdown("")

                img_option = gr.Radio(
                    label="Choose an option",
                    choices=[
                        "Upload Image",
                        "Generate from Prompt",
                        "Use Images from Source Content",
                    ],
                    value="Upload Image",
                )

                with gr.Column(visible=True) as upload_col:
                    image_uploader = gr.Image(
                        label="Drop image here or click to upload",
                        type="filepath",
                    )
                    upload_status = gr.Markdown("")

                with gr.Column(visible=False) as generate_col:
                    prompt_tb = gr.Textbox(
                        label="Prompt to generate image",
                        placeholder="Describe the image you want...",
                    )

                with gr.Row():
                    step3_back_btn = gr.Button("Back")
                    step3_finish_btn = gr.Button("Finish", variant="primary")

        start_btn.click(_show, inputs=None, outputs=modal)
        def on_step1_back():
            return gr.update(visible=False)
        step1_back_btn.click(on_step1_back, inputs=None, outputs=modal)
        skip_btn.click(on_skip, inputs=None, outputs=[step1, step2, results_md])
        next_btn.click(on_next, inputs=[url, upload], outputs=[step1, step2, results_md])

        def on_option_change(choice: str):
            return (
                gr.update(visible=choice == "Upload Image"),
                gr.update(visible=choice == "Generate from Prompt"),
            )

        img_option.change(on_option_change, inputs=img_option, outputs=[upload_col, generate_col])

        def on_text_option_change(choice: str):
            # Text box is common to both options; keep it visible
            return gr.update(visible=True)

        text_option.change(on_text_option_change, inputs=text_option, outputs=text_prompt_col)

        # Image upload handling: store locally under user_images/
        def on_image_upload(image_path: str):
            try:
                if not image_path:
                    return "No image provided."
                src = Path(image_path)
                if not src.exists():
                    return "Uploaded file not found."
                dest_dir = Path(__file__).resolve().parent.parent / "user_images"
                dest_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                dest = dest_dir / f"{ts}_{src.name}"
                shutil.copy2(src, dest)
                try:
                    gr.Info(f"Image saved: {dest}")
                except Exception:
                    pass
                return f"Saved to: {dest}"
            except Exception as exc:
                try:
                    gr.Error(f"Image save error: {exc}")
                except Exception:
                    pass
                return f"Image save error: {exc}"

        image_uploader.change(on_image_upload, inputs=image_uploader, outputs=upload_status)

        def on_step2_skip():
            # Hide step2, show step3
            return gr.update(visible=False), gr.update(visible=True)

        def on_step2_next(choice: str, prompt: str, current_text: str):
            new_text = current_text or ""
            try:
                topic = (prompt or "").strip()
                if choice == "Generate from Prompt":
                    if not topic:
                        try:
                            gr.Warning("Please enter a prompt or choose another option.")
                        except Exception:
                            pass
                        return gr.update(visible=True), gr.update(visible=False), new_text
                    r = requests.post(f"{API_BASE}/generate_text", json={"prompt": topic}, timeout=120)
                    if r.ok:
                        new_text = (r.json() or {}).get("content", "") or new_text
                        try:
                            gr.Info("Generated text from prompt.")
                        except Exception:
                            pass
                    else:
                        try:
                            gr.Warning(f"Text generation failed: {r.status_code}")
                        except Exception:
                            pass
                else:
                    r = requests.post(f"{API_BASE}/search_text", json={"query": topic}, timeout=120)
                    if r.ok:
                        items = (r.json() or {}).get("results", [])
                        parts = [i.get("payload", "") for i in items if isinstance(i, dict)]
                        new_text = "\n\n".join([p for p in parts if p]) or new_text
                        try:
                            gr.Info("Collected text from source content.")
                        except Exception:
                            pass
                    else:
                        try:
                            gr.Warning(f"Search text failed: {r.status_code}")
                        except Exception:
                            pass
            except Exception as exc:
                try:
                    gr.Error(f"Step 2 error: {exc}")
                except Exception:
                    pass
            # Hide step2, show step3 and persist new_text
            return gr.update(visible=False), gr.update(visible=True), new_text

        def on_step2_back():
            # Hide step2, show step1
            return gr.update(visible=False), gr.update(visible=True)
        step2_back_btn.click(on_step2_back, inputs=None, outputs=[step2, step1])
        step2_skip_btn.click(on_step2_skip, inputs=None, outputs=[step2, step3])
        step2_next_btn.click(on_step2_next, inputs=[text_option, text_prompt_tb, text_state], outputs=[step2, step3, text_state])

        # Extend flow: After finishing step3, go to Review (Step 4) with slideshow tools
        with gr.Column(visible=False) as step4:
            gr.Markdown("### Step 4: Review & Modify - Slideshow/Post Series")
            gr.Markdown("Reorder images and edit your post text.")
            gallery = gr.Gallery(label="Slides", value=[], columns=4, height=200)
            select_idx = gr.Dropdown(label="Select slide index", choices=[], value=None)
            with gr.Row():
                move_up_btn = gr.Button("Move Up")
                move_down_btn = gr.Button("Move Down")
            post_text = gr.Textbox(label="Post text", lines=8, placeholder="Edit your LinkedIn post here...")
            save_text_btn = gr.Button("Save Text")
            with gr.Row():
                step4_back_btn = gr.Button("Back")
                step4_next_btn = gr.Button("Next", variant="primary")

        with gr.Column(visible=False) as step5:
            gr.Markdown("### Step 5: Preview & Download")
            preview_gallery = gr.Gallery(label="Preview", value=[], columns=3, height=240)
            preview_text = gr.Markdown("")
            download_file = gr.File(label="Download composed post (ZIP)")
            with gr.Row():
                step5_back_btn = gr.Button("Back")
                step5_finish_btn = gr.Button("Finish", variant="primary")

        def on_step3_finish(choice: str, image_path: str, prompt: str, current_images: list):
            new_images = list(current_images or [])
            try:
                if choice == "Upload Image":
                    if image_path:
                        r = requests.post(f"{API_BASE}/store_uploaded_image", json={"file_path": image_path}, timeout=60)
                        if r.ok:
                            saved = (r.json() or {}).get("saved_path")
                            if saved and saved not in new_images:
                                new_images.append(saved)
                            try:
                                gr.Info("Image uploaded and stored.")
                            except Exception:
                                pass
                        else:
                            try:
                                gr.Warning(f"Upload store failed: {r.status_code}")
                            except Exception:
                                pass
                    else:
                        try:
                            gr.Warning("Please upload an image.")
                        except Exception:
                            pass
                elif choice == "Generate from Prompt":
                    topic = (prompt or "").strip()
                    if not topic:
                        try:
                            gr.Warning("Please enter a prompt to generate image.")
                        except Exception:
                            pass
                    else:
                        r = requests.post(f"{API_BASE}/generate_image", json={"prompt": topic}, timeout=180)
                        if r.ok:
                            img_path = (r.json() or {}).get("image_path")
                            if img_path and img_path not in new_images:
                                new_images.append(img_path)
                            try:
                                gr.Info("Generated image.")
                            except Exception:
                                pass
                        else:
                            try:
                                gr.Warning(f"Image generation failed: {r.status_code}")
                            except Exception:
                                pass
                else:
                    # Use images from source content (session search)
                    q = (prompt or "image").strip()
                    r = requests.post(f"{API_BASE}/search_images", json={"query": q}, timeout=120)
                    if r.ok:
                        items = (r.json() or {}).get("results", [])
                        paths = [i.get("payload") for i in items if isinstance(i, dict)]
                        for p in paths:
                            if p and p not in new_images:
                                new_images.append(p)
                        try:
                            gr.Info("Collected images from source content.")
                        except Exception:
                            pass
                    else:
                        try:
                            gr.Warning(f"Search images failed: {r.status_code}")
                        except Exception:
                            pass
            except Exception as exc:
                try:
                    gr.Error(f"Step 3 error: {exc}")
                except Exception:
                    pass

            # Prepare Step 4 UI values
            choices = [str(i + 1) for i in range(len(new_images))]
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                new_images,
                gr.update(value=new_images),
                gr.update(choices=choices, value=(choices[-1] if choices else None)),
            )

        def on_step3_back():
            # Hide step3, show step2
            return gr.update(visible=False), gr.update(visible=True)
        step3_back_btn.click(on_step3_back, inputs=None, outputs=[step3, step2])
        step3_finish_btn.click(
            on_step3_finish,
            inputs=[img_option, image_uploader, prompt_tb, images_state],
            outputs=[step3, step4, images_state, gallery, select_idx],
        )

        # Step 4 helpers
        def move_image(images: list, index_str: str, direction: str):
            arr = list(images or [])
            if not arr or not index_str:
                return arr, gr.update(value=arr), gr.update(choices=[str(i + 1) for i in range(len(arr))], value=(index_str or None))
            idx = int(index_str) - 1
            if idx < 0 or idx >= len(arr):
                return arr, gr.update(value=arr), gr.update(choices=[str(i + 1) for i in range(len(arr))], value=index_str)
            new_idx = idx - 1 if direction == "up" else idx + 1
            if 0 <= new_idx < len(arr):
                arr[idx], arr[new_idx] = arr[new_idx], arr[idx]
            choices = [str(i + 1) for i in range(len(arr))]
            return arr, gr.update(value=arr), gr.update(choices=choices, value=str(new_idx + 1))

        move_up_btn.click(lambda imgs, idx: move_image(imgs, idx, "up"), inputs=[images_state, select_idx], outputs=[images_state, gallery, select_idx])
        move_down_btn.click(lambda imgs, idx: move_image(imgs, idx, "down"), inputs=[images_state, select_idx], outputs=[images_state, gallery, select_idx])

        def save_text(text: str):
            try:
                gr.Info("Text saved.")
            except Exception:
                pass
            return text
        save_text_btn.click(save_text, inputs=post_text, outputs=text_state)

        def on_step4_back():
            return gr.update(visible=False), gr.update(visible=True)
        step4_back_btn.click(on_step4_back, inputs=None, outputs=[step4, step3])

        def on_step4_next(images: list, text: str):
            md = f"### Post Preview\n\n{text}" if text else ""
            return gr.update(visible=False), gr.update(visible=True), gr.update(value=images or []), gr.update(value=md)
        step4_next_btn.click(on_step4_next, inputs=[images_state, text_state], outputs=[step4, step5, preview_gallery, preview_text])

        # Step 5
        def on_step5_back():
            return gr.update(visible=False), gr.update(visible=True)
        step5_back_btn.click(on_step5_back, inputs=None, outputs=[step5, step4])

        def make_zip(images: list):
            if not images:
                try:
                    gr.Warning("No images to download.")
                except Exception:
                    pass
                return None
            downloads_dir = Path(__file__).resolve().parent.parent / "downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            zip_path = downloads_dir / f"post_{ts}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in images:
                    try:
                        fp = Path(p)
                        if fp.exists():
                            zf.write(fp, arcname=fp.name)
                    except Exception:
                        continue
            return str(zip_path)
        step5_finish_btn.click(make_zip, inputs=images_state, outputs=download_file)

    return demo


