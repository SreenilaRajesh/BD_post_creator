from __future__ import annotations

import requests
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
            with gr.Column(visible=True) as step1:
                gr.Markdown("### Do you have any source")
                url = gr.Textbox(label="Web URL", placeholder="https://example.com")
                upload = gr.File(label="Upload PDF", file_count="single", file_types=[".pdf"])
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next", variant="primary")

            with gr.Column(visible=False) as step2:
                gr.Markdown("### Next Step")
                results_md = gr.Markdown("")

        start_btn.click(_show, inputs=None, outputs=modal)
        skip_btn.click(on_skip, inputs=None, outputs=[step1, step2, results_md])
        next_btn.click(on_next, inputs=[url, upload], outputs=[step1, step2, results_md])

    return demo


