Interview Case: Gen AI Social Media Post Creator (PoC)
Objective
Develop a functional proof-of-concept (PoC) web app for creating LinkedIn-style social
media image posts using generative AI. The solution must use Streamlit, Gradio, or
Dash for the UI, and Python for backend logic. All AI components should use open-source
models (e.g., Llama2, Mistral, BLIP2, Stable Diffusion, etc.). The app should
support multimodal RAG (retrieval-augmented generation from PDFs, images, and web
content) and include a web search agent powered by MCP or an agentic AI framework
(e.g., Crew AI, AutoGen, LangGraph Agents, Langchain etc.).
Note: Models/Frameworks mentioned are just for reference and any open source
model or framework can be used.
Core Features
1. Image Upload & Creation
• Users can upload their own images or generate images via an AI prompt (using
open-source models like Stable Diffusion, with proper code and placeholders if no
access).
• Optionally, users can generate images based on content extracted from PDFs or
web search results.
2. Text Boxes with AI Assistance
• Each text box can be filled by:
o AI-generated text (using a prompt, PDF content, or web search agent).
3. Multimodal RAG Integration
• Users can upload a PDF (e.g., resume, brochure) containing tables, charts,images,
and text.
• System extracts relevant content from the PDF using RAG (LangChain, LlamaIndex,
Haystack, etc.).
• Users can provide a web URL or search query; the system uses a web search
agent (MCP or agentic AI framework) to retrieve and summarize relevant content.
• All retrieved content (text, images) can be used for post/caption generation.

4. Slideshow/Post Series
• Support for multiple slides/images in a post.
• Each slide/image allows for AI-generated or user-generated captions and
descriptions.
• Allow rearranging slides and previewing the post as a sequence.
5. Preview & Download
• Preview the post as it would appear on LinkedIn.
• Option to download the final composed post (single image or slideshow as images).








social_post_creator/
│
├── app.py                  # Streamlit/Gradio main entry point
│
├── ui/                     # UI Layer
│   ├── upload_page.py      # Upload center (PDF, URL, images)
│   ├── builder_page.py     # Post/slide builder with AI text & image
│   ├── preview_page.py     # LinkedIn-style preview + download
│
├── services/               # Backend AI services
│   ├── text_generator.py   # Wrapper for LLM (Mistral/Llama2 + LangChain)
│   ├── image_generator.py  # Stable Diffusion pipeline
│   ├── pdf_extractor.py    # Extract text, images, tables
│   ├── image_extractor.py  # BLIP2 captioning
│   ├── web_agent.py        # LangChain Agent for web search & summarization
│   ├── rag_engine.py       # Vector DB + retrieval + RAG pipeline
│
├── utils/                  # Helpers
│   ├── config.py           # Model paths, API keys (if any)
│   ├── session.py          # State management
│   ├── formatting.py       # Text/image formatting utilities
│
├── data/                   # Uploaded/Generated files (temp)
│
└── requirements.txt

---

API Controller (FastAPI)

This repo includes a simple `controller.py` that exposes two endpoints to parse web pages and PDFs into separate text and image files.

Install

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```

Run

```bash
uvicorn controller:app --reload --host 0.0.0.0 --port 8000
```

Endpoints

1) POST `/parse_web_url`

Body

```json
{ "url": "https://example.com" }
```

Response

```json
{
  "text_file": "C:/.../web_inputs/20240101_120000/content.txt",
  "image_files": ["C:/.../web_inputs/20240101_120000/images/image_001.jpg", "..."]
}
```

Behavior

- Saves fetched text to `web_inputs/<timestamp>/content.txt`
- Saves images to `web_inputs/<timestamp>/images/*.jpg`

2) POST `/parse_document`

Body

```json
{ "file_path": "C:/path/to/file.pdf" }
```

Response

```json
{
  "text_file": "C:/.../doc_inputs/20240101_120000/content.txt",
  "image_files": ["C:/.../doc_inputs/20240101_120000/images/image_001.jpg", "..."]
}
```

Behavior

- Copies the input PDF to `input_documents/<timestamp>/`
- Extracts text with `pdfplumber` and images with `PyMuPDF` into `doc_inputs/<timestamp>/`

Notes

- Keep functions small and straightforward; no agent framework is required for this PoC.
- For large or complex pages/PDFs, you can add chunking or rate limits later if needed.