# Copilot Instructions for this repo

- This repo is a Streamlit-based RAG demo built around `app.py` and `check.py`.
- `app.py` is the main interactive application for document upload, chunking, vector search, and conversational QA.
- `check.py` is the more advanced evaluation variant: it adds parent/child chunk metadata, a BM25 hybrid retrieval path, and richer PDF/table/image extraction.

## Key concepts

- The system depends on a single external API key: `DASHSCOPE_API_KEY`.
- Both apps use `OpenAI(api_key=..., base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")` and local models from `sentence-transformers` and `chromadb`.
- `build_index()` always clears the existing Chromadb collection before indexing new data.
- `app.py` uses punctuation-based sentence chunking and `CrossEncoder` reranking.
- `check.py` also stores `parent` metadata with each chunk, then combines Chroma vector retrieval and `BM25Okapi` re-ranking before final answer generation.
- `evaluate_answer()` in both apps expects the judge model to emit strict JSON and uses regexp fallback to parse the returned object.

## Important patterns

- Document parsing is the core data flow: upload → `process_uploaded_file()` → `extract_text_from_file()` → text normalization → chunk creation → `build_index()`.
- Image support is explicit: `jpg`, `jpeg`, `png` are handled with `Pillow` + `pytesseract` in `app.py`.
- `check.py` extends PDF extraction with table markdown conversion and VLM image analysis via `qwen-vl-max`.
- Session state is critical for Streamlit flows; both apps keep `st.session_state['chunks']`, `['messages']`, and evaluation cases in state.

## Dev workflow

- Use the provided virtual environment: `& .venv\Scripts\Activate.ps1` on Windows.
- Install dependencies with `pip install -r requirements.txt`.
- Run the UI apps with Streamlit:
  - `streamlit run app.py`
  - `streamlit run check.py`
- Set `DASHSCOPE_API_KEY` in a `.env` file or the environment before launching.

## Project-specific notes

- The repo is not organized as a package; there is no `src/` folder or separate module imports.
- Most logic lives in the top-level Python script files.
- The app uses a hardcoded local Tesseract path for OCR. If Tesseract is missing, image parsing will fail.
- Avoid changing the prompt structure dramatically without preserving `system` and `user` role semantics in `client_ai.chat.completions.create()`.

## Files to inspect first

- `app.py` — main RAG UI and simple document ingestion.
- `check.py` — advanced evaluation pipeline, hybrid retrieval, and image/VLM extraction.
- `requirements.txt` — dependency list for reproducing the environment.
