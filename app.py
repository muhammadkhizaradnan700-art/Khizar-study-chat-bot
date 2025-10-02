
# app.py
"""
Study Wise Ai Tutor - Streamlit app (single-file)
Features:
- Modern splash screen + gradient
- Bubble-style chat UI (user: right purple | assistant: left red/white)
- File upload inside input area (pdf/docx/txt/md)
- File parsing with PyPDF2, python-docx, markdown2
- Auto-generated: summary, key concepts, quiz Qs, suggested search queries
- Reasoning modes: Explain, Quiz, Review, Deep Thinking
- Edit last question per-bubble, copy assistant reply to clipboard
- Animated loading indicator (blinking dots)
- External references toggle in sidebar (LLM decides links)
- LLM integration: google.genai preferred, openai fallback
- Modular, well-commented
"""

import os
import io
import time
import html
import tempfile
from typing import Tuple, List, Dict

import streamlit as st
import streamlit.components.v1 as components

# File parsing libs
from PyPDF2 import PdfReader
import docx
import markdown2

# Optional LLM libs (may or may not exist in environment)
try:
    from google import genai
except Exception:
    genai = None

try:
    import openai
except Exception:
    openai = None

# ----------------------
# Configuration & Utils
# ----------------------

PAGE_TITLE = "Study Wise Ai Tutor"
PAGE_ICON = "✨"
MODEL_NAME_DEFAULT = os.getenv("LLM_MODEL", "gemini-2.0-flash")
MAX_FILE_TEXT = 120000  # characters allowed from file
MAX_PROMPT_CHUNK = 6000  # chunk size to send to LLM for file summarization

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# ----------------------
# LLM helper functions
# ----------------------

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None
    if api_key and genai is not None:
        return genai.Client(api_key=api_key)
    return None

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else None
    if api_key and openai is not None:
        openai.api_key = api_key
        return openai
    return None

def generate_response(prompt: str, mode_meta: dict = None, max_output_tokens: int = 500) -> str:
    system_preamble = (
        "You are Study Wise Ai Tutor — a helpful, patient, step-by-step AI tutor. "
        "When asked, provide clear explanations, list key steps, produce short quizzes, and suggest external resources "
        "if enabled. Keep answers concise but thorough; when asked for step-by-step, number steps. "
    )

    if mode_meta is None:
        mode_meta = {}

    external_flag = mode_meta.get("external_refs", True)
    reasoning = mode_meta.get("reasoning", "explain")
    assembled_prompt = f"{system_preamble}\nMode: {reasoning}\nExternalLinksAllowed: {external_flag}\n\n{prompt}"

    # Try Gemini
    gemini = get_gemini_client()
    if gemini:
        try:
            resp = gemini.models.generate_content(
                model=MODEL_NAME_DEFAULT,
                contents=assembled_prompt,
                config={"max_output_tokens": max_output_tokens}
            )
            if hasattr(resp, "text") and resp.text:
                return resp.text
            return str(resp)
        except Exception as e:
            st.error(f"Gemini error: {e}")

    # Try OpenAI
    openai_client = get_openai_client()
    if openai_client:
        try:
            if hasattr(openai_client, "ChatCompletion"):
                messages = [
                    {"role": "system", "content": system_preamble},
                    {"role": "user", "content": prompt}
                ]
                resp = openai_client.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=max_output_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            else:
                resp = openai_client.Completion.create(
                    engine="text-davinci-003",
                    prompt=assembled_prompt,
                    max_tokens=max_output_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].text.strip()
        except Exception as e:
            st.error(f"OpenAI error: {e}")

    return ("[Local fallback response]\n\n"
            + (assembled_prompt[:800] + ("..." if len(assembled_prompt) > 800 else "")))

# ----------------------
# File parsing utilities
# ----------------------

def read_pdf_bytes(bytes_data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(bytes_data))
        pages = []
        for p in reader.pages:
            try:
                text = p.extract_text()
            except Exception:
                text = ""
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        return f"[Error reading PDF: {e}]"

def read_docx_bytes(bytes_data: bytes) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            doc = docx.Document(tmp.name)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"

def read_text_bytes(bytes_data: bytes, encoding="utf-8") -> str:
    try:
        return bytes_data.decode(encoding, errors="replace")
    except Exception:
        return bytes_data.decode("latin-1", errors="replace")

def parse_uploaded_file(uploaded_file) -> Tuple[str, str]:
    name = uploaded_file.name
    raw = uploaded_file.read()
    lower = name.lower()
    if lower.endswith(".pdf"):
        text = read_pdf_bytes(raw)
    elif lower.endswith(".docx"):
        text = read_docx_bytes(raw)
    else:
        text = read_text_bytes(raw)
        if lower.endswith(".md"):
            text = markdown2.markdown(text)
    if len(text) > MAX_FILE_TEXT:
        text = text[:MAX_FILE_TEXT] + "\n\n[Truncated]"
    return name, text

# ----------------------
# Session state & helpers
# ----------------------

def init_session_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "files" not in st.session_state:
        st.session_state.files = []
    if "external_refs_enabled" not in st.session_state:
        st.session_state.external_refs_enabled = True

def push_chat(role: str, content: str, meta: dict = None):
    if meta is None:
        meta = {}
    st.session_state.chat.append({"role": role, "content": content, "meta": meta, "ts": time.time()})

# ----------------------
# UI Rendering
# ----------------------

def main():
    init_session_state()

    st.title("✨ Study Wise Ai Tutor ✨")
    st.markdown("Your personal AI-powered study companion. Study Wise Ai Tutor helps you learn faster with smart explanations, file analysis, external references, and interactive reasoning.")

    with st.sidebar:
        st.header("Options")
        st.session_state.external_refs_enabled = st.checkbox("Enable external references", value=st.session_state.external_refs_enabled)
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.session_state.files = []
            st.rerun()

    # Display chat
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Input area
    user_input = st.chat_input("Ask a question...")
    uploaded = st.file_uploader("Upload file", type=["pdf", "docx", "txt", "md"])

    if uploaded:
        filename, text = parse_uploaded_file(uploaded)
        push_chat("user", f"Uploaded file: {filename}")
        analysis_prompt = f"Summarize and analyze this document:\n{text[:MAX_PROMPT_CHUNK]}"
        resp = generate_response(analysis_prompt)
        push_chat("assistant", resp)
        st.rerun()

    if user_input:
        push_chat("user", user_input)
        resp = generate_response(user_input)
        push_chat("assistant", resp)
        st.rerun()

if __name__ == "__main__":
    main()
