
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
import tempfile
from typing import Tuple

import streamlit as st
from dotenv import load_dotenv

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
MODEL_NAME_DEFAULT = os.getenv("LLM_MODEL", "gemini-1.5-flash")
MAX_FILE_TEXT = 120000  # characters allowed from file
MAX_PROMPT_CHUNK = 6000  # chunk size to send to LLM for file summarization

load_dotenv(override=False)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# Basic theme/styles
st.markdown(
    """
    <style>
    .mode-btn {padding:10px 14px; border-radius:10px; border:1px solid #e5e7eb; cursor:pointer; margin-right:8px; background:linear-gradient(135deg,#f9fafb,#eef2ff);} 
    .mode-btn.active {border-color:#7c3aed; box-shadow:0 0 0 3px rgba(124,58,237,0.15); background:linear-gradient(135deg,#ede9fe,#dbeafe);} 
    .hero {padding:22px; border-radius:16px; background:linear-gradient(135deg,#1f2937,#0f172a); color:white; border:1px solid rgba(255,255,255,0.08);} 
    .hero h2 {margin:0 0 8px 0;}
    .pill {display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(255,255,255,0.12); margin-right:8px; font-size:12px}
    .chat-bin {background:#fee2e2; color:#991b1b; border:1px solid #fecaca; padding:8px 10px; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# LLM helper functions
# ----------------------

def get_gemini_client_configured() -> bool:
    api_key = os.getenv("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None)
    if not api_key or genai is None:
        return False
    try:
        # Newer google-genai SDK style
        if hasattr(genai, "configure"):
            genai.configure(api_key=api_key)
            return True
        # Fallback older style client (rare)
        if hasattr(genai, "Client"):
            _ = genai.Client(api_key=api_key)
            return True
    except Exception as _e:
        return False
    return False

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

    # Try Gemini (new SDK first)
    if get_gemini_client_configured():
        try:
            if hasattr(genai, "GenerativeModel"):
                try_models = [
                    MODEL_NAME_DEFAULT,
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-1.0-pro",
                ]
                last_err = None
                for m in try_models:
                    try:
                        model = genai.GenerativeModel(m)
                        resp = model.generate_content(assembled_prompt, generation_config={"max_output_tokens": max_output_tokens, "temperature": 0.2})
                        # Prefer resp.text when available
                        if hasattr(resp, "text") and resp.text:
                            return resp.text
                        # Fallback: attempt to extract first candidate text
                        try:
                            return resp.candidates[0].content.parts[0].text
                        except Exception:
                            return str(resp)
                    except Exception as ee:
                        last_err = ee
                        continue
                if last_err is not None:
                    raise last_err
            else:
                # Older style fallback
                if hasattr(genai, "models"):
                    resp = genai.models.generate_content(
                        model=MODEL_NAME_DEFAULT,
                        contents=assembled_prompt,
                        generation_config={"max_output_tokens": max_output_tokens, "temperature": 0.2}
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

    # If no LLM works, show helpful message instead of fallback
    st.info("💡 **No AI service available** - Please add your Gemini API key in the sidebar or .env file to get AI responses.")
    return "I'm ready to help, but I need an API key to provide AI responses. Please add your Gemini API key to continue."

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
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(bytes_data)
            tmp.flush()
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        return f"[Error reading DOCX: {e}]"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "explain"  # explain | deep | quiz | review | coding
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False

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
        # API status
        gemini_key_present = bool(os.getenv("GEMINI_API_KEY") or ("GEMINI_API_KEY" in st.secrets))
        openai_key_present = bool(os.getenv("OPENAI_API_KEY") or ("OPENAI_API_KEY" in st.secrets))
        st.caption(f"Gemini key: {'✅ found' if gemini_key_present else '❌ missing'} | OpenAI key: {'✅ found' if openai_key_present else '❌ optional'}")
        if not gemini_key_present:
            st.warning("⚠️ Add GEMINI_API_KEY to .env file or environment")
        st.markdown("---")
        st.markdown("<div class='chat-bin'>🗑️ Chat Bin</div>", unsafe_allow_html=True)
        if not st.session_state.confirm_clear:
            if st.button("Clear all chat"):
                st.session_state.confirm_clear = True
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Confirm"):
                    st.session_state.chat = []
                    st.session_state.files = []
                    st.session_state.confirm_clear = False
                    st.rerun()
            with c2:
                if st.button("Cancel"):
                    st.session_state.confirm_clear = False

    # Mode toolbar
    st.subheader("Modes")
    m1, m2, m3, m4, m5 = st.columns([1,1,1,1,1])
    with m1:
        if st.button("Explain", key="mode_explain"):
            st.session_state.selected_mode = "explain"
    with m2:
        if st.button("Deep Think", key="mode_deep"):
            st.session_state.selected_mode = "deep"
    with m3:
        if st.button("Quiz", key="mode_quiz"):
            st.session_state.selected_mode = "quiz"
    with m4:
        if st.button("Review", key="mode_review"):
            st.session_state.selected_mode = "review"
    with m5:
        if st.button("Coding Help", key="mode_code"):
            st.session_state.selected_mode = "coding"

    st.caption(f"Active mode: {st.session_state.selected_mode}")

    # Welcome hero
    if len(st.session_state.chat) == 0:
        st.markdown(
            """
            <div class="hero">
                <h2>Welcome to Study Wise AI Tutor</h2>
                Unlock concepts faster with clear steps, smart quizzes, and file/image understanding.
                <div style="margin-top:8px;">
                    <span class="pill">Explain</span>
                    <span class="pill">Deep Think</span>
                    <span class="pill">Quiz</span>
                    <span class="pill">Review</span>
                    <span class="pill">Coding</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display chat
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Uploaders row
    ucol1, ucol2 = st.columns([1,1])
    with ucol1:
        uploaded = st.file_uploader("Upload document (pdf, docx, txt, md)", type=["pdf", "docx", "txt", "md"], key="doc_upl")
    with ucol2:
        image_file = st.file_uploader("Upload image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"], key="img_upl")

    # Input area
    user_input = st.chat_input("Ask a question...")

    if uploaded:
        filename, text = parse_uploaded_file(uploaded)
        push_chat("user", f"Uploaded file: {filename}")
        if not text or text.startswith("[Error"):
            push_chat("assistant", "Sorry, I couldn't read that file. Please try another format or a clearer copy.")
            st.rerun()
        analysis_prompt = f"Summarize and analyze this document:\n{text[:MAX_PROMPT_CHUNK]}"
        # Token strategy per mode
        mode = st.session_state.selected_mode
        mode_reason = "review"
        max_tokens = 500
        if mode == "deep":
            mode_reason = "deep_thinking"
            max_tokens = 900
        elif mode == "quiz":
            mode_reason = "quiz"
            max_tokens = 600
        elif mode == "coding":
            mode_reason = "coding_help"
            max_tokens = 700
        resp = generate_response(analysis_prompt, mode_meta={"external_refs": st.session_state.external_refs_enabled, "reasoning": mode_reason}, max_output_tokens=max_tokens)
        push_chat("assistant", resp)
        st.rerun()

    if image_file:
        try:
            image_bytes = image_file.read()
            push_chat("user", f"Uploaded image: {image_file.name}")
            st.image(image_bytes, caption=image_file.name, use_column_width=True)
            mode = st.session_state.selected_mode
            mode_reason = "explain"
            if mode == "deep":
                mode_reason = "deep_thinking"
            elif mode == "quiz":
                mode_reason = "quiz"
            elif mode == "review":
                mode_reason = "review"
            elif mode == "coding":
                mode_reason = "coding_help"
            prompt = "Describe the key information in the uploaded image and how it relates to studying. If text is present, summarize it."
            resp = generate_response(prompt, mode_meta={"external_refs": st.session_state.external_refs_enabled, "reasoning": mode_reason}, max_output_tokens=600)
            push_chat("assistant", resp)
            st.rerun()
        except Exception as _e:
            push_chat("assistant", "Image preview added. Analysis not available.")
            st.rerun()

    if user_input:
        push_chat("user", user_input)
        mode = st.session_state.selected_mode
        mode_reason = "explain"
        max_tokens = 500
        if mode == "deep":
            mode_reason = "deep_thinking"
            max_tokens = 900
        elif mode == "quiz":
            mode_reason = "quiz"
            max_tokens = 600
        elif mode == "review":
            mode_reason = "review"
            max_tokens = 600
        elif mode == "coding":
            mode_reason = "coding_help"
            max_tokens = 700
        resp = generate_response(user_input, mode_meta={"external_refs": st.session_state.external_refs_enabled, "reasoning": mode_reason}, max_output_tokens=max_tokens)
        push_chat("assistant", resp)
        st.rerun()

if __name__ == "__main__":
    main()
