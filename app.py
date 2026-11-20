import datetime
import hashlib
import io
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import requests
import streamlit as st
from pypdf import PdfReader

# ── Suppress noisy HTTP logs ──────────────────────────────────────────────────
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="विवेचना एवं केस डायरी प्रणाली",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #061a35 0%, #0b2b52 48%, #071b38 100%);
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg,
            rgba(30,64,175,0.28) 0%,
            rgba(124,58,237,0.20) 50%,
            rgba(16,185,129,0.14) 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-title {
        font-size: 2.0rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 4px 0;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        margin-top: 10px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Chat bubbles ── */
    .chat-user {
        display: flex;
        justify-content: flex-end;
        margin: 12px 0;
    }
    .chat-user .bubble {
        background: linear-gradient(135deg, #3730a3, #4f46e5);
        color: #e0e7ff;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        max-width: 80%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 16px rgba(79,70,229,0.3);
    }
    .chat-bot {
        display: flex;
        justify-content: flex-start;
        margin: 12px 0;
    }
    .chat-bot .bubble {
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(99,102,241,0.25);
        color: #e2e8f0;
        border-radius: 18px 18px 18px 4px;
        padding: 18px 22px;
        max-width: 85%;
        font-size: 0.94rem;
        line-height: 1.85;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
        margin: 0 10px;
    }
    .avatar-user {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid rgba(99,102,241,0.4);
    }

    /* ── Source cards ── */
    .source-card {
        background: rgba(17,24,39,0.7);
        border: 1px solid rgba(52,211,153,0.2);
        border-left: 3px solid #34d399;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .source-card strong { color: #6ee7b7; }

    /* ── Warning & Guardrail box ── */
    .warn-box {
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.4);
        border-radius: 12px;
        padding: 14px 18px;
        color: #fcd34d;
        font-size: 0.90rem;
        margin: 10px 0;
        line-height: 1.6;
    }
    .guardrail-box {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 4px solid #ef4444;
        border-radius: 12px;
        padding: 14px 18px;
        color: #fca5a5;
        font-size: 0.92rem;
        margin: 10px 0;
        line-height: 1.65;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #34d399;
        font-weight: 500;
    }
    .status-dot {
        width: 8px; height: 8px;
        background: #34d399;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.8); }
    }

    /* ── Model provider pill ── */
    .model-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 3px 10px;
        font-size: 0.76rem;
        color: #a5b4fc;
        font-family: monospace;
        margin-top: 6px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79,70,229,0.5) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(10,14,26,0.95) !important;
        border-right: 1px solid rgba(99,102,241,0.15) !important;
    }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #c7d2fe !important; }

    /* ── Log box ── */
    .log-box {
        background: rgba(0,0,0,0.5);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 10px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.78rem;
        color: #64748b;
        max-height: 320px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("This PDF is password-protected.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[Page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def extract_docx_text(file_bytes: bytes) -> str:
    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as document:
        xml_content = document.read("word/document.xml")

    root = ET.fromstring(xml_content)
    paragraphs = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(
            node.text or "" for node in paragraph.iter(f"{namespace}t")
        ).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


# ── Session State Bootstrap ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "case_facts" not in st.session_state:
    st.session_state.case_facts = None
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "document_upload_fingerprint" not in st.session_state:
    st.session_state.document_upload_fingerprint = None
if "document_errors" not in st.session_state:
    st.session_state.document_errors = []


def reset_chat_session() -> None:
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.case_facts = None
    try:
        fetch_saved_sessions.clear()
    except Exception:
        pass


@st.cache_data(ttl=10, show_spinner=False)
def check_api_health(endpoint: str) -> bool:
    try:
        res = requests.get(f"{endpoint}/health", timeout=2)
        return res.status_code == 200 and res.json().get("status") == "ok"
    except requests.RequestException:
        return False


def get_api_headers() -> dict[str, str]:
    api_key = os.getenv("CHATBOT_API_KEY", "").strip()
    return {"X-API-Key": api_key} if api_key else {}


@st.cache_data(ttl=10, show_spinner=False)
def fetch_saved_sessions(endpoint: str) -> list[dict]:
    try:
        res = requests.get(
            f"{endpoint}/api/sessions",
            headers=get_api_headers(),
            timeout=2,
        )
        if res.status_code == 200:
            return res.json().get("sessions", [])
    except requests.RequestException:
        pass
    return []


def format_api_error(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text

    if not isinstance(body, dict):
        return response.text

    errors = body.get("errors")
    if isinstance(errors, list):
        details = []
        for error in errors:
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                field = error.get("field")
                message = error["message"]
                details.append(f"{field}: {message}" if isinstance(field, str) else message)
        if details:
            heading = body.get("message", "Request validation failed.")
            return f"{heading}\n\n" + "\n".join(details)

    detail = body.get("detail", body.get("message"))
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return json.dumps(detail, ensure_ascii=False)
    return response.text


def format_chat_as_markdown(session_id: str | None, case_facts: str | None, messages: list) -> str:
    """Formats current session and messages into a clean Case Diary Markdown document."""
    lines = [
        "# पुलिस विवेचना एवं केस डायरी अभिलेख (Investigation & Case Diary Record)",
        f"**सत्र पहचान (Session ID):** `{session_id or 'Not assigned yet'}`",
        f"**निर्यात दिनांक (Export Date):** {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    if case_facts:
        lines.extend([
            "## 📋 प्रारंभिक घटना विवरण / केस तथ्य (Case Facts)",
            case_facts.strip(),
            "",
            "---",
            "",
        ])
    lines.append("## 💬 विवेचना संवाद एवं विधिक विश्लेषण (Case Analysis History)\n")
    for msg in messages:
        role = "अन्वेषण अधिकारी (IO)" if msg.get("role") == "user" else "विधिक सहायक AI (Legal AI Assistant)"
        lines.append(f"### 🔹 {role}:")
        lines.append(msg.get("content", "").strip())
        if msg.get("sources"):
            lines.append("\n**संदर्शित विधिक संदर्भ दस्तावेज:**")
            for src in msg["sources"]:
                lines.append(f"- `[{src.get('idx')}]` {src.get('file')}: {src.get('snippet', '')}")
        lines.append("\n" + "-" * 40 + "\n")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚖️ Police Investigation & Case Diary System")
    st.markdown("*पुलिस विवेचना एवं केस डायरी प्रणाली*")
    st.divider()

    st.markdown("### 🔧 सिस्टम जानकारी")
    st.markdown(
        """
| घटक | मान |
|-----|-----|
| प्राथमिक AI मॉडल | `gemini-3.1-flash-lite` |
| बैकअप मॉडल | `gemini-3.5-flash` · Groq |
| एम्बेडिंग्स सर्च | `all-MiniLM-L6-v2` (FAISS) |
| दंडनीय अपराध संहिता | **BNS 2023** (धाराएं व दंड) |
| अन्वेषण व प्रक्रिया संहिता | **BNSS 2023** (प्रक्रिया एवं CD) |
| साक्ष्य अधिनियम | **BSB 2023** (डिजिटल साक्ष्य Sec 63) |
| आउटपुट क्षमता | **3,000 टोकन (पूर्ण रिपोर्ट · शून्य IPC)** |
| सुरक्षा दायरा | **केवल पुलिस विवेचना व केस डायरी** |
        """
    )
    st.divider()

    api_url = st.text_input(
        "Chat API URL",
        value=os.getenv("CHAT_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        help="Set CHAT_API_URL before starting Streamlit to use a different local API URL.",
        disabled=True,
    )
    api_ok = check_api_health(api_url)
    if api_ok:
        st.caption("🟢 Chat API connected")
    else:
        st.caption("🔴 Chat API is unavailable")
        st.warning("Chat API is offline. Start it with `python -m uvicorn api:app --port 8000`.")
    st.divider()

    st.markdown("### 🗂️ सत्र एवं चैट इतिहास (Session & History)")
    if st.session_state.session_id:
        st.markdown(f"**सत्र ID:** `{st.session_state.session_id}`")
    else:
        st.caption("पहला संदेश भेजने पर सिस्टम सत्र ID बनाएगा।")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ नया सत्र (New)", use_container_width=True, help="नया केस सत्र शुरू करें"):
            reset_chat_session()
            st.rerun()

    with col_s2:
        if st.button("🗑️ साफ़ करें (Clear)", use_container_width=True, help="वर्तमान चैट साफ़ करें"):
            reset_chat_session()
            st.rerun()

    # Chat Export Download
    if st.session_state.messages:
        md_content = format_chat_as_markdown(
            st.session_state.session_id,
            st.session_state.case_facts,
            st.session_state.messages,
        )
        st.download_button(
            label="💾 केस डायरी निर्यात (.md)",
            data=md_content,
            file_name=f"{st.session_state.session_id or 'chat'}_case_diary.md",
            mime="text/markdown",
            use_container_width=True,
            help="इस विवेचना सत्र की केस डायरी व चैट को Markdown प्रारूप में डाउनलोड करें",
        )

    # Saved Sessions Browser
    saved_list = fetch_saved_sessions(api_url)
    if saved_list:
        with st.expander(f"📜 पूर्व सहेजे गए सत्र ({len(saved_list)})", expanded=False):
            session_options = {
                f"{s['session_id']} ({s.get('message_count', 0)} संदेश) - {s.get('title', '')[:20]}": s["session_id"]
                for s in saved_list
            }
            selected_label = st.selectbox(
                "सत्र चुनें",
                options=list(session_options.keys()),
                label_visibility="collapsed",
            )
            if st.button("📂 चयनित सत्र लोड करें (Load)", use_container_width=True):
                target_id = session_options[selected_label]
                try:
                    load_res = requests.get(
                        f"{api_url}/api/sessions/{target_id}",
                        headers=get_api_headers(),
                        timeout=3,
                    )
                    if load_res.status_code == 200:
                        loaded_data = load_res.json()
                        st.session_state.session_id = loaded_data.get("session_id", target_id)
                        st.session_state.case_facts = loaded_data.get("case_facts")
                        st.session_state.messages = loaded_data.get("messages", [])
                        st.success(f"सत्र `{target_id}` लोड किया गया!")
                        st.rerun()
                    else:
                        st.error("सत्र लोड करने में विफलता।")
                except requests.RequestException:
                    st.error("सर्वर से संपर्क नहीं हो सका।")

    st.divider()

    st.markdown("### 🛡️ सुरक्षा दायरा (Guardrail)")
    st.caption(
        "यह प्रणाली केवल आपराधिक मामलों, पुलिस विवेचना प्रक्रिया, साक्ष्य संकलन तथा "
        "केस डायरी (Case Diary / पर्चा) प्रविष्टियों के उत्तर देने के लिए अधिकृत है।"
    )
    st.divider()

    st.markdown("### 📋 ऑडिट लॉग")
    if st.button("📂 हालिया लॉग देखें (View Logs)", use_container_width=True):
        candidate_paths = [
            Path("logs/query_log.jsonl"),
            Path("logs/query_log.txt"),
            Path("query_log.jsonl"),
            Path("query_log.txt"),
        ]
        target_log = next((p for p in candidate_paths if p.is_file()), None)
        if target_log is not None:
            try:
                log_content = target_log.read_text(encoding="utf-8")
                tail = log_content[-3000:] if len(log_content) > 3000 else log_content
                st.markdown(f'<div class="log-box">{tail}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"लॉग पढ़ने में त्रुटि: {e}")
        else:
            st.warning("कोई ऑडिट लॉग उपलब्ध नहीं है।")

    st.divider()
    st.caption("© 2024 IGWA · केवल आधिकारिक पुलिस विवेचना उपयोग हेतु")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">⚖️ पुलिस विवेचना एवं केस डायरी प्रणाली</p>
        <p class="hero-sub">
            भारतीय न्याय संहिता (BNS 2023), BNSS एवं BSB के अंतर्गत विवेचना अधिकारी (IO) हेतु सटीक विधिक विश्लेषण, अपराध वर्गीकरण एवं केस डायरी SOP सहायक।
        </p>
        <span class="hero-badge">Statutory Offence Relevance Engine · BNS 2023 · BNSS 2023 · BSB 2023 · IT Act</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_mode = st.radio(
    "चैट प्रकार / Chat mode",
    options=["Simple legal chat", "Chat with Source Data"],
    horizontal=True,
    key="chat_mode",
    on_change=reset_chat_session,
)
st.caption("चैट प्रकार बदलने पर अलग सत्र शुरू होगा।")

source_text_val = ""
source_error = None

if chat_mode == "Simple legal chat":
    st.caption("सीधे कोई भी पुलिस विधिक प्रक्रिया या कानूनी सवाल पूछें।")
else:
    st.caption("फ़ील्ड 1 में स्रोत डेटा (तहरीर/प्राथमिकी/केस तथ्य) दर्ज करें और फ़ील्ड 2 में उसके आधार पर प्रश्न पूछें।")

    with st.expander("📄 फ़ील्ड 1: स्रोत डेटा (Source Data) — तहरीर / केस तथ्य / दस्तावेज़", expanded=True):
        uploaded_files = st.file_uploader(
            "दस्तावेज़ से टेक्स्ट निकालें (वैकल्पिक - PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt", "json"],
            accept_multiple_files=True,
            help="अपलोड किए गए दस्तावेज़ से पठनीय टेक्स्ट स्वतः निकालकर नीचे स्रोत डेटा में जोड़ा जाएगा।",
            key="source_uploaded_docs",
        )

        upload_digest = hashlib.sha256()
        document_parts = []
        document_errors = []
        for uploaded_file in uploaded_files:
            if getattr(uploaded_file, "size", 0) > 20 * 1024 * 1024:
                document_errors.append(
                    f"{uploaded_file.name}: फ़ाइल 20 MB से बड़ी है।"
                )
                continue
            file_bytes = uploaded_file.getvalue()
            upload_digest.update(uploaded_file.name.encode("utf-8"))
            upload_digest.update(file_bytes)
            suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

            try:
                if suffix == "pdf":
                    extracted_text = extract_pdf_text(file_bytes)
                    if not extracted_text:
                        raise ValueError("इस PDF में कोई पठनीय टेक्स्ट नहीं मिला।")
                    document_parts.append(extracted_text)
                elif suffix == "docx":
                    extracted_text = extract_docx_text(file_bytes)
                    if not extracted_text:
                        raise ValueError("इस Word दस्तावेज़ में कोई पठनीय टेक्स्ट नहीं मिला।")
                    document_parts.append(extracted_text)
                elif suffix in ("txt", "json"):
                    raw_content = file_bytes.decode("utf-8", errors="replace").strip()
                    if suffix == "json":
                        try:
                            parsed_json = json.loads(raw_content, strict=False)
                            if isinstance(parsed_json, dict) and "source" in parsed_json:
                                extracted_text = str(parsed_json["source"]).strip()
                            else:
                                extracted_text = raw_content
                        except Exception:
                            extracted_text = raw_content
                    else:
                        extracted_text = raw_content

                    if not extracted_text:
                        raise ValueError("इस फ़ाइल में कोई पठनीय सामग्री नहीं मिली।")
                    document_parts.append(extracted_text)
                else:
                    raise ValueError("केवल PDF, DOCX एवं TXT दस्तावेज़ समर्थित हैं।")
            except Exception as exc:
                document_errors.append(f"{uploaded_file.name}: {exc}")

        upload_fingerprint = upload_digest.hexdigest() if uploaded_files else None
        if upload_fingerprint != st.session_state.document_upload_fingerprint:
            st.session_state.document_upload_fingerprint = upload_fingerprint
            if document_parts:
                st.session_state.source_text = "\n\n".join(document_parts)
            st.session_state.document_errors = document_errors
            reset_chat_session()

        for document_error in st.session_state.document_errors:
            st.warning(document_error)

        source_text_val = st.text_area(
            "स्रोत डेटा (वादी की तहरीर, प्राथमिकी, गवाह बयान, या विवेचना तथ्य):",
            value=st.session_state.source_text,
            height=160,
            placeholder="यहाँ वादी की तहरीर, प्राथमिकी (FIR), बयान, पंचनामा या केस तथ्य पेस्ट करें...",
            key="source_text_input",
            help="यह स्रोत डेटा AI को विधिक विश्लेषण के लिए मुख्य आधार के रूप में भेजा जाएगा।",
        )
        st.session_state.source_text = source_text_val

        if len(source_text_val) > 100_000:
            source_error = f"स्रोत डेटा {len(source_text_val):,} वर्ण है; अधिकतम सीमा 100,000 वर्ण है।"
            st.error(source_error)

# ── Live Mode Indicator & Quick Action ─────────────────────────────────────────
col_status, col_btn = st.columns([0.76, 0.24])
with col_status:
    if chat_mode == "Simple legal chat":
        status_text = "💬 सामान्य विधिक चैट — पुलिस एवं विधिक प्रश्न पूछें"
    elif st.session_state.source_text.strip() or st.session_state.case_facts:
        status_text = "📄 स्रोत डेटा सक्रिय — विधिक प्रश्न पूछें"
    else:
        status_text = "📄 स्रोत डेटा दर्ज करें, फिर प्रश्न पूछें"
    st.markdown(
        f'<div class="status-pill"><span class="status-dot"></span> {status_text}</div>',
        unsafe_allow_html=True,
    )
with col_btn:
    if st.session_state.case_facts or st.session_state.source_text:
        if st.button("🔄 नया मामला", use_container_width=True, help="वर्तमान मामला साफ़ कर नया मामला शुरू करें"):
            st.session_state.source_text = ""
            reset_chat_session()
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── Chat History Display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-user">
                <div class="bubble">{msg["content"]}</div>
                <div class="avatar avatar-user">👮</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        _, answer_col = st.columns([0.06, 0.94], gap="small")
        with answer_col:
            if msg.get("guardrail"):
                st.markdown(f'<div class="guardrail-box">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                with st.container(border=True):
                    st.markdown(msg["content"])
                if msg.get("model"):
                    st.markdown(f'<div class="model-pill">⚡ {msg["model"]}</div>', unsafe_allow_html=True)

        if msg.get("warn"):
            warning = "⚠️ मॉडल का उत्तर दोहरावदार था; कृपया विवरण स्पष्ट करें।"
            st.markdown(f'<div class="warn-box">{warning}</div>', unsafe_allow_html=True)

        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} विधिक संदर्भ दस्तावेज प्राप्त हुए", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>[{src["idx"]}] {src["file"]}</strong><br>'
                        f'{src["snippet"]}…'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Quick Action Suggestions for IO ──────────────────────────────────────────
st.markdown("##### ⚡ त्वरित विवेचना प्रश्न (Quick Investigation Actions)")
q_col1, q_col2, q_col3 = st.columns(3)
quick_prompt = None
with q_col1:
    if st.button("📋 संपूर्ण विधिक विश्लेषण व SOP", use_container_width=True, help="4-चरणीय पूर्ण रिपोर्ट तैयार करें"):
        quick_prompt = "इस मामले का पूर्ण विधिक विश्लेषण तैयार करें (मामले का सारांश, अपराध वर्गीकरण तालिका, लागू BNS धाराएं एवं विवेचना अधिकारी हेतु केस डायरी SOP)।"
with q_col2:
    if st.button("⚖️ अपराध वर्गीकरण व BNS धाराएँ", use_container_width=True, help="संज्ञेय/असंज्ञेय व जमानतीय स्थिति तालिका"):
        quick_prompt = "इस मामले में आकर्षित होने वाले सभी अपराधों का प्रथम अनुसूची अनुसार वर्गीकरण (संज्ञेय/असंज्ञेय, जमानतीय स्थिति एवं विचारणीय न्यायालय) तालिका रूप में बताएं।"
with q_col3:
    if st.button("📝 केस डायरी (पर्चा दैनिकी) SOP", use_container_width=True, help="केस डायरी, जब्ती व डिजिटल साक्ष्य"):
        quick_prompt = "इस मामले में विवेचना अधिकारी हेतु केस डायरी (BNSS 193/168), फर्द जब्ती व वीडियोग्राफी (BNSS 105) तथा BSB 2023 धारा 63 इलेक्ट्रॉनिक साक्ष्य संकलन SOP बताएं।"

# ── Chat Input & Execution ────────────────────────────────────────────────────
st.markdown("---")
input_placeholder = (
    "पुलिस या विधिक प्रश्न पूछें (उदा. क्या बिना वारंट गिरफ्तारी संभव है?)..."
    if chat_mode == "Simple legal chat"
    else "फ़ील्ड 2: स्रोत डेटा के आधार पर प्रश्न पूछें (उदा. कौन-सी BNS धाराएं आकर्षित होंगी?)..."
)
query = st.chat_input(
    input_placeholder,
    max_chars=100_000,
    disabled=bool(source_error),
)

active_query = query or quick_prompt
if active_query:
    active_query = active_query.strip()
    if active_query:
        if chat_mode == "Chat with Source Data" and not (st.session_state.source_text.strip() or st.session_state.case_facts):
            st.warning("⚠️ कृपया पहले फ़ील्ड 1 में स्रोत डेटा (तहरीर/प्राथमिकी या केस तथ्य) दर्ज करें।")
        else:
            st.session_state.messages.append({"role": "user", "content": active_query})

            spinner_text = (
                "विधिक उत्तर तैयार हो रहा है…"
                if chat_mode == "Simple legal chat"
                else "स्रोत डेटा के आधार पर विधिक विश्लेषण तैयार हो रहा है…"
            )
            with st.spinner(spinner_text):
                try:
                    payload = {
                        "session_id": st.session_state.session_id,
                        "query": active_query,
                    }
                    if chat_mode == "Chat with Source Data":
                        curr_source = st.session_state.source_text.strip()
                        if curr_source:
                            payload["source"] = curr_source

                    response = requests.post(
                        f"{api_url}/api/v1/chat",
                        json=payload,
                        headers=get_api_headers(),
                        timeout=(15, 240),
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.session_id = result.get("session_id", st.session_state.session_id)
                    st.session_state.case_facts = result.get("source") or result.get("case_facts")
                    fetch_saved_sessions.clear()
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result.get("sources", []),
                            "model": result.get("model", ""),
                            "warn": result.get("warn", False),
                            "guardrail": result.get("guardrail", False),
                        }
                    )
                except requests.RequestException as exc:
                    detail = ""
                    if exc.response is not None:
                        detail = format_api_error(exc.response)
                    if not detail:
                        if isinstance(exc, requests.exceptions.ConnectTimeout):
                            detail = "API सर्वर से जुड़ने में समय अधिक लगा (Connection Timeout)। सर्वर व्यस्त हो सकता है, कृपया पुनः प्रयास करें।"
                        elif isinstance(exc, requests.exceptions.ConnectionError):
                            detail = f"Chat API ({api_url}) से संपर्क नहीं हो सका। कृपया सुनिश्चित करें कि Uvicorn सर्वर चालू है (`python -m uvicorn api:app --port 8000`)।"
                        elif isinstance(exc, requests.exceptions.ReadTimeout):
                            detail = "विधिक विश्लेषण में समय सीमा समाप्त हुई (Read Timeout)। कृपया पुनः प्रयास करें।"
                        else:
                            detail = f"सर्वर से संपर्क विफल: {exc}"

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"❌ {detail}",
                            "sources": [],
                            "warn": False,
                            "guardrail": False,
                        }
                    )

            st.rerun()
