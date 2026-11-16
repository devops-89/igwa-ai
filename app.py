import base64
import hashlib
import io
import logging
import mimetypes
import os
import xml.etree.ElementTree as ET
import zipfile

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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "case_facts" not in st.session_state:
    st.session_state.case_facts = None
if "document_upload_fingerprint" not in st.session_state:
    st.session_state.document_upload_fingerprint = None
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "image_attachments" not in st.session_state:
    st.session_state.image_attachments = []
if "document_errors" not in st.session_state:
    st.session_state.document_errors = []


def expand_scenario_for_law_query(user_input: str) -> str:
    """Keyword expansion helper to target relevant legal texts and BNSS classifications."""
    text = user_input.lower()
    expansions = []
    patterns = {
        "missing|failed to return|gone missing|disappeared|kidnap|abduct|गुमशुदा|लापता|गायब|अपहरण|व्यपहरण": [
            "missing person kidnapping abduction wrongful confinement trafficking BNS 137 138 140 143 BNSS missing person report FIR",
        ],
        "cyber|social media|online account|message|profile|whatsapp|email|it act|साइबर|सोशल मीडिया|ऑनलाइन खाता|संदेश|धमकी|फर्जी प्रोफाइल": [
            "cyber crime information technology act identity theft cheating personation stalking criminal intimidation BNS 77 78 318 319 351 IT Act 66 66C 66D",
        ],
        "bank|unauthorized transaction|account debit|withdrawal|upi|atm|phishing|otp|बैंक|अनधिकृत लेनदेन|राशि निकाली|रुपये निकाले|ओटीपी|फिशिंग": [
            "bank unauthorized transaction UPI net banking phishing OTP identity theft cheating electronic record IT Act 66C 66D BNS 318 319",
        ],
        "illegal occupation|occupy|trespass|burglary|break-in|housebreaking|कब्जा|अवैध कब्जा|घर में घुस|अनधिकृत प्रवेश|अतिचार|गृह-अतिचार": [
            "criminal trespass house trespass house breaking lurking house trespass BNS 329 330 331 332 333 Specific Relief Act",
        ],
        "stolen|theft|vehicle stolen|snatching|robbery|dacoity|extortion|चोरी|चुराया|लूट|लूटपाट|डकैती|छीनना|झपटमारी|उद्यापन|जबरन वसूली": [
            "theft snatching extortion robbery dacoity criminal breach of trust BNS 303 304 308 309 310 316 317",
        ],
        "assault|beaten|hit|struck|attacked|injured|hurt|grievous hurt|मारपीट|पीटा|हमला|चोट|घायल|झगड़ा|हाथापाई|उपहति|घोर उपहति": [
            "voluntarily causing hurt grievous hurt criminal force assault BNS 115 116 117 118 121 131 132 common intention BNS 3(5)",
        ],
        "fraud|cheating|forged cheque|fake identity|loan|forgery|impersonation|धोखाधड़ी|छल|ठगी|जालसाजी|कूटरचना|फर्जी दस्तावेज|जाली दस्तावेज": [
            "cheating fraud cheating personation forgery false document use forged document BNS 61 318 319 335 336 338 340",
        ],
        "murder|dead|death|killed|homicide|attempt to murder|मृत|मृत्यु|मौत|हत्या|हत्या का प्रयास|जान से मार|शव": [
            "murder culpable homicide causing death attempt to murder BNS 100 101 103 105 109 postmortem forensic evidence",
        ],
        "mischief|damage to property|destruction|crops|fire|arson|तोड़फोड़|संपत्ति को नुकसान|फसल नष्ट|आग लगाई|आगजनी": [
            "mischief property damage destruction arson crop damage BNS 324 326",
        ],
        "firearm|arms|illegal weapons|gun|pistol|revolver|ammunition|हथियार|अवैध हथियार|बंदूक|पिस्तौल|रिवॉल्वर|कारतूस": [
            "Arms Act illegal possession firearm ammunition use weapon section 25 27 Arms Act",
        ],
        "narcotic|drug|ganja|cocaine|heroin|opium|नशीला पदार्थ|मादक पदार्थ|नारकोटिक्स|गांजा|चरस|कोकीन|हेरोइन|अफीम": [
            "NDPS Act narcotic drugs psychotropic substance possession sale sections 8 20 21 22",
        ],
        "rape|sexual assault|stalking|voyeurism|dowry death|domestic violence|cruelty|बलात्कार|दुष्कर्म|छेड़छाड़|यौन उत्पीड़न|पीछा करना|दहेज मृत्यु|घरेलू हिंसा": [
            "BNS offences against women rape sexual assault sexual harassment voyeurism stalking cruelty husband 63 64 74 75 78 85 86 Dowry Prohibition Act",
        ],
        "road accident|motor accident|hit and run|rash driving|negligent driving|सड़क दुर्घटना|हिट एंड रन|टक्कर|तेज रफ्तार|सड़क हादसा": [
            "motor vehicle road accident rash negligent driving BNS 281 106 125 Motor Vehicles Act",
        ],
    }

    for pattern, keywords in patterns.items():
        if any(key in text for key in pattern.split("|")):
            expansions.extend(keywords)

    expansions.append(
        "BNSS 2023 First Schedule classification of offences: cognizable or non-cognizable; bailable or non-bailable; case diary SOP"
    )
    return " ".join(dict.fromkeys(expansions))


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
| मॉडल | `gemini-3.6-flash` |
| एम्बेडिंग्स | `all-MiniLM-L6-v2` |
| आउटपुट भाषा | **मानक पुलिस एवं विधिक हिंदी** |
| विधिक कार्यक्षेत्र | BNS · BNSS · BSB · IT · NDPS |
| सुरक्षा दायरा (Guardrail) | **केवल विवेचना एवं केस डायरी** |
        """
    )
    st.divider()

    api_url = st.text_input(
        "Chat API URL",
        value=os.getenv("CHAT_API_URL", "http://127.0.0.1:8000").rstrip("/"),
        help="Set CHAT_API_URL before starting Streamlit to use a different local API URL.",
        disabled=True,
    )
    try:
        api_status = requests.get(f"{api_url}/health", timeout=2).json()
        st.caption(
            "🟢 Chat API connected"
            if api_status.get("status") == "ok"
            else "🔴 Chat API is unavailable"
        )
    except requests.RequestException:
        st.warning("Chat API is offline. Start it with `python -m uvicorn api:app --port 8000`.")
    st.divider()

    st.markdown("### 🛡️ सुरक्षा दायरा (Guardrail)")
    st.caption(
        "यह प्रणाली केवल आपराधिक मामलों, पुलिस विवेचना प्रक्रिया, साक्ष्य संकलन तथा "
        "केस डायरी (Case Diary / पर्चा) प्रविष्टियों के उत्तर देने के लिए अधिकृत है।"
    )
    st.divider()

    if st.button("🔄 नया मामला दर्ज करें (New Case)", use_container_width=True, help="नया आपराधिक मामला दर्ज कर उसका विश्लेषण करें"):
        st.session_state.messages = []
        st.session_state.case_facts = None
        st.rerun()

    if st.button("🗑️ चैट इतिहास साफ़ करें (Clear Chat)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.case_facts = None
        st.rerun()

    st.divider()
    st.markdown("### 📋 ऑडिट लॉग")
    if st.button("📂 हालिया लॉग देखें (View Logs)", use_container_width=True):
        try:
            with open("query_log.txt", "r", encoding="utf-8") as f:
                log_content = f.read()
            tail = log_content[-3000:] if len(log_content) > 3000 else log_content
            st.markdown(f'<div class="log-box">{tail}</div>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning("कोई ऑडिट लॉग उपलब्ध नहीं है।")
        except Exception as e:
            st.error(f"लॉग पढ़ने में त्रुटि: {e}")

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
            भारतीय न्याय संहिता (BNS), BNSS एवं BSB के अंतर्गत विवेचना अधिकारी (IO) हेतु विधिक विश्लेषण एवं केस डायरी सहायक।
        </p>
        <span class="hero-badge">मानक पुलिस हिंदी · BNS · BNSS · BSB · केस डायरी SOP</span>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Attach documents (PDF, image, or Word DOCX)",
    type=["pdf", "png", "jpg", "jpeg", "webp", "docx"],
    accept_multiple_files=True,
    help="Text is extracted from PDFs and DOCX files. Images are read when you send a question.",
)
st.caption("Uploaded content is sent to Gemini only when you submit a question.")

upload_digest = hashlib.sha256()
document_parts = []
image_attachments = []
document_errors = []
for uploaded_file in uploaded_files:
    file_bytes = uploaded_file.getvalue()
    upload_digest.update(uploaded_file.name.encode("utf-8"))
    upload_digest.update(file_bytes)
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    try:
        if suffix == "pdf":
            extracted_text = extract_pdf_text(file_bytes)
            if not extracted_text:
                raise ValueError("No readable text found. Scanned PDFs need OCR.")
            document_parts.append(extracted_text)
        elif suffix == "docx":
            extracted_text = extract_docx_text(file_bytes)
            if not extracted_text:
                raise ValueError("No readable text found in this document.")
            document_parts.append(extracted_text)
        else:
            mime_type = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0]
            if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Unsupported image format.")
            image_attachments.append(
                {
                    "name": uploaded_file.name,
                    "mime_type": mime_type,
                    "data": file_bytes,
                }
            )
    except Exception as exc:
        document_errors.append(f"{uploaded_file.name}: {exc}")

upload_fingerprint = upload_digest.hexdigest() if uploaded_files else None
if upload_fingerprint != st.session_state.document_upload_fingerprint:
    st.session_state.document_upload_fingerprint = upload_fingerprint
    st.session_state.document_text = "\n\n".join(document_parts)
    st.session_state.image_attachments = image_attachments
    st.session_state.document_errors = document_errors
    st.session_state.messages = []
    st.session_state.case_facts = None

for document_error in st.session_state.document_errors:
    st.warning(document_error)

# ── Live Mode Indicator & Quick Action ─────────────────────────────────────────
col_status, col_btn = st.columns([0.76, 0.24])
with col_status:
    if st.session_state.case_facts:
        st.markdown(
            '<div class="status-pill"><span class="status-dot"></span> 💬 लाइव केस चैट सक्रिय — विवेचना अथवा केस डायरी संबंधी प्रश्न पूछें</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill"><span class="status-dot"></span> 📋 तैयार — विवेचना व केस डायरी हेतु घटना का विवरण दर्ज करें</div>',
            unsafe_allow_html=True,
        )
with col_btn:
    if st.session_state.case_facts:
        if st.button("🔄 नया मामला", use_container_width=True, help="वर्तमान मामला साफ़ कर नया मामला शुरू करें"):
            st.session_state.messages = []
            st.session_state.case_facts = None
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

# ── Chat Input & Execution ────────────────────────────────────────────────────
st.markdown("---")
input_placeholder = (
    "विवेचना, साक्ष्य, धारा, गिरफ्तारी या केस डायरी प्रविष्टि संबंधी प्रश्न पूछें"
    if st.session_state.case_facts
    else "आपराधिक घटना का विवरण दर्ज करें"
)
query = st.chat_input(input_placeholder)

if query:
    query = query.strip()
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        spinner_text = (
            "मामले का विधिक विश्लेषण एवं केस डायरी तैयार हो रही है…"
            if not st.session_state.case_facts
            else "विवेचना प्रविष्टि तैयार की जा रही है…"
        )
        with st.spinner(spinner_text):
            try:
                payload = {
                    "query": query,
                    "case_facts": st.session_state.case_facts,
                    "document_text": st.session_state.document_text or None,
                    "images": [
                        {
                            "name": image["name"],
                            "mime_type": image["mime_type"],
                            "data": base64.b64encode(image["data"]).decode("ascii"),
                        }
                        for image in st.session_state.image_attachments
                    ],
                }
                response = requests.post(
                    f"{api_url}/api/chat",
                    json=payload,
                    timeout=(5, 240),
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.case_facts = result.get("case_facts")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "warn": result.get("warn", False),
                        "guardrail": result.get("guardrail", False),
                    }
                )
            except requests.RequestException as exc:
                detail = ""
                if exc.response is not None:
                    try:
                        detail = exc.response.json().get("detail", "")
                    except ValueError:
                        detail = ""
                error_message = detail or (
                    "Chat API से संपर्क नहीं हो सका। API सर्वर चालू करें और CHAT_API_URL जाँचें।"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"❌ {error_message}",
                        "sources": [],
                        "warn": False,
                        "guardrail": False,
                    }
                )

        st.rerun()
