import base64
import binascii
import datetime
import json
import logging
from pathlib import Path
import re
import threading
from typing import Literal, Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger("chatbot.api")
app = FastAPI(
    title="IGWA Legal Chatbot API",
    description="Local API for the police investigation and case diary assistant.",
    version="1.0.0",
)

GUARDRAIL_REFUSAL_MESSAGE = (
    "⚠️ **विधिक सुरक्षा दायरा एवं अस्वीकरण (Guardrail Notice):**\n\n"
    "यह प्रणाली केवल **भारतीय आपराधिक विधि (BNS 2023, BNSS 2023, BSB 2023, IPC, CrPC), पुलिस अन्वेषण/विवेचना प्रक्रिया, साक्ष्य संकलन, तथा केस डायरी (Case Diary / पर्चा दैनिकी) प्रविष्टि** से संबंधित प्रश्नों के लिए अधिकृत है।\n\n"
    "❌ **अप्रासंगिक, सामान्य ज्ञान अथवा गैर-विवेचनात्मक प्रश्नों के उत्तर देना इस प्रणाली के कार्यक्षेत्र से बाहर है।**\n\n"
    "📌 कृपया केवल किसी आपराधिक घटना, पुलिस जांच प्रक्रिया, विधिक धारा, जब्ती/गिरफ्तारी अथवा केस डायरी प्रारूप से संबंधित प्रश्न ही पूछें।"
)

_rag = None
_rag_lock = threading.Lock()


def is_police_investigation_query(text: str) -> bool:
    """Pre-filter requests unrelated to police investigation, crime, or law."""
    text_clean = text.lower().strip()
    if not text_clean:
        return False

    irrelevant_patterns = [
        r"^(hi|hello|hey|greetings|how are you|who are you|kya haal hai|kaise ho)\b",
        r"\b(recipe|cook|bake|biryani|tea|coffee|maggi|pizza|burger|khana|sabzi)\b",
        r"\b(python|javascript|java|c\+\+|html|css|sql|function|react|angular|code|debug|script|compile)\b",
        r"\b(poem|poetry|shayari|joke|chutkula|movie|song|cinema|cricket|ipl|football|actor|actress|bollywood)\b",
        r"\b(capital of|weather in|temperature in|president of|prime minister of|who won the match)\b",
        r"\b(solve this equation|differentiate|integrate|calculate \d+)\b",
        r"(\d+\s*[\+\-\*\/]\s*\d+)",
    ]

    legal_override = [
        "police", "investigation", "fir", "bns", "bnss", "bsb", "ipc", "crpc",
        "arrest", "bail", "seizure", "case diary", "diary", "crime", "theft", "murder",
        "robbery", "cyber", "weapon", "assault", "harassment", "ndps", "pocso",
        "पुलिस", "विवेचना", "अन्वेषण", "केस डायरी", "पर्चा", "अपराध", "धारा", "गिरफ्तारी",
        "जमानत", "फर्द", "बरामदगी", "साक्ष्य", "गवाह", "चोरी", "हत्या", "लूट",
    ]

    for pattern in irrelevant_patterns:
        if re.search(pattern, text_clean) and not any(
            term in text_clean for term in legal_override
        ):
            return False
    return True


def detect_degenerate_output(response_text: str, repeat_threshold: int = 3) -> bool:
    words = response_text.split()
    if len(words) < 20:
        return False

    phrase_counts = {}
    for index in range(len(words) - 7):
        phrase = " ".join(words[index : index + 8])
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        if phrase_counts[phrase] > repeat_threshold:
            return True
    return False


SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)


def get_session_file(session_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", session_id)
    return SESSIONS_DIR / f"{safe_id}.json"


def load_session(session_id: str) -> dict | None:
    path = get_session_file(session_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load session %s", session_id)
    return None


def save_session_interaction(
    session_id: str,
    query: str,
    answer: str,
    sources: list[dict],
    case_facts: str | None = None,
    guardrail: bool = False,
    warn: bool = False,
) -> dict:
    session_data = load_session(session_id)
    now = datetime.datetime.now().isoformat()
    if not session_data:
        title = query[:60] + ("…" if len(query) > 60 else "")
        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "case_facts": case_facts,
            "messages": [],
        }
    else:
        session_data["updated_at"] = now
        if case_facts and not session_data.get("case_facts"):
            session_data["case_facts"] = case_facts

    session_data["messages"].append({
        "role": "user",
        "content": query,
        "timestamp": now,
    })
    session_data["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "guardrail": guardrail,
        "warn": warn,
        "timestamp": now,
    })

    try:
        path = get_session_file(session_id)
        path.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save session %s", session_id)

    return session_data


def log_interaction(query: str, response: str, sources: list[dict], session_id: str = "") -> None:
    try:
        with open("query_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{datetime.datetime.now().isoformat()}] [Session: {session_id or 'none'}]\n")
            log_file.write(f"Query: {query}\n")
            log_file.write(f"Response: {response}\n")
            for source in sources:
                log_file.write(f"  Source: {source['file']}\n")
            log_file.write("-" * 50 + "\n")
    except OSError:
        logger.exception("Could not write the chatbot audit log.")


class ImageAttachment(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    data: str = Field(min_length=1, max_length=14_000_000)

    @field_validator("data")
    @classmethod
    def validate_image_data(cls, value: str) -> str:
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Image data must be valid base64.") from exc
        if len(image_bytes) > 10 * 1024 * 1024:
            raise ValueError("Each image must be 10 MB or smaller.")
        return value


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=100)
    query: str = Field(min_length=1, max_length=10_000)
    case_facts: str | None = Field(default=None, max_length=50_000)
    document_text: str | None = Field(default=None, max_length=50_000)
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)
    chat_history: list[dict] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_total_image_size(self):
        total_size = sum(
            len(base64.b64decode(image.data, validate=True)) for image in self.images
        )
        if total_size > 20 * 1024 * 1024:
            raise ValueError("Combined image attachments must be 20 MB or smaller.")
        return self


class OCRPage(BaseModel):
    page_number: int | None = Field(default=1, description="Page number of the document")
    text: str = Field(description="Extracted OCR text of this page")


class OCRIngestRequest(BaseModel):
    model_config = {"extra": "allow"}

    document_name: str | None = Field(
        default=None,
        description="Name or identifier of the document (e.g. 'FIR_04_2026.pdf')",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional unique document ID",
    )
    title: str | None = Field(
        default=None,
        description="Optional title or case summary",
    )
    text: str | None = Field(
        default=None,
        description="Full extracted text if not separated into pages",
    )
    pages: list[OCRPage] = Field(
        default_factory=list,
        description="List of individual page texts from OCR",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary OCR metadata (e.g., confidence, language, timestamps, scanner details)",
    )
    raw_ocr_data: Any | None = Field(
        default=None,
        description="Any raw vendor-specific OCR payload (e.g. Google Vision fullTextAnnotation, AWS Textract Blocks)",
    )


def extract_ocr_chunks(payload: OCRIngestRequest) -> tuple[str, list[dict[str, Any]]]:
    """Extracts document name and chunk list from various OCR JSON formats."""
    from rag import chunk_text

    doc_name = (
        payload.document_name
        or payload.document_id
        or payload.title
        or f"ocr_doc_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    chunks = []

    # 1. Individual pages
    if payload.pages:
        for p in payload.pages:
            p_num = p.page_number or 1
            page_text = p.text.strip()
            if page_text:
                for text_chunk in chunk_text(page_text):
                    chunks.append({
                        "text": text_chunk,
                        "source": doc_name,
                        "page": p_num,
                    })

    # 2. Top-level single text
    elif payload.text and payload.text.strip():
        for text_chunk in chunk_text(payload.text.strip()):
            chunks.append({
                "text": text_chunk,
                "source": doc_name,
                "page": 1,
            })

    # 3. Fallback: inspect raw_ocr_data or metadata
    elif payload.raw_ocr_data:
        raw = payload.raw_ocr_data
        extracted_text = ""

        if isinstance(raw, dict):
            # Google Cloud Vision format
            if "fullTextAnnotation" in raw and isinstance(raw["fullTextAnnotation"], dict):
                extracted_text = raw["fullTextAnnotation"].get("text", "")
            elif "responses" in raw and isinstance(raw["responses"], list) and raw["responses"]:
                extracted_text = raw["responses"][0].get("fullTextAnnotation", {}).get("text", "")
            # AWS Textract format
            elif "Blocks" in raw and isinstance(raw["Blocks"], list):
                extracted_text = "\n".join(
                    b.get("Text", "")
                    for b in raw["Blocks"]
                    if b.get("BlockType") == "LINE" and b.get("Text")
                )
            # Azure Computer Vision format
            elif "analyzeResult" in raw and isinstance(raw["analyzeResult"], dict):
                lines = []
                for read_res in raw["analyzeResult"].get("readResults", []):
                    for line in read_res.get("lines", []):
                        if "text" in line:
                            lines.append(line["text"])
                extracted_text = "\n".join(lines)
            # Generic nested "content" or "text" or "pages"
            elif "content" in raw and isinstance(raw["content"], str):
                extracted_text = raw["content"]
            elif "text" in raw and isinstance(raw["text"], str):
                extracted_text = raw["text"]
            elif "pages" in raw and isinstance(raw["pages"], list):
                for idx, page in enumerate(raw["pages"], 1):
                    p_text = page.get("text") or page.get("content") or ""
                    if isinstance(p_text, str) and p_text.strip():
                        for text_chunk in chunk_text(p_text.strip()):
                            chunks.append({
                                "text": text_chunk,
                                "source": doc_name,
                                "page": page.get("page_number", idx),
                            })
        elif isinstance(raw, list):
            for idx, item in enumerate(raw, 1):
                if isinstance(item, str) and item.strip():
                    for text_chunk in chunk_text(item.strip()):
                        chunks.append({
                            "text": text_chunk,
                            "source": doc_name,
                            "page": idx,
                        })
                elif isinstance(item, dict):
                    p_text = item.get("text") or item.get("content") or ""
                    if isinstance(p_text, str) and p_text.strip():
                        for text_chunk in chunk_text(p_text.strip()):
                            chunks.append({
                                "text": text_chunk,
                                "source": doc_name,
                                "page": item.get("page_number", idx),
                            })

        if extracted_text.strip() and not chunks:
            for text_chunk in chunk_text(extracted_text.strip()):
                chunks.append({
                    "text": text_chunk,
                    "source": doc_name,
                    "page": 1,
                })

    return doc_name, chunks


def get_rag():
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                from rag import LocalFAISSRag

                _rag = LocalFAISSRag()
    return _rag


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "rag_loaded": _rag is not None}


@app.post("/api/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or f"case_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    if not is_police_investigation_query(request.query):
        save_session_interaction(
            session_id=session_id,
            query=request.query,
            answer=GUARDRAIL_REFUSAL_MESSAGE,
            sources=[],
            case_facts=request.case_facts,
            guardrail=True,
            warn=False,
        )
        return {
            "session_id": session_id,
            "answer": GUARDRAIL_REFUSAL_MESSAGE,
            "sources": [],
            "warn": False,
            "guardrail": True,
            "case_facts": request.case_facts,
        }

    images = [
        {
            "name": image.name,
            "mime_type": image.mime_type,
            "data": base64.b64decode(image.data, validate=True),
        }
        for image in request.images
    ]

    if request.case_facts:
        scenario = request.case_facts
        follow_up = request.query
    else:
        scenario = request.document_text or request.query
        follow_up = None

    try:
        rag = get_rag()
        answer, retrieved_sources, _ = rag.analyze_gemini(
            scenario,
            follow_up=follow_up,
            allow_groq_fallback=True,
            image_attachments=images,
            chat_history=request.chat_history,
        )
    except Exception as exc:
        logger.exception("Chat analysis failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Chat analysis failed: {exc}",
        ) from exc

    sources = [
        {
            "idx": index,
            "file": source.get("source", "अज्ञात स्रोत"),
            "snippet": source.get("text", "").replace("\n", " ")[:200],
        }
        for index, source in enumerate(retrieved_sources, 1)
    ]
    warn_flag = detect_degenerate_output(answer)
    save_session_interaction(
        session_id=session_id,
        query=request.query,
        answer=answer,
        sources=sources,
        case_facts=scenario,
        guardrail=False,
        warn=warn_flag,
    )
    log_interaction(request.query, answer, sources, session_id=session_id)
    return {
        "session_id": session_id,
        "answer": answer,
        "sources": sources,
        "warn": warn_flag,
        "guardrail": False,
        "case_facts": scenario,
    }


@app.get("/api/sessions")
def list_sessions():
    """List all saved chat sessions ordered by latest update."""
    sessions = []
    for file in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id", file.stem),
                "title": data.get("title", "अनाम प्रकरण"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "message_count": len(data.get("messages", [])),
                "case_facts": data.get("case_facts"),
            })
        except Exception:
            continue
    sessions.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Retrieve full chat history for a session."""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/documents/ingest")
@app.post("/api/documents/ingest")
def ingest_ocr_document(payload: OCRIngestRequest):
    """
    Receive and ingest OCR JSON data into the legal FAISS vector index.
    Supports single text, structured pages, or vendor-specific raw OCR outputs (Google Vision, AWS Textract, Azure).
    """
    doc_name, chunks = extract_ocr_chunks(payload)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text found in the OCR JSON payload. Please provide 'text', 'pages', or valid 'raw_ocr_data'.",
        )

    try:
        rag = get_rag()
        raw_dict = payload.model_dump()
        result = rag.ingest_ocr_document(
            document_name=doc_name,
            chunks_data=chunks,
            raw_json=raw_dict,
        )
        return {
            "status": "success",
            "document_name": doc_name,
            "document_id": payload.document_id,
            "chunks_indexed": result["chunks_indexed"],
            "total_vectors": result["total_vectors"],
            "message": f"Successfully ingested OCR document '{doc_name}' ({result['chunks_indexed']} chunks indexed).",
        }
    except Exception as exc:
        logger.exception("Failed to ingest OCR document: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest OCR document: {exc}",
        ) from exc

