import base64
import binascii
import datetime
import logging
import re
import threading
from typing import Literal

from fastapi import FastAPI, HTTPException
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


def log_interaction(query: str, response: str, sources: list[dict]) -> None:
    try:
        with open("query_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[{datetime.datetime.now().isoformat()}]\n")
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
    query: str = Field(min_length=1, max_length=10_000)
    case_facts: str | None = Field(default=None, max_length=50_000)
    document_text: str | None = Field(default=None, max_length=50_000)
    images: list[ImageAttachment] = Field(default_factory=list, max_length=4)

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


def get_rag():
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                from rag import LocalFAISSRag

                _rag = LocalFAISSRag()
    return _rag


@app.get("/health")
def health():
    return {"status": "ok", "rag_loaded": _rag is not None}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not is_police_investigation_query(request.query):
        return {
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
        follow_up = request.query if request.document_text else None

    try:
        rag = get_rag()
        answer, retrieved_sources, _ = rag.analyze_gemini(
            scenario,
            follow_up=follow_up,
            image_attachments=images,
        )
    except Exception as exc:
        logger.exception("Chat analysis failed.")
        raise HTTPException(
            status_code=502,
            detail="Chat analysis failed. Check the API server logs for details.",
        ) from exc

    sources = [
        {
            "idx": index,
            "file": source.get("source", "अज्ञात स्रोत"),
            "snippet": source.get("text", "").replace("\n", " ")[:200],
        }
        for index, source in enumerate(retrieved_sources, 1)
    ]
    log_interaction(request.query, answer, sources)
    return {
        "answer": answer,
        "sources": sources,
        "warn": detect_degenerate_output(answer),
        "guardrail": False,
        "case_facts": scenario,
    }
