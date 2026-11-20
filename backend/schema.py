"""Request and response models for the IGWA Legal Chatbot API."""

import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


# -----------------------------
# Chat History Models
# -----------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10_000)


# -----------------------------
# OCR Models (Backward Compatibility)
# -----------------------------

class OCRPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str = Field(default="", max_length=100_000)


class OCRData(BaseModel):
    pages: list[OCRPage] = Field(default_factory=list)
    full_text: str = Field(default="", max_length=100_000)

    @model_validator(mode="after")
    def validate_page_text_limit(self) -> "OCRData":
        if not self.full_text:
            page_text = "\n\n".join(
                page.text.strip()
                for page in self.pages
                if page.text.strip()
            )
            if len(page_text) > 100_000:
                raise ValueError("Combined OCR page text must not exceed 100000 characters.")

        return self


class OCRMetadata(BaseModel):
    """Structured metadata (backward compatible)."""

    model_config = ConfigDict(extra="allow")

    document_type: str | None = None
    fir_number: str | None = None
    case_number: str | None = None
    document_date: str | None = None
    police_station: str | None = None
    district: str | None = None
    investigating_officer: str | None = None
    legal_sections: list[str] = Field(default_factory=list)
    complainant_name: str | None = None
    accused_names: list[str] = Field(default_factory=list)
    witness_names: list[str] = Field(default_factory=list)
    incident_date_time: str | None = None
    incident_location: str | None = None
    document_narrative: str | None = None
    investigation_actions_findings: list[str] = Field(default_factory=list)


# -----------------------------
# Chat Request Model
# -----------------------------

class ChatRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "source": (
                    "तहरीर: वादी की दुकान से रात्रि में 50,000 रुपये नकद एवं लैपटॉप "
                    "चोरी हुए। आरोपी सीसीटीवी में दिखे हैं।"
                ),
                "query": "इस स्रोत के आधार पर कौन-सी BNS धाराएं लगेंगी और विवेचना की क्या SOP होगी?",
                "session_id": "case_20261119_180000",
                "chat_history": []
            }
        },
    )

    # Field 1: Source data provided by frontend
    source: str | None = Field(
        default=None,
        max_length=100_000,
        description="Source data (complaint, FIR, witness statement, or case record) provided by frontend."
    )

    # Field 2: User query
    query: str = Field(
        min_length=1,
        max_length=100_000,
        description="Investigating officer's query regarding the source data or Indian criminal law."
    )

    # Session identification
    session_id: str | None = Field(
        default=None,
        max_length=100
    )

    # Response language
    language: Literal["auto", "hi", "en"] = "auto"

    # Case-specific information
    case_facts: str | None = Field(
        default=None,
        max_length=100_000
    )

    # Optional summary supplied by clients that compact previous conversation turns.
    summary: str | None = Field(
        default=None,
        max_length=100_000,
        description="Conversation summary supplied as context for this request."
    )

    # Extracted document text (backward compatibility)
    document_text: str | None = Field(
        default=None,
        max_length=100_000
    )

    # Document identifier
    document_id: str | None = Field(
        default=None,
        max_length=100,
    )

    # OCR extracted data (backward compatibility)
    ocr_data: OCRData | None = None

    # Structured metadata (backward compatibility)
    metadata: OCRMetadata | None = None

    # Previous conversation
    chat_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=100_000
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query must not be blank.")

        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None

        if value is not None and not _SESSION_ID.fullmatch(value):
            raise ValueError(
                "Session ID may contain only letters, numbers, "
                "underscores, and hyphens."
            )

        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None

        return value.strip() if value else None


# -----------------------------
# Chat Response Models
# -----------------------------

class ChatSource(BaseModel):
    idx: int
    file: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str | None = None
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)
    warn: bool = False
    guardrail: bool = False
    source: str | None = None
    case_facts: str | None = None
    model: str | None = None


# -----------------------------
# Health Response
# -----------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"]
    rag_loaded: bool


# -----------------------------
# Session Response Models
# -----------------------------

class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None
    message_count: int
    source: str | None = None
    case_facts: str | None = None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)
