import time
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

class OCRStatus(str, Enum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class OCRPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str = ""

class OCRData(BaseModel):
    pages: list[OCRPage] = Field(default_factory=list)
    full_text: str = ""

class MetadataStatus(str, Enum):
    PENDING = "pending"  # background task scheduled
    PROCESSING = "processing"  # background extraction running
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ApplicationLocation(BaseModel):
    """Where the model says the attached application/complaint is. Locations only, never text.

    start_text/end_text are optional short verbatim quotes used ONLY when the application begins or
    ends in the middle of a page; Python then slices the OCR text of that page between them.
    """

    found: bool = False
    start_page: int | None = None
    end_page: int | None = None
    start_text: str | None = None  # first words of the application, only if it starts mid-page
    end_text: str | None = None  # last words of the application, only if it ends mid-page


class ApplicationResult(BaseModel):
    """Public metadata.application. text is always cut from the OCR pages by Python.

    found=False means confirmed absent; found=True with text=None means the application was located
    but its text could not be cut out reliably (unresolved mid-page boundary).
    """

    found: bool = False
    start_page: int | None = None
    end_page: int | None = None
    text: str | None = None


class OCRMetadata(BaseModel):
    """Metadata extracted from FIR and police documents."""

    # FIR details
    document_type: str | None = None
    fir_number: str | None = None
    police_station: str | None = None
    district: str | None = None
    case_number: str | None = None
    fir_date: str | None = None
    case_type: str | None = None

    # Incident details
    incident_date: str | None = None
    incident_date_time: str | None = None
    incident_location: str | None = None

    # People involved
    complainant_name: str | None = None
    accused_names: list[str] = Field(default_factory=list)
    witness_names: list[str] = Field(default_factory=list)

    # Legal details
    legal_sections: list[str] = Field(default_factory=list)

    # Investigation details
    investigating_officer: str | None = None
    document_narrative: str | None = None
    investigation_actions_findings: list[str] = Field(
        default_factory=list
    )

    # Case diary details
    title: str | None = None

    # Internal: the model's document classification, used only to derive the API's isFir flag.
    # exclude=True keeps the public metadata object at exactly the 15 fields above.
    document_class: Literal["FIR", "APPLICATION", "OTHER", "UNCLEAR"] | None = Field(
        default=None, exclude=True
    )

    # Internal: the model's answer for where the application is (never sent back to clients).
    application_location: ApplicationLocation | None = Field(default=None, exclude=True)

    # Public, filled by Python from the OCR pages after the response. SkipJsonSchema keeps it out of
    # the schema sent to the model so the model can never be asked to generate application text.
    # None = unavailable/unresolved (not a confirmed absence); see ApplicationResult.
    application: SkipJsonSchema[ApplicationResult | None] = None

    @field_validator("application_location", mode="before")
    @classmethod
    def _tolerate_bad_location(cls, value: Any) -> Any:
        """A malformed location must never invalidate the 15 real fields."""
        if isinstance(value, ApplicationLocation):
            return value
        try:
            return ApplicationLocation.model_validate(value) if isinstance(value, dict) else None
        except ValueError:
            return None

    @field_validator("document_class", mode="before")
    @classmethod
    def _normalize_document_class(cls, value: Any) -> Any:
        """Tolerate case/spacing; anything unrecognized becomes UNCLEAR (never invalidates the
        15 real fields, never counts as a confident FIR)."""
        if value is None:
            return None
        text = str(value).strip().upper() if isinstance(value, str) else ""
        return text if text in ("FIR", "APPLICATION", "OTHER", "UNCLEAR") else "UNCLEAR"

class OCRUploadResponse(BaseModel):
    success: bool = True
    message: str = "OCR processing completed successfully"
    document_id: str
    job_id: str = Field(..., description="Same value as document_id; use it to poll metadata")
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
    ocr_provider: str | None = Field(
        default=None, description="gemini or groq"
    )
    ocr_data: OCRData | None = None
    metadata: OCRMetadata | None = None
    metadata_model: str | None = Field(
        default=None, description="Model that produced the metadata (may differ from ocr_model)"
    )
    metadata_status: MetadataStatus = MetadataStatus.COMPLETED
    metadata_error: str | None = Field(
        default=None, description="Set when metadata extraction failed or was skipped"
    )
    isFir: bool = Field(
        default=False, description="true only when the document is confidently an official FIR"
    )

class OCRResultResponse(BaseModel):
    """Stored OCR result as returned by GET /python/ocr/{document_id}."""

    success: bool = True
    document_id: str
    job_id: str = ""
    status: OCRStatus
    filename: str
    ocr_model: str = Field(..., description="Model that produced the OCR text")
    ocr_provider: str | None = Field(
        default=None, description="gemini or groq"
    )
    ocr_data: OCRData | None = None
    metadata: OCRMetadata | None = None
    metadata_model: str | None = Field(
        default=None, description="Model that produced the metadata (may differ from ocr_model)"
    )
    metadata_status: MetadataStatus = MetadataStatus.COMPLETED
    metadata_error: str | None = Field(
        default=None, description="Set when metadata extraction failed or was skipped"
    )
    isFir: bool = Field(
        default=False, description="true only when the document is confidently an official FIR"
    )
    created_at: datetime | None = None
    updated_at: datetime | None = None

class MetadataStatusResponse(BaseModel):
    """GET /python/ocr/{job_id}/metadata: same shape in every state."""

    job_id: str
    metadata_status: MetadataStatus
    metadata: OCRMetadata | None = None
    metadata_provider: str | None = Field(
        default=None, description="Model that produced the metadata"
    )
    metadata_error: str | None = None
    isFir: bool = False

class ErrorResponse(BaseModel):
    detail: str

class _ProviderPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str = ""

class GeminiOCRResult(BaseModel):
    """Schema Gemini is asked to follow (kept minimal for stability)."""

    pages: list[_ProviderPage]
    # No full_text: Python builds it from the pages (build_full_text), so no text is generated twice.
    # Filled by Gemini in the same request as the OCR text (single call, no second pass).
    metadata: OCRMetadata | None = None
    # Location only (page numbers), never text: Python cuts the text from pages[].
    application_location: ApplicationLocation | None = None

def build_full_text(pages: list[OCRPage]) -> str:
    """Combine page texts in page order. The text itself is never modified."""
    return "\n\n".join(p.text for p in sorted(pages, key=lambda p: p.page_number))


def ocr_data_from_dict(raw: dict[str, Any], timings: dict[str, float] | None = None) -> OCRData:
    """Validate a parsed provider dict and build the OCR data stored/returned by the API.

    full_text is always built here from the pages; any full_text a provider might send is ignored.
    If `timings` is given it receives "parse" and "full_text" durations in seconds.
    """
    started = time.perf_counter()
    # Metadata is validated separately (metadata_from_dict): malformed metadata must never
    # discard otherwise good OCR text.
    result = GeminiOCRResult.model_validate(
        {k: v for k, v in raw.items() if k not in ("metadata", "application_location")}
    )
    pages = [
        OCRPage(page_number=p.page_number, text=p.text)
        for p in sorted(result.pages, key=lambda p: p.page_number)
    ]
    parsed = time.perf_counter()
    full_text = build_full_text(pages)
    if timings is not None:
        timings["parse"] = parsed - started
        timings["full_text"] = time.perf_counter() - parsed
    return OCRData(pages=pages, full_text=full_text)


def metadata_from_dict(raw: dict[str, Any]) -> OCRMetadata | None:
    """Validated metadata from a combined OCR response, or None if absent/invalid/empty.

    None means "extract it separately": malformed or all-empty metadata is never accepted.
    """
    candidate = raw.get("metadata")
    if not isinstance(candidate, dict):
        return None
    top_level = raw.get("application_location")
    if isinstance(top_level, dict):  # the combined request returns it next to pages/metadata
        candidate = {**candidate, "application_location": top_level}
    try:
        metadata = OCRMetadata.model_validate(candidate)
    except ValueError:
        return None
    has_content = any(
        (v if isinstance(v, str) else bool(v)) for v in metadata.model_dump().values() if v is not None
    )
    return metadata if has_content else None
