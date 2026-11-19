import asyncio
import base64
import json
import logging
import re
from typing import Any

import fitz

from groq import AsyncGroq
from pydantic import ValidationError
from config.settings import get_settings
from services.ocr.gemini_client import METADATA_PROMPT, describe_provider_error
from services.ocr.schemas import OCRMetadata

logger = logging.getLogger(__name__)

GROQ_OCR_PROMPT = """You are an OCR extraction engine.

Extract all visible text from this page image exactly as it appears.

Rules:
1. Do not summarize, translate, rewrite or correct anything.
2. Preserve the original language, reading order, headings, paragraphs and lists.
3. Preserve tables using Markdown table syntax.
4. Preserve numbers, dates, names and identifiers exactly.
5. Do not invent missing text; mark unreadable text as [UNREADABLE].
6. Return only the extracted text, with no commentary."""

_PDF_RENDER_ZOOM = 2.0
_MAX_CONCURRENT_PAGES = 1
# Groq limits base64 image payloads to 4MB.
_MAX_IMAGE_BYTES = 3 * 1024 * 1024

class GroqOCRError(Exception):
    """Raised when the Groq provider fails or returns unusable output."""

_client: AsyncGroq | None = None

def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = get_settings().groq_api_key
        if not api_key:
            raise GroqOCRError("GROQ_API_KEY is not configured")
        _client = AsyncGroq(api_key=api_key)
    return _client

def _render_pdf_pages(data: bytes) -> list[tuple[bytes, str]]:
    pages: list[tuple[bytes, str]] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        matrix = fitz.Matrix(_PDF_RENDER_ZOOM, _PDF_RENDER_ZOOM)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append((pix.tobytes("jpeg", jpg_quality=85), "image/jpeg"))
    return pages

def _shrink_image(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """Re-encode an oversized image as a downscaled JPEG so it fits Groq's limit."""
    if len(data) <= _MAX_IMAGE_BYTES:
        return data, mime_type
    try:
        pix = fitz.Pixmap(data)
        if pix.alpha or pix.colorspace.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        while True:
            out = pix.tobytes("jpeg", jpg_quality=80)
            if len(out) <= _MAX_IMAGE_BYTES or min(pix.width, pix.height) < 400:
                return out, "image/jpeg"
            pix.shrink(1)
    except Exception:
        return data, mime_type

def _prepare_images(data: bytes, mime_type: str) -> list[tuple[bytes, str]]:
    if mime_type == "application/pdf":
        images = _render_pdf_pages(data)
    else:
        images = [(data, mime_type)]
    return [_shrink_image(img, mime) for img, mime in images]

async def _ocr_page(image: bytes, mime_type: str, sem: asyncio.Semaphore) -> str:
    settings = get_settings()
    b64 = base64.b64encode(image).decode()
    async with sem:
        response = await _get_client().chat.completions.create(
            model=settings.groq_model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GROQ_OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                }
            ],
        )
    return (response.choices[0].message.content or "").strip()

async def extract_text(data: bytes, mime_type: str) -> dict[str, Any]:
    """OCR a document with Groq; returns the same dict shape as the Gemini client."""
    try:
        images = await asyncio.to_thread(_prepare_images, data, mime_type)
        if not images:
            raise GroqOCRError("Document has no pages")
        sem = asyncio.Semaphore(_MAX_CONCURRENT_PAGES)
        texts = await asyncio.gather(*(_ocr_page(img, mime, sem) for img, mime in images))
    except GroqOCRError:
        raise
    except Exception as exc:
        logger.error("Groq request failed: %s", type(exc).__name__)
        raise GroqOCRError("Groq request failed") from exc

    pages = [{"page_number": i, "text": t} for i, t in enumerate(texts, start=1)]
    return {"pages": pages, "full_text": "\n\n".join(texts)}


class GroqMetadataError(Exception):
    """Raised when Groq metadata extraction fails or returns unusable output."""

_METADATA_TIMEOUT_SECONDS = 60.0

_METADATA_JSON_SHAPE = """Return ONLY one JSON object with exactly these keys:
{
"document_type": string|null,
"title": string|null,
"fir_number": string|null,
"fir_date": string|null,
"case_number": string|null,
"case_type": string|null,
"document_date": string|null,
"police_station": string|null,
"district": string|null,
"investigating_officer": string|null,
"legal_sections": [string],
"complainant_name": string|null,
"accused_names": [string],
"witness_names": [string],
"incident_date": string|null,
"incident_date_time": string|null,
"incident_location": string|null,
"document_narrative": string|null,
"investigation_actions_findings": [string]
}
Use null for missing single values and [] for missing lists.
No commentary."""

def _parse_metadata_json(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        raise GroqMetadataError("Empty response from Groq")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise GroqMetadataError("Groq response was not valid JSON") from None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            raise GroqMetadataError("Groq response was not valid JSON") from None
    if not isinstance(parsed, dict):
        raise GroqMetadataError("Groq response had unexpected structure")
    return parsed

async def extract_metadata(ocr_text: str) -> OCRMetadata:
    """Extract FIR and police document metadata from existing OCR text."""
    if not ocr_text or not ocr_text.strip():
        raise GroqMetadataError("No OCR text available for metadata extraction")
    settings = get_settings()
    try:
        client = _get_client()
    except GroqOCRError as exc:
        raise GroqMetadataError(str(exc)) from exc
    user_content = "\n\n".join([_METADATA_JSON_SHAPE, "<<<OCR_TEXT_START>>>"]) + "\n" + ocr_text + "\n<<<OCR_TEXT_END>>>"
    est_tokens = (len(METADATA_PROMPT) + len(user_content)) // 4
    try:
        response = await client.with_options(
            max_retries=0, timeout=_METADATA_TIMEOUT_SECONDS
        ).chat.completions.create(
            model=settings.groq_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": METADATA_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        choice = response.choices[0]
        content = choice.message.content
        finish = getattr(choice, "finish_reason", None)
    except Exception as exc:
        status, category, detail = describe_provider_error(exc)
        headers = getattr(getattr(exc, "response", None), "headers", None)
        retry_after = headers.get("retry-after") if headers is not None else None
        logger.error(
            "Groq metadata request failed: model=%s exc=%s http_status=%s category=%s "
            "retry_after=%s input_chars=%d est_tokens=%d detail=%s",
            settings.groq_model, type(exc).__name__, status, category, retry_after,
            len(ocr_text), est_tokens, detail,
        )
        raise GroqMetadataError("Groq metadata request failed") from exc
    try:
        return OCRMetadata.model_validate(_parse_metadata_json(content))
    except (GroqMetadataError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            category = "schema_validation fields=%s" % sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
        elif finish == "length":
            category = "truncated_output"
        else:
            category = "empty_response" if not (content or "").strip() else "invalid_json"
        logger.error(
            "Groq metadata output invalid: model=%s category=%s finish_reason=%s response_chars=%d",
            settings.groq_model, category, finish, len(content or ""),
        )
        raise GroqMetadataError("Groq returned invalid metadata") from exc
