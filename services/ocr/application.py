"""Cut the attached application out of the OCR pages (pure Python, no model calls).

The model only reports WHERE the application is (page range, plus optional short quotes when it
starts/ends mid-page). The text is always sliced from the existing pages[] OCR text, so nothing is
regenerated, translated or duplicated. Cost is O(P + A): one pass over the pages, one join.
"""
import logging
import re
import time

from services.ocr.schemas import ApplicationLocation, ApplicationResult, OCRMetadata, OCRPage

logger = logging.getLogger(__name__)

NOT_FOUND = ApplicationResult(found=False)


def _valid_page(value: object) -> bool:
    return type(value) is int and value >= 1


def _find(text: str, quote: str, *, last: bool) -> tuple[int, int] | None:
    """Span of a verbatim quote in `text`: exact first, then whitespace-tolerant. None if absent."""
    quote = quote.strip()
    if not quote:
        return None
    idx = text.rfind(quote) if last else text.find(quote)
    if idx != -1:
        return idx, idx + len(quote)
    pattern = r"\s+".join(re.escape(tok) for tok in quote.split())
    spans = [m.span() for m in re.finditer(pattern, text)]
    if not spans:
        return None
    return spans[-1] if last else spans[0]


def build_marked_text(pages: list[OCRPage]) -> str:
    """OCR text for the metadata model with page markers, so it can report page numbers.

    Only used for the background metadata request (Groq OCR / metadata fallback). The stored
    full_text is never changed.
    """
    ordered = sorted(pages, key=lambda p: p.page_number)
    return "\n\n".join(f"[[PAGE {p.page_number}]]\n{p.text}" for p in ordered)


def extract_application(pages: list[OCRPage], location: ApplicationLocation | None) -> ApplicationResult:
    """Application text for a location reported by the model.

    Raises ValueError for an invalid/unusable page range (callers treat that as "unresolved", never
    as "absent"). Whole-page applications are a plain join of the pages. With start_text/end_text the
    first/last page is cut at the quote; a quote that cannot be found verbatim yields found=True with
    text=None rather than returning unrelated FIR text.
    """
    if location is None or not location.found:
        return NOT_FOUND
    start, end = location.start_page, location.end_page
    if not _valid_page(start) or not _valid_page(end) or end < start:
        raise ValueError("Invalid application page range")

    by_number = {p.page_number: p.text for p in pages}
    if any(n not in by_number for n in range(start, end + 1)):
        raise ValueError("Missing application pages")
    texts = [by_number[n] for n in range(start, end + 1)]

    if location.start_text:
        span = _find(texts[0], location.start_text, last=False)
        if span is None:
            return ApplicationResult(found=True, start_page=start, end_page=end, text=None)
        texts[0] = texts[0][span[0]:]
    if location.end_text:
        span = _find(texts[-1], location.end_text, last=True)
        if span is None:
            return ApplicationResult(found=True, start_page=start, end_page=end, text=None)
        texts[-1] = texts[-1][: span[1]]
    return ApplicationResult(found=True, start_page=start, end_page=end, text="\n\n".join(texts))


def attach_application(metadata: OCRMetadata, pages: list[OCRPage], job_id: str) -> OCRMetadata:
    """Return metadata with `application` filled from the pages.

    No location from the model, or an unusable one, leaves application as None (unresolved) so it is
    never confused with a confirmed "no application" (found=False). Only counts/timings are logged,
    never application text.
    """
    location = metadata.application_location
    if location is None:
        logger.warning("job=%s Application location missing from model output: application not determined", job_id)
        return metadata
    started = time.perf_counter()
    try:
        application = extract_application(pages, location)
    except ValueError as exc:
        logger.warning("job=%s Application location unusable (%s; start=%s end=%s): application not determined",
                       job_id, exc, location.start_page, location.end_page)
        return metadata
    logger.info("job=%s Application extraction: found=%s pages=%s-%s chars=%d resolved=%s python=%.4fs",
                job_id, application.found, application.start_page, application.end_page,
                len(application.text or ""), application.text is not None or not application.found,
                time.perf_counter() - started)
    return metadata.model_copy(update={"application": application})
