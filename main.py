import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config.settings import get_settings
from services.ocr import gemini_key_manager
from services.ocr.routes import router as ocr_router

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=f"{settings.app_name} OCR API",
    description="IGWA OCR service: upload a PDF/image, OCR it with Gemini Flash, "
    "keep the result temporarily in memory and fetch it by document_id.",
    version="0.1.0",
)



@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s: %s", request.url.path, type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "IGWA OCR API", "docs": "/docs"}


@app.get("/python/ocr/health", tags=["Health"], summary="Check the API is running")
@app.get("/health", include_in_schema=False)  # kept for load balancers / probes
async def health() -> dict[str, Any]:
    return {"status": "healthy", "service": "IGWA OCR API",
            # counts only: no slot details, reasons or credentials over HTTP
            "gemini_keys": gemini_key_manager.get_manager().counts()}

# Included last so fixed paths like /python/ocr/health are matched before /python/ocr/{document_id}.
app.include_router(ocr_router)
