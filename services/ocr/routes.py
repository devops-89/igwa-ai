from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from services.ocr import service
from services.ocr.schemas import (
    ErrorResponse,
    MetadataStatusResponse,
    OCRResultResponse,
    OCRUploadResponse,
)

router = APIRouter(prefix="/python/ocr", tags=["OCR"])

@router.post(
    "/upload",
    response_model=OCRUploadResponse,
    summary="Upload a document and extract its text with OCR",
    description="Accepts a PDF, PNG, JPG, JPEG or WEBP file (multipart/form-data), "
    "runs OCR, keeps the result temporarily in memory and returns it with a job_id. "
    "Metadata is extracted in the background: the response has metadata=null and "
    "metadata_status=pending; poll GET /python/ocr/{job_id}/metadata.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported or empty file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        502: {"model": ErrorResponse, "description": "OCR provider failure"},
    },
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or image to OCR"),
) -> OCRUploadResponse:
    return await service.process_upload(file, background_tasks)

@router.get(
    "/{job_id}/metadata",
    response_model=MetadataStatusResponse,
    summary="Get the background metadata extraction status/result for a job",
    responses={404: {"model": ErrorResponse, "description": "Job not found"}},
)
async def get_metadata(job_id: str) -> MetadataStatusResponse:
    return service.get_metadata_status(job_id)

@router.get(
    "/{document_id}",
    response_model=OCRResultResponse,
    summary="Retrieve an OCR result by document_id",
    responses={404: {"model": ErrorResponse, "description": "Document not found"}},
)
async def get_document(document_id: str) -> OCRResultResponse:
    return service.get_ocr_result(document_id)
