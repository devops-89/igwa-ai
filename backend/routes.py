"""HTTP routes and application factory for the chatbot API."""

import hmac
import json
import logging
import uuid

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.config import get_settings
from backend.schema import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionListResponse,
)
from backend.service import (
    APIError,
    chat,
    get_session,
    is_rag_loaded,
    list_sessions,
)

logger = logging.getLogger("chatbot.api")


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = get_settings().chatbot_api_key
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
        raise APIError(401, "unauthorized", "A valid API key is required.")


chat_router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key)])
session_router = APIRouter(
    prefix="/api/sessions",
    tags=["sessions"],
    dependencies=[Depends(require_api_key)],
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


@chat_router.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat_v1(
    request: Request,
    payload: ChatRequest = Body(
        ...,
        openapi_examples={
            "simple_chat": {
                "summary": "Simple legal chat",
                "description": "Send a direct legal question without external source context.",
                "value": {
                    "query": "What steps should police take after registering an FIR under BNSS?"
                },
            },
            "source_chat": {
                "summary": "Chat with Source Data",
                "description": "Provide source data (complaint, FIR, witness statement, or case facts) and ask a query based on it.",
                "value": {
                    "source": (
                        "Complainant alleges that on 15.01.2026 at 21:00 hrs, accused intercepted her vehicle, "
                        "snatched her purse containing Rs. 5000 and mobile phone, and threatened to kill her if reported."
                    ),
                    "query": "Which BNS sections are attracted and what immediate investigative steps must the IO take under BNSS?",
                },
            },
        },
    ),
):
    return chat(
        payload,
        _request_id(request),
        persist_session=True,
    )


@chat_router.post(
    "/api/chat",
    include_in_schema=False,
    response_model=ChatResponse,
    response_model_exclude_none=True,
)
def chat_legacy(payload: ChatRequest, request: Request):
    return chat(payload, _request_id(request), persist_session=True)


@session_router.get("", response_model=SessionListResponse)
def get_sessions():
    return list_sessions()


@session_router.get("/{session_id}")
def get_saved_session(session_id: str):
    return get_session(session_id)


def create_app() -> FastAPI:
    application = FastAPI(
        title="IGWA Legal Chatbot API",
        description="Versioned API for the legal investigation and case diary assistant.",
        version="1.0.0",
    )

    default_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]
    configured_origins = get_settings().cors_origins
    allowed_origins = list(dict.fromkeys(default_origins + configured_origins))
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if get_settings().preload_rag:
        @application.on_event("startup")
        async def preload_rag_provider() -> None:
            from backend.service import get_rag_provider

            get_rag_provider()

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID")
        if not request_id or len(request_id) > 128:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.middleware("http")
    async def sanitize_raw_json_middleware(request: Request, call_next):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.body()
                if body:
                    try:
                        json.loads(body)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Decode raw body permitting non-strict control characters (e.g. unescaped newlines/tabs)
                        decoded_text = body.decode("utf-8", errors="replace")
                        parsed = json.loads(decoded_text, strict=False)
                        # Re-encode into strictly valid standard JSON bytes
                        sanitized_body = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
                        request._body = sanitized_body
                        if hasattr(request, "_json"):
                            delattr(request, "_json")

                        async def receive():
                            return {"type": "http.request", "body": sanitized_body, "more_body": False}

                        request._receive = receive
            except Exception:
                # If truly malformed syntax, allow downstream RequestValidationError to handle it
                pass

        return await call_next(request)

    @application.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = {**detail, "request_id": _request_id(request)}
        else:
            body = {
                "code": "http_error",
                "message": str(detail),
                "request_id": _request_id(request),
            }
        return JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": "Request validation failed.",
                "errors": errors,
                "request_id": _request_id(request),
            },
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error for request %s.", _request_id(request))
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "request_id": _request_id(request),
            },
        )

    @application.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @application.get("/health", tags=["health"], response_model=HealthResponse)
    def health():
        return {"status": "ok", "rag_loaded": is_rag_loaded()}

    application.include_router(chat_router)
    application.include_router(session_router)
    return application


app = create_app()
