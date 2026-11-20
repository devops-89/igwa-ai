"""Compatibility entry point for ``uvicorn api:app``."""

from backend.routes import app, create_app
from backend.schema import ChatRequest
from backend.service import APIError, get_rag_provider

__all__ = [
    "APIError",
    "ChatRequest",
    "app",
    "create_app",
    "get_rag_provider",
]
