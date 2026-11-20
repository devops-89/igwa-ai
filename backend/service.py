
"""Chatbot business logic and integrations with RAG and local session storage."""

from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any
import uuid

from fastapi import HTTPException

from backend.config import get_settings
from backend.schema import ChatRequest
from backend.guardrails import (
    GUARDRAIL_REFUSAL_MESSAGE,
    is_degenerate_output,
    is_police_investigation_query,
)

logger = logging.getLogger("chatbot.api")

_audit_lock = threading.Lock()
_session_lock = threading.Lock()
_session_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")

_DETAILED_PROCEDURE_QUERY = re.compile(
    r"\b(?:detailed\s+)?(?:investigation|investigative)\s+"
    r"(?:procedure|process|steps|plan|sop)\b|"
    r"\b(?:sop|step[- ]by[- ]step|checklist)\b|"
    r"\bhow\s+to\s+investigate\b|"
    r"(?:विवेचना|जांच|अनुसंधान).{0,15}(?:प्रक्रिया|चरण|कार्यवाही)",
    re.IGNORECASE,
)

_SECTION_EXPLANATION_QUERY = re.compile(
    r"\b(?:explain|explanation|meaning|describe|elaborate)\b|"
    r"(?:समझाइए|समझाएं|व्याख्या|अर्थ)",
    re.IGNORECASE,
)

_LEGAL_SECTION_REFERENCE = re.compile(
    r"\b(?:section|sec\.?|धारा)\s*(?:no\.?|number|#)?\s*\d+|"
    r"\b(?:BNS|BNSS|BSB|CrPC)\s*(?:(?:section|sec\.?|धारा)\s*)?\d+|"
    r"\b(?:BNS|BNSS|BSB|CrPC)\b|धारा",
    re.IGNORECASE,
)

_CASE_ANALYSIS_QUERY = re.compile(
    r"\b(?:analy[sz]e|analysis|assess|evaluate|case\s+facts|case\s+analysis|"
    r"what\s+charges|which\s+sections?\s+apply|what\s+sections?\s+apply|"
    r"applicable\s+sections?|incident|accused|victim|FIR\s*(?:no\.?\s*)?[\w/-]+)\b|"
    r"(?:मामले|प्रकरण).{0,12}(?:विश्लेषण|आकलन)|"
    r"(?:कौन सी|कौन-कौन सी).{0,15}धाराएं.{0,15}लागू|"
    r"(?:आरोपी|अभियुक्त|पीड़ित|घटना)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Output token selection
# ---------------------------------------------------------------------

def get_output_token_limit(
    query: str,
    *,
    has_case_context: bool = False,
) -> int:
    """Choose a response budget from the request's intent."""

    if _DETAILED_PROCEDURE_QUERY.search(query):
        return 1200

    if (
        _SECTION_EXPLANATION_QUERY.search(query)
        and _LEGAL_SECTION_REFERENCE.search(query)
    ):
        return 400

    if _CASE_ANALYSIS_QUERY.search(query):
        return 800

    if has_case_context and re.search(
        r"\b(?:analy[sz]e|analysis|assess|evaluate)\b",
        query,
        re.IGNORECASE,
    ):
        return 800

    return 150


# ---------------------------------------------------------------------
# API errors and RAG provider
# ---------------------------------------------------------------------

class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "code": code,
                "message": message,
            },
        )


@lru_cache(maxsize=1)
def get_rag_provider():
    """Initialize the RAG provider once."""

    from rag import LocalFAISSRag

    return LocalFAISSRag()


# ---------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------

def _audit(
    request_id: str,
    action: str,
    **fields: Any,
) -> None:
    """Write an audit event without logging sensitive case contents."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "action": action,
        **fields,
    }

    path = Path(get_settings().audit_log_path)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with _audit_lock, path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )

    except OSError:
        logger.exception(
            "Could not write chatbot audit event %s.",
            action,
        )


# ---------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------

def _generate_session_id() -> str:
    """Generate a clean, compact session ID based on date and time (e.g. case_20261119_153647)."""
    base = datetime.now().strftime("case_%Y%m%d_%H%M%S")
    candidate = base
    suffix = 1
    while _session_path(candidate).exists():
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _session_path(session_id: str) -> Path:
    """Return the storage path after validating the session ID."""

    if not _SESSION_ID.fullmatch(session_id):
        raise APIError(
            400,
            "invalid_session_id",
            "Invalid session ID.",
        )

    return Path(get_settings().sessions_dir) / f"{session_id}.json"


def _load_session(
    session_id: str,
) -> dict[str, Any] | None:
    """Load an existing chat session."""

    path = _session_path(session_id)

    if not path.is_file():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError):
        logger.exception(
            "Could not read chatbot session %s.",
            session_id,
        )

        raise APIError(
            500,
            "session_read_failed",
            "The saved session could not be read.",
        )


def _save_session(
    session_id: str,
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    case_facts: str | None,
    guardrail: bool,
    warn: bool,
    document_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save chat messages and associated document information."""

    path = _session_path(session_id)

    try:
        with _session_lock:
            path.parent.mkdir(parents=True, exist_ok=True)

            now = datetime.now(timezone.utc).isoformat()

            session = _load_session(session_id)

            if session is None:
                session = {
                    "session_id": session_id,
                    "title": query[:60],
                    "created_at": now,
                    "messages": [],
                }

            session.setdefault("messages", [])

            session["updated_at"] = now

            # Preserve the existing case context unless a new one is supplied.
            if case_facts:
                session["source"] = case_facts
                session["case_facts"] = case_facts

            # Associate the session with the uploaded document.
            if document_id:
                session["document_id"] = document_id

            # Preserve structured OCR metadata.
            if metadata:
                existing_metadata = session.get("metadata", {})
                if not isinstance(existing_metadata, dict):
                    existing_metadata = {}

                session["metadata"] = {
                    **existing_metadata,
                    **metadata,
                }

            session["messages"].extend(
                [
                    {
                        "role": "user",
                        "content": query,
                        "timestamp": now,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "guardrail": guardrail,
                        "warn": warn,
                        "timestamp": now,
                    },
                ]
            )

            path.write_text(
                json.dumps(
                    session,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    except APIError:
        raise

    except OSError as exc:
        logger.exception(
            "Could not save chatbot session %s.",
            session_id,
        )

        raise APIError(
            500,
            "session_write_failed",
            "The chat response could not be saved.",
        ) from exc


# ---------------------------------------------------------------------
# OCR and case context preparation
# ---------------------------------------------------------------------

def _metadata_to_dict(
    metadata: Any,
) -> dict[str, Any]:
    """Convert Pydantic metadata or a dictionary into a dictionary."""

    if metadata is None:
        return {}

    if hasattr(metadata, "model_dump"):
        return metadata.model_dump(exclude_none=True)

    if isinstance(metadata, dict):
        return metadata

    return {}


def build_case_context(
    payload: ChatRequest,
) -> str | None:
    """
    Build context from source data, case facts, document text,
    or structured metadata.
    """

    context_parts: list[str] = []

    # 1. Primary source data supplied by frontend
    if getattr(payload, "source", None) and payload.source.strip():
        context_parts.append(
            "SOURCE DATA:\n" + payload.source.strip()
        )

    # 2. Case facts supplied by the frontend
    if getattr(payload, "case_facts", None) and payload.case_facts.strip():
        context_parts.append(
            "CASE FACTS:\n" + payload.case_facts.strip()
        )

    if getattr(payload, "summary", None) and payload.summary.strip():
        context_parts.append(
            "CONVERSATION SUMMARY:\n" + payload.summary.strip()
        )

    # 3. Document text or OCR data (if provided)
    document_text = getattr(payload, "document_text", None) or ""

    if not document_text and getattr(payload, "ocr_data", None) is not None:
        document_text = (
            payload.ocr_data.full_text.strip()
            or "\n\n".join(
                page.text.strip()
                for page in payload.ocr_data.pages
                if page.text.strip()
            )
        )

    if document_text.strip():
        context_parts.append(
            "EXTRACTED DOCUMENT TEXT:\n"
            + document_text.strip()
        )

    # 4. Structured metadata (if provided)
    metadata = _metadata_to_dict(
        getattr(payload, "metadata", None)
    )

    if metadata:
        metadata_lines = []

        for key, value in metadata.items():
            if value in (None, "", []):
                continue

            label = key.replace("_", " ").title()

            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)

            metadata_lines.append(
                f"{label}: {value}"
            )

        if metadata_lines:
            context_parts.append(
                "STRUCTURED CASE METADATA:\n"
                + "\n".join(metadata_lines)
            )

    if not context_parts:
        return None

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------
# Chat history preparation
# ---------------------------------------------------------------------

def _get_session_history(
    saved_session: dict[str, Any] | None,
    payload: ChatRequest,
) -> list[dict[str, str]]:
    """
    Prefer saved backend history for an existing session.
    Use request history only when there is no saved session.
    """

    if saved_session is not None:
        messages = saved_session.get("messages", [])

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in messages[-6:]
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]

    return [
        message.model_dump()
        for message in payload.chat_history
    ]


# ---------------------------------------------------------------------
# Main chat business logic
# ---------------------------------------------------------------------

def chat(
    payload: ChatRequest,
    request_id: str,
    *,
    persist_session: bool = True,
) -> dict[str, Any]:

    session_id = payload.session_id
    saved_session = None

    # 1. Load or create session
    if persist_session:

        if session_id:
            saved_session = _load_session(session_id)

            if saved_session is None:
                raise APIError(
                    404,
                    "session_not_found",
                    "Session not found.",
                )

        else:
            session_id = _generate_session_id()

    # 2. Prepare document and metadata information
    context = build_case_context(payload)

    incoming_metadata = _metadata_to_dict(
        getattr(payload, "metadata", None)
    )

    # Reuse previously stored context for follow-up questions.
    if saved_session is not None:

        if context is None:
            context = saved_session.get("source") or saved_session.get("case_facts")

        # Reuse stored metadata if the current request has none.
        if not incoming_metadata:
            incoming_metadata = saved_session.get(
                "metadata", {}
            )

        # Reuse the existing document association.
        document_id = (
            getattr(payload, "document_id", None)
            or saved_session.get("document_id")
        )

    else:
        document_id = getattr(
            payload,
            "document_id",
            None,
        )

    # 3. Build scenario and follow-up query
    scenario = context or payload.query

    follow_up = (
        payload.query
        if context
        else None
    )

    # 4. Select output token limit
    output_token_limit = get_output_token_limit(
        payload.query,
        has_case_context=bool(context),
    )

    # 5. Load conversation history
    history = _get_session_history(
        saved_session,
        payload,
    )

    # 6. Apply guardrails
    guardrail_text = " ".join(
        [
            payload.query,
            context or "",
            *(
                message["content"]
                for message in history
            ),
        ]
    )

    if not is_police_investigation_query(guardrail_text):

        answer = GUARDRAIL_REFUSAL_MESSAGE

        result = {
            "answer": answer,
            "sources": [],
            "warn": False,
            "guardrail": True,
            "source": context,
            "case_facts": context,
        }

        if persist_session:
            assert session_id is not None

            _save_session(
                session_id=session_id,
                query=payload.query,
                answer=answer,
                sources=[],
                case_facts=context or payload.query,
                guardrail=True,
                warn=False,
                document_id=document_id,
                metadata=incoming_metadata,
            )

            result["session_id"] = session_id

        _audit(
            request_id,
            "chat.guardrail",
            document_id=document_id,
        )

        return result

    # 7. Generate response through RAG
    try:

        answer, retrieved_sources, provider = (
            get_rag_provider().analyze_gemini(
                scenario,
                follow_up=follow_up,
                allow_groq_fallback=True,
                chat_history=history,
                max_output_tokens=output_token_limit,
            )
        )

    except Exception as exc:

        logger.exception(
            "Chat analysis failed for request %s.",
            request_id,
        )

        _audit(
            request_id,
            "chat.failed",
            error_type=type(exc).__name__,
            document_id=document_id,
        )

        raise APIError(
            502,
            "upstream_failure",
            "The chatbot could not complete this request.",
        ) from exc

    # 8. Format retrieved sources
    sources = []

    for index, source in enumerate(
        retrieved_sources or [],
        start=1,
    ):

        if not isinstance(source, dict):
            continue

        sources.append(
            {
                "idx": index,
                "file": source.get(
                    "source",
                    source.get("file", "अज्ञात स्रोत"),
                ),
                "snippet": str(
                    source.get("text", "")
                ).replace("\n", " ")[:200],
            }
        )

    # 9. Check response quality
    warn = is_degenerate_output(answer)

    result = {
        "answer": answer,
        "sources": sources,
        "warn": warn,
        "guardrail": False,
        "source": scenario if context else None,
        "case_facts": scenario,
        "model": provider,
    }

    # 10. Persist conversation
    if persist_session:

        assert session_id is not None

        _save_session(
            session_id=session_id,
            query=payload.query,
            answer=answer,
            sources=sources,
            case_facts=scenario,
            guardrail=False,
            warn=warn,
            document_id=document_id,
            metadata=incoming_metadata,
        )

        result["session_id"] = session_id

    # 11. Audit successful request
    _audit(
        request_id,
        "chat.completed",
        source_count=len(sources),
        guardrail=False,
        document_id=document_id,
        provider=provider,
    )

    return result


# ---------------------------------------------------------------------
# Session APIs
# ---------------------------------------------------------------------

def list_sessions(limit: int = 50) -> dict[str, Any]:
    """Return saved chat sessions with fast mtime-based caching."""

    directory = Path(get_settings().sessions_dir)
    sessions = []

    if directory.exists():
        with _session_lock:
            for path in directory.glob("*.json"):
                try:
                    stat = path.stat()
                    cache_signature = (stat.st_mtime_ns, stat.st_size)
                    cached = _session_meta_cache.get(path.name)
                    if cached and cached[0] == cache_signature:
                        summary = cached[1]
                    else:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        summary = {
                            "session_id": data.get("session_id", path.stem),
                            "title": data.get("title", "अनाम प्रकरण"),
                            "created_at": data.get("created_at"),
                            "updated_at": data.get("updated_at"),
                            "message_count": len(data.get("messages", [])),
                            "source": data.get("source") or data.get("case_facts"),
                            "case_facts": data.get("case_facts"),
                            "document_id": data.get("document_id"),
                        }
                        _session_meta_cache[path.name] = (cache_signature, summary)
                    sessions.append(summary)
                except (OSError, json.JSONDecodeError):
                    logger.exception("Could not read session file %s; skipping.", path)
                    continue

    sessions.sort(
        key=lambda session: session.get("updated_at") or "",
        reverse=True,
    )

    if limit > 0:
        sessions = sessions[:limit]

    return {
        "sessions": sessions,
    }


def get_session(
    session_id: str,
) -> dict[str, Any]:
    """Return a complete saved chat session."""

    session = _load_session(session_id)

    if session is None:
        raise APIError(
            404,
            "session_not_found",
            "Session not found.",
        )

    return session


# ---------------------------------------------------------------------
# RAG health
# ---------------------------------------------------------------------

def is_rag_loaded() -> bool:
    """Check whether the RAG provider has been initialized."""

    return get_rag_provider.cache_info().currsize > 0
