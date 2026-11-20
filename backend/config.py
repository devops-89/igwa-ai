"""Settings, read once from environment variables (and .env if python-dotenv is installed)."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # so CHATBOT_* values in .env are visible before the app is created
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    audit_log_path: Path
    api_key: str | None      # if set, every request must send an X-API-Key header
    cors_origins: list[str]
    preload_rag: bool        # load FAISS/embeddings at startup, not on the first request
    sessions_dir: Path = Path("sessions")

    @property
    def chatbot_api_key(self) -> str | None:
        return self.api_key


@lru_cache
def get_settings() -> Settings:
    return Settings(
        audit_log_path=Path(os.getenv("CHATBOT_AUDIT_LOG", "query_log.jsonl")),
        api_key=os.getenv("CHATBOT_API_KEY") or None,
        cors_origins=[o.strip() for o in os.getenv("CHATBOT_CORS_ORIGINS", "").split(",") if o.strip()],
        preload_rag=os.getenv("CHATBOT_PRELOAD_RAG", "false").strip().lower() in {"1", "true", "yes", "on"},
        sessions_dir=Path(os.getenv("CHATBOT_SESSIONS_DIR", "sessions")),
    )
