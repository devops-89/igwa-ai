from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IGWA"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    gemini_api_key: str = ""
    # Ordered failover pool. Precedence: GEMINI_API_KEYS (comma list) > GEMINI_API_KEY_1..10 >
    # legacy GEMINI_API_KEY. See services/ocr/gemini_key_manager.py.
    gemini_api_keys: str = ""
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    gemini_api_key_4: str = ""
    gemini_api_key_5: str = ""
    gemini_api_key_6: str = ""
    gemini_api_key_7: str = ""
    gemini_api_key_8: str = ""
    gemini_api_key_9: str = ""
    gemini_api_key_10: str = ""
    gemini_request_timeout_seconds: float = Field(default=90.0, gt=0)  # per attempt
    gemini_overall_budget_seconds: float = Field(default=110.0, gt=0)  # whole failover chain
    gemini_shared_failure_cooldown_seconds: float = Field(default=10.0, ge=0)  # fast-fail after network/5xx
    gemini_rate_limit_cooldown_seconds: float = Field(default=60.0, ge=0)  # 429 without Retry-After
    gemini_quota_cooldown_seconds: float = Field(default=3600.0, ge=0)  # daily quota exhausted
    gemini_billing_cooldown_seconds: float = Field(default=3600.0, ge=0)  # 402
    gemini_forbidden_cooldown_seconds: float = Field(default=3600.0, ge=0)  # 403
    gemini_model: str = "gemini-3.1-flash-lite"

    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"

    max_file_size_mb: int = 20

    # --- OCR: Gemini (ordered key failover) -> Groq, whole document ---
    ocr_enable_gemini_fallback: bool = True  # False skips Gemini entirely
    ocr_enable_groq_fallback: bool = True
    ocr_gemini_max_retries: int = Field(default=1, ge=0, le=3)  # Gemini failover already bounds retries
    ocr_groq_max_retries: int = Field(default=1, ge=0, le=3)
    ocr_remote_timeout_seconds: float = Field(default=120.0, gt=0)

    # Per-provider cap for one background metadata attempt (Gemini and Groq each).
    metadata_timeout_seconds: float = 90.0

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
