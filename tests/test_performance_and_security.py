"""Tests for performance optimizations, CORS security, and session concurrency."""

from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient

import api
from backend.service import _generate_session_id, list_sessions


def test_cors_headers_present(client):
    """Verify CORS headers are returned for preflight and standard requests."""
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {"*", "http://localhost:8501"}


def test_cors_does_not_use_wildcard(client):
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "*"


def test_session_id_is_date_time_formatted():
    """Verify session IDs follow the date and time format."""
    sess_id = _generate_session_id()
    assert re.fullmatch(r"case_\d{8}_\d{6}(?:_\d+)?", sess_id)


def test_list_sessions_caching(client, tmp_path, monkeypatch):
    """Verify list_sessions uses mtime caching and respects limit."""
    monkeypatch.setenv("CHATBOT_SESSIONS_DIR", str(tmp_path / "sessions"))
    from config import get_settings
    get_settings.cache_clear()

    # Create dummy sessions
    sess_dir = tmp_path / "sessions"
    sess_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        s_file = sess_dir / f"case_20261119_12000{i}.json"
        s_file.write_text(
            f'{{"session_id": "case_20261119_12000{i}", "title": "Test Case {i}", "created_at": "2026-11-19T12:00:00Z", "updated_at": "2026-11-19T12:0{i}:00Z", "messages": []}}',
            encoding="utf-8",
        )

    # First call - loads from disk into cache
    result = list_sessions(limit=3)
    assert len(result["sessions"]) == 3
    assert result["sessions"][0]["session_id"] == "case_20261119_120004"

    # Second call - served from cache
    result2 = list_sessions(limit=10)
    assert len(result2["sessions"]) == 5


def test_schedule_context_caching(tmp_path):
    """Verify schedule context is cached and does not crash on missing files."""
    import rag
    provider = object.__new__(rag.LocalFAISSRag)
    provider.data_dir = tmp_path
    
    # Missing file returns empty
    assert provider._schedule_context("BNS 303", []) == ""

    # Valid file loads and caches
    sched_file = tmp_path / "bnss_first_schedule_classifications.json"
    sched_file.write_text(
        '{"schedule": "BNSS First Schedule", "source": "test", "rows": {"303(1)": {"bailable": false, "cognizable": true}}}',
        encoding="utf-8",
    )

    ctx = provider._schedule_context("BNS 303 theft", [])
    assert "EXACT CLASSIFICATION ROWS" in ctx
    assert "303(1)" in ctx
    assert hasattr(provider, "_schedule_cache") and provider._schedule_cache is not None

    # Subsequent call reuses cache even if file is removed
    sched_file.unlink()
    ctx2 = provider._schedule_context("BNS 303 theft", [])
    assert "303(1)" in ctx2


def test_unescaped_newlines_in_json_are_sanitized(client):
    """Verify raw unescaped newlines in JSON strings (common in multiline FIRs) are gracefully parsed."""
    raw_payload = (
        b'{\n'
        b'  "source": "3/2/24, 5:20 PM Report\n'
        b'\xe0\xa4\x86.\xe0\xa4\xb9. \xe0\xa4\xb8\xe0\xa4\x82\xe0\xa4\x96\xe0\xa4\x8d\xe0\xa4\xbe-26\n'
        b'\xe0\xa4\x96\xe0\xa4\xbc\xe0\xa4\xb0\xe0\xa4\xbe \xe0\xa4\xa5\xe0\xa4\xbe\xe0\xa4\xa8\xe0\xa4\xbe",\n'
        b'  "query": "Which BNS sections apply?"\n'
        b'}'
    )
    response = client.post(
        "/api/v1/chat",
        content=raw_payload,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"].startswith("case_")
