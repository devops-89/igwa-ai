"""MOCKED tests for metadata provider diagnostics. No live API calls."""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from services.ocr import gemini_key_manager as key_manager
from services.ocr import gemini_client, groq_client
from services.ocr.gemini_client import GeminiMetadataError, describe_provider_error
from services.ocr.groq_client import GroqMetadataError

SECRET = "SECRET-PERSON-NAME-Ravi-Khanna"


class FakeAPIError(Exception):
    def __init__(self, status, msg="boom", headers=None):
        super().__init__(msg)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


def gemini_with(monkeypatch, text=None, exc=None, sent=None):
    async def generate_content(**k):
        if sent is not None:
            sent.append(k["contents"][0])
        if exc:
            raise exc
        return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason="STOP")])

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(key_manager, "_manager", key_manager.GeminiKeyManager(
        ["AIzaSyTESTKEY000000000000000000000001"], client_factory=lambda k, p: client))
    monkeypatch.setattr(gemini_client, "get_settings", lambda: SimpleNamespace(gemini_model="gem"))


def groq_with(monkeypatch, content=None, exc=None, finish="stop", sent=None):
    async def create(**k):
        if sent is not None:
            sent.append(k)
        if exc:
            raise exc
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)])

    client = SimpleNamespace(with_options=lambda **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    monkeypatch.setattr(groq_client, "get_settings", lambda: SimpleNamespace(groq_model="grq"))


@pytest.mark.parametrize("status,cat", [(401, "auth"), (404, "model_not_found"), (429, "rate_limit"),
                                        (400, "bad_request"), (503, "server_error"), (413, "payload_too_large")])
def test_error_categories(status, cat):
    assert describe_provider_error(FakeAPIError(status))[:2] == (status, cat)


def test_gemini_success(monkeypatch):
    gemini_with(monkeypatch, text='{"fir_number": "F1"}')
    assert asyncio.run(gemini_client.extract_metadata("t")).fir_number == "F1"


def test_gemini_http_error_logged_safely(monkeypatch, caplog):
    gemini_with(monkeypatch, exc=FakeAPIError(404, "model not found"))
    with caplog.at_level(logging.ERROR), pytest.raises(GeminiMetadataError):
        asyncio.run(gemini_client.extract_metadata(SECRET))
    log = caplog.text
    assert "http_status=404" in log and "category=model_not_found" in log and "model=gem" in log
    assert "input_chars=" in log and "est_tokens=" in log and SECRET not in log


@pytest.mark.parametrize("text,cat", [("not json", "invalid_json"), ("", "empty_response"),
                                      ('{"legal_sections": "x"}', "schema_validation")])
def test_gemini_bad_output(monkeypatch, caplog, text, cat):
    gemini_with(monkeypatch, text=text)
    with caplog.at_level(logging.ERROR), pytest.raises(GeminiMetadataError):
        asyncio.run(gemini_client.extract_metadata("t"))
    assert cat in caplog.text


def test_groq_rate_limit_logged_with_retry_after(monkeypatch, caplog):
    groq_with(monkeypatch, exc=FakeAPIError(429, "tpm", {"retry-after": "12"}))
    with caplog.at_level(logging.ERROR), pytest.raises(GroqMetadataError):
        asyncio.run(groq_client.extract_metadata(SECRET))
    assert "category=rate_limit" in caplog.text and "retry_after=12" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize("content,finish,cat", [("garbage", "stop", "invalid_json"),
                                                ('{"accused_names": 5}', "stop", "schema_validation"),
                                                ('{"fir_number": "F', "length", "truncated_output"),
                                                ("", "stop", "empty_response")])
def test_groq_bad_output(monkeypatch, caplog, content, finish, cat):
    groq_with(monkeypatch, content=content, finish=finish)
    with caplog.at_level(logging.ERROR), pytest.raises(GroqMetadataError):
        asyncio.run(groq_client.extract_metadata("t"))
    assert cat in caplog.text


def test_large_input_sent_in_full_no_truncation(monkeypatch):
    big = "FIR line. " * 50_000 + "END-MARKER"
    g, q = [], []
    gemini_with(monkeypatch, text="{}", sent=g)
    groq_with(monkeypatch, content="{}", sent=q)
    asyncio.run(gemini_client.extract_metadata(big))
    asyncio.run(groq_client.extract_metadata(big))
    assert big in g[0] and "END-MARKER" in q[0]["messages"][1]["content"]
    assert q[0]["messages"][1]["content"].count("FIR line.") == 50_000
