import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from config.settings import get_settings
from services.ocr import gemini_key_manager as key_manager
from services.ocr import gemini_client, groq_client, service
from services.ocr.gemini_client import GeminiMetadataError, GeminiOCRError
from services.ocr.groq_client import GroqMetadataError, GroqOCRError
from services.ocr.schemas import MetadataStatus, OCRMetadata

_REAL_GEMINI_META = gemini_client.extract_metadata
RAW = {"pages": [{"page_number": 1, "text": "FIR 1/2026"}], "full_text": "FIR 1/2026"}
EMPTY = {"pages": [{"page_number": 1, "text": ""}], "full_text": ""}


class Calls(dict):
    pass


@pytest.fixture
def env(monkeypatch):
    calls = Calls(g_ocr=0, q_ocr=0, g_meta=0, q_meta=0)
    cfg = get_settings().model_copy(update=dict(
        gemini_model="gem", groq_model="grq", max_file_size_bytes=10**6,
        metadata_timeout_seconds=5))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    monkeypatch.setattr(service.utils, "sanitize_filename", lambda n: n)
    monkeypatch.setattr(service.utils, "validate_upload_metadata", lambda n, c: "pdf")
    async def read(f, m): return b"x"
    monkeypatch.setattr(service.utils, "read_upload_limited", read)
    monkeypatch.setattr(service.utils, "validate_content_matches_extension", lambda d, e: "application/pdf")
    monkeypatch.setattr(service.utils, "preprocess_document", lambda d, m: (d, m))
    service.ocr_results.clear()

    def setup(g_ocr=RAW, q_ocr=RAW, g_meta=None, q_meta=None):
        async def go(d, m):
            calls["g_ocr"] += 1
            if g_ocr is None: raise GeminiOCRError("x")
            return g_ocr
        async def qo(d, m):
            calls["q_ocr"] += 1
            if q_ocr is None: raise GroqOCRError("x")
            return q_ocr
        async def gm(t):
            calls["g_meta"] += 1
            if g_meta is None: raise GeminiMetadataError("x")
            return g_meta
        async def qm(t):
            calls["q_meta"] += 1
            if q_meta is None: raise GroqMetadataError("x")
            return q_meta
        monkeypatch.setattr(gemini_client, "extract_text", go)
        monkeypatch.setattr(groq_client, "extract_text", qo)
        monkeypatch.setattr(gemini_client, "extract_metadata", gm)
        monkeypatch.setattr(groq_client, "extract_metadata", qm)
        return calls

    setup.calls = calls
    return setup


def upload():
    """Upload, run the scheduled background task, and return the final stored record."""
    f = SimpleNamespace(filename="a.pdf", content_type="application/pdf")
    tasks = BackgroundTasks()

    async def go():
        resp = await service.process_upload(f, tasks)
        await tasks()
        return service.get_ocr_result(resp.document_id)

    return asyncio.run(go())


M = OCRMetadata(fir_number="FIR 1/2026")


def test_gemini_both(env):
    c = env(g_meta=M); r = upload()
    assert r.ocr_model == "gem" and r.metadata_model == "gem" and r.metadata == M
    assert c == dict(g_ocr=1, q_ocr=0, g_meta=1, q_meta=0)


def test_groq_ocr_gemini_meta(env):
    c = env(g_ocr=None, g_meta=M); r = upload()
    assert r.ocr_model == "grq" and r.metadata_model == "gem"
    assert c == dict(g_ocr=1, q_ocr=1, g_meta=1, q_meta=0)


def test_groq_meta_fallback(env):
    c = env(q_meta=M); r = upload()
    assert r.ocr_model == "gem" and r.metadata_model == "grq" and r.metadata_status == MetadataStatus.COMPLETED
    assert c == dict(g_ocr=1, q_ocr=0, g_meta=1, q_meta=1)


def test_all_groq(env):
    c = env(g_ocr=None, q_meta=M); r = upload()
    assert r.ocr_model == "grq" and r.metadata_model == "grq"
    assert c == dict(g_ocr=1, q_ocr=1, g_meta=1, q_meta=1)


def test_both_meta_fail_keeps_ocr(env):
    env(); r = upload()
    assert r.metadata is None and r.metadata_status == MetadataStatus.FAILED
    assert r.metadata_model is None and r.ocr_data.full_text == "FIR 1/2026"


def test_empty_ocr_skips(env):
    c = env(g_ocr=EMPTY, g_meta=M); r = upload()
    assert r.metadata_status == MetadataStatus.SKIPPED
    assert c["g_meta"] == 0 and c["q_meta"] == 0


def test_invalid_gemini_metadata_triggers_groq(env, monkeypatch):
    real = _REAL_GEMINI_META

    class Bad:
        text = "not json"

    async def generate_content(**k):
        return Bad()

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    c = env(q_meta=M)

    async def wrapper(t):
        c["g_meta"] += 1
        return await real(t)

    monkeypatch.setattr(gemini_client, "extract_metadata", wrapper)
    monkeypatch.setattr(key_manager, "_manager", key_manager.GeminiKeyManager(
        ["AIzaSyTESTKEY000000000000000000000001"], client_factory=lambda k, p: client))
    monkeypatch.setattr(gemini_client, "get_settings", lambda: SimpleNamespace(gemini_model="gem"))
    r = upload()
    assert r.metadata_model == "grq" and c["q_meta"] == 1 and c["g_meta"] == 1


def _groq_with(monkeypatch, content, raises=False):
    async def create(**k):
        if raises: raise RuntimeError("429")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    client = SimpleNamespace()
    client.with_options = lambda **k: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    monkeypatch.setattr(groq_client, "get_settings", lambda: SimpleNamespace(groq_model="grq"))


def test_groq_valid_json_parsed(monkeypatch):
    _groq_with(monkeypatch, '```json\n{"fir_number": "F1", "legal_sections": ["303(2)"]}\n```')
    m = asyncio.run(groq_client.extract_metadata("text"))
    assert m.fir_number == "F1" and m.legal_sections == ["303(2)"]


@pytest.mark.parametrize("content", ["garbage", '{"legal_sections": "oops"}', "", "[1]"])
def test_groq_invalid_output_safe_failure(monkeypatch, content):
    _groq_with(monkeypatch, content)
    with pytest.raises(GroqMetadataError):
        asyncio.run(groq_client.extract_metadata("text"))


def test_groq_api_error_safe(monkeypatch):
    _groq_with(monkeypatch, "", raises=True)
    with pytest.raises(GroqMetadataError):
        asyncio.run(groq_client.extract_metadata("text"))


def test_get_uses_store_without_provider_calls(env):
    c = env(q_meta=M); r = upload()
    before = dict(c)
    got = service.get_ocr_result(r.document_id)
    assert got.metadata == M and got.metadata_model == "grq" and dict(c) == before
