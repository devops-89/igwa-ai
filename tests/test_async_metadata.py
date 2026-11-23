"""MOCKED tests for background metadata extraction. No live API calls."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr.gemini_client import GeminiMetadataError, GeminiOCRError
from services.ocr.groq_client import GroqMetadataError, GroqOCRError
from services.ocr.schemas import MetadataStatus, OCRMetadata

RAW = {"pages": [{"page_number": 1, "text": "FIR 1/2026"}], "full_text": "FIR 1/2026"}
EMPTY = {"pages": [{"page_number": 1, "text": ""}], "full_text": ""}
M = OCRMetadata(fir_number="FIR 1/2026")
FILE = SimpleNamespace(filename="a.pdf", content_type="application/pdf")


@pytest.fixture
def env(monkeypatch):
    calls = dict(g_meta=0, q_meta=0)
    cfg = get_settings().model_copy(update=dict(
        gemini_model="gem", groq_model="grq", max_file_size_bytes=10**6,
        metadata_timeout_seconds=0.2))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    monkeypatch.setattr(service.utils, "sanitize_filename", lambda n: n)
    monkeypatch.setattr(service.utils, "validate_upload_metadata", lambda n, c: "pdf")

    async def read(f, m):
        return b"x"

    monkeypatch.setattr(service.utils, "read_upload_limited", read)
    monkeypatch.setattr(service.utils, "validate_content_matches_extension", lambda d, e: "application/pdf")
    monkeypatch.setattr(service.utils, "preprocess_document", lambda d, m: (d, m))
    service.ocr_results.clear()

    def setup(ocr=RAW, g_meta=M, q_meta=M, g_ocr_fails=False, q_ocr_fails=False):
        async def go(d, m):
            if g_ocr_fails:
                raise GeminiOCRError("x")
            return ocr

        async def qo(d, m):
            if q_ocr_fails:
                raise GroqOCRError("x")
            return ocr

        async def gm(t):
            calls["g_meta"] += 1
            return await _resolve(g_meta, GeminiMetadataError)

        async def qm(t):
            calls["q_meta"] += 1
            return await _resolve(q_meta, GroqMetadataError)

        monkeypatch.setattr(gemini_client, "extract_text", go)
        monkeypatch.setattr(groq_client, "extract_text", qo)
        monkeypatch.setattr(gemini_client, "extract_metadata", gm)
        monkeypatch.setattr(groq_client, "extract_metadata", qm)
        return calls

    return setup


async def _resolve(value, err):
    """value: metadata | None (provider error) | "hang" | Exception | async callable."""
    if value is None:
        raise err("x")
    if value == "hang":
        await asyncio.sleep(10)
    if isinstance(value, Exception):
        raise value
    if callable(value):
        return await value()
    return value


def test_upload_returns_before_metadata_runs(env):
    """The endpoint must not await metadata: it only schedules it."""
    calls = env()

    async def scenario():
        tasks = BackgroundTasks()
        resp = await service.process_upload(FILE, tasks)
        assert resp.job_id == resp.document_id
        assert resp.metadata is None and resp.metadata_status == MetadataStatus.PENDING
        assert calls == dict(g_meta=0, q_meta=0)  # no provider touched yet
        assert len(tasks.tasks) == 1  # scheduled exactly once
        assert service.get_metadata_status(resp.job_id).metadata_status == MetadataStatus.PENDING
        await tasks()
        return resp.job_id

    job = asyncio.run(scenario())
    out = service.get_metadata_status(job)
    assert out.metadata_status == MetadataStatus.COMPLETED and out.metadata == M
    assert out.metadata_provider == "gem" and calls == dict(g_meta=1, q_meta=0)


def test_status_is_processing_while_running(env):
    seen = []

    async def probe():
        seen.append(next(iter(service.ocr_results.values())).metadata_status)
        return M

    env(g_meta=probe)

    async def scenario():
        tasks = BackgroundTasks()
        await service.process_upload(FILE, tasks)
        await tasks()

    asyncio.run(scenario())
    assert seen == [MetadataStatus.PROCESSING]


def test_empty_text_skips_and_schedules_nothing(env):
    calls = env(ocr=EMPTY)
    tasks = BackgroundTasks()
    resp = asyncio.run(service.process_upload(FILE, tasks))
    assert resp.metadata_status == MetadataStatus.SKIPPED
    assert tasks.tasks == [] and calls == dict(g_meta=0, q_meta=0)


def test_ocr_failure_schedules_nothing(env):
    calls = env(g_ocr_fails=True, q_ocr_fails=True)
    tasks = BackgroundTasks()
    with pytest.raises(HTTPException) as e:
        asyncio.run(service.process_upload(FILE, tasks))
    assert e.value.status_code == 502 and tasks.tasks == [] and calls == dict(g_meta=0, q_meta=0)
    assert service.ocr_results == {}


def test_groq_fallback_records_provider(env):
    calls = env(g_meta=None)
    job = _run_one()
    out = service.get_metadata_status(job)
    assert out.metadata_status == MetadataStatus.COMPLETED and out.metadata_provider == "grq"
    assert calls == dict(g_meta=1, q_meta=1)  # no duplicate extraction


def test_both_fail_is_terminal_failed_with_safe_error(env):
    env(g_meta=None, q_meta=None)
    out = service.get_metadata_status(_run_one())
    assert out.metadata_status == MetadataStatus.FAILED and out.metadata is None
    assert out.metadata_error == "Metadata extraction could not be completed."
    assert service.get_ocr_result(out.job_id).ocr_data.full_text == "FIR 1/2026"  # OCR kept


def test_gemini_timeout_falls_back_to_groq(env):
    env(g_meta="hang")
    out = service.get_metadata_status(_run_one())
    assert out.metadata_status == MetadataStatus.COMPLETED and out.metadata_provider == "grq"


def test_both_timeout_fails_not_stuck(env):
    env(g_meta="hang", q_meta="hang")
    out = service.get_metadata_status(_run_one())
    assert out.metadata_status == MetadataStatus.FAILED


def test_unexpected_exception_does_not_leave_job_processing(env):
    env(g_meta=RuntimeError("boom secret"))
    out = service.get_metadata_status(_run_one())
    assert out.metadata_status == MetadataStatus.FAILED
    assert "boom" not in (out.metadata_error or "")


def test_concurrent_jobs_are_isolated(env):
    env()

    async def scenario():
        tasks = BackgroundTasks()
        a, b = await asyncio.gather(service.process_upload(FILE, tasks), service.process_upload(FILE, tasks))
        assert a.job_id != b.job_id and len(tasks.tasks) == 2
        service._update_job(a.job_id, metadata_status=MetadataStatus.FAILED)
        assert service.get_ocr_result(b.job_id).metadata_status == MetadataStatus.PENDING
        await tasks()
        return a.job_id, b.job_id

    a, b = asyncio.run(scenario())
    assert service.get_ocr_result(a).metadata_status == MetadataStatus.COMPLETED
    assert service.get_ocr_result(b).metadata_status == MetadataStatus.COMPLETED


def _run_one() -> str:
    async def scenario():
        tasks = BackgroundTasks()
        resp = await service.process_upload(FILE, tasks)
        await tasks()
        return resp.job_id

    return asyncio.run(scenario())


# ---- HTTP layer -------------------------------------------------------------------------


def test_http_unknown_job_404():
    r = TestClient(app).get("/python/ocr/does-not-exist/metadata")
    assert r.status_code == 404


def test_http_upload_contract_and_metadata_endpoint(env):
    env()
    client = TestClient(app)
    r = client.post("/python/ocr/upload", files={"file": ("a.pdf", b"x", "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    # Existing fields still present; metadata is null and pending in the initial response.
    for key in ("success", "message", "document_id", "status", "filename", "ocr_model",
                "ocr_data", "metadata", "metadata_model", "metadata_status", "metadata_error"):
        assert key in body
    assert body["job_id"] == body["document_id"]
    assert body["metadata"] is None and body["metadata_status"] == "pending"
    assert body["ocr_data"]["full_text"] == "FIR 1/2026"

    # TestClient has run the background task by now.
    m = client.get(f"/python/ocr/{body['job_id']}/metadata")
    assert m.status_code == 200
    data = m.json()
    assert set(data) == {"job_id", "metadata_status", "metadata", "metadata_provider", "metadata_error", "isFir"}
    assert data["metadata_status"] == "completed" and data["metadata_provider"] == "gem"
    assert data["metadata"]["fir_number"] == "FIR 1/2026" and len(data["metadata"]) == 16 and "application" in data["metadata"]
    assert client.get(f"/python/ocr/{body['job_id']}").json()["metadata_status"] == "completed"
