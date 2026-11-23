"""MOCKED tests: OCR text and metadata come back from ONE Gemini request."""
import asyncio
import json
from types import SimpleNamespace

import fitz
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.schemas import GeminiOCRResult, MetadataStatus, OCRMetadata, metadata_from_dict

KEY = "AQ.Ab8RN6FAKEKEY000000000000000000000000001"
HINDI_NAME = "श्री मंटू सिंह"
META = {
    "document_type": "FIR", "fir_number": "5799028240086", "case_number": None, "document_date": "02/03/2024",
    "police_station": "खैरा थाना", "district": "जमुई", "investigating_officer": None,
    "legal_sections": ["135"], "complainant_name": None, "accused_names": [HINDI_NAME], "witness_names": [],
    "incident_date_time": "26/02/2024 11:20", "incident_location": "धर्मपुर", "document_narrative": None,
    "investigation_actions_findings": [],
}
PAGES = [{"page_number": 1, "text": "FIR No. 5799028240086 खैरा थाना जमुई"}]


class Gemini:
    """Fake SDK client recording every request."""

    def __init__(self, combined):
        self.requests = []
        self.combined = combined

        async def generate_content(**kw):
            schema = getattr(kw["config"], "response_schema", None)
            self.requests.append(schema)
            if schema is OCRMetadata:  # the separate (fallback) metadata request
                return SimpleNamespace(text=json.dumps(META, ensure_ascii=False), candidates=[])
            return SimpleNamespace(text=json.dumps(self.combined, ensure_ascii=False), candidates=[])

        self.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


class FakeFile:
    def __init__(self, name="a.pdf", ctype="application/pdf"):
        doc = fitz.open()
        doc.new_page().insert_text((72, 100), "x", fontsize=11)
        self.filename, self.content_type, self._data = name, ctype, doc.tobytes()

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


@pytest.fixture
def setup(monkeypatch):
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq"))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    service.ocr_results.clear()

    async def groq_must_not_run(data, mime):
        raise AssertionError("Groq must not be called")

    monkeypatch.setattr(groq_client, "extract_text", groq_must_not_run)

    def install(combined):
        g = Gemini(combined)
        monkeypatch.setattr(km, "_manager", km.GeminiKeyManager([KEY], client_factory=lambda k, p: g.client))
        return g

    return install


def upload():
    async def go():
        tasks = BackgroundTasks()
        resp = await service.process_upload(FakeFile(), tasks)
        return resp, tasks

    return asyncio.run(go())


def test_one_request_returns_ocr_and_metadata_immediately(setup):
    g = setup({"pages": PAGES, "full_text": PAGES[0]["text"], "metadata": META})
    resp, tasks = upload()
    assert len(g.requests) == 1 and g.requests[0] is GeminiOCRResult  # single combined request
    assert resp.metadata is not None and resp.metadata.fir_number == "5799028240086"
    assert resp.metadata_status == MetadataStatus.COMPLETED and resp.metadata_model == "gem"
    assert tasks.tasks == []  # nothing left to run in the background
    assert resp.ocr_data.full_text == PAGES[0]["text"]
    assert "completed successfully" in resp.message


def test_metadata_is_stored_and_status_endpoint_returns_it(setup):
    setup({"pages": PAGES, "metadata": META})
    resp, _ = upload()
    out = service.get_metadata_status(resp.job_id)
    assert out.metadata_status == MetadataStatus.COMPLETED and out.metadata.district == "जमुई"
    assert out.metadata_provider == "gem"


def test_hindi_metadata_values_are_kept_verbatim(setup):
    setup({"pages": PAGES, "metadata": META})
    resp, _ = upload()
    assert resp.metadata.accused_names == [HINDI_NAME]
    assert resp.metadata.police_station == "खैरा थाना" and resp.metadata.incident_location == "धर्मपुर"
    assert len(resp.metadata.model_dump()) == 16  # the 15 fields + metadata.application


def test_invalid_metadata_falls_back_to_background_extraction(setup):
    bad = dict(META, legal_sections="not-a-list")
    g = setup({"pages": PAGES, "metadata": bad})
    resp, tasks = upload()
    assert resp.metadata is None and resp.metadata_status == MetadataStatus.PENDING and len(tasks.tasks) == 1
    asyncio.run(tasks())
    assert g.requests == [GeminiOCRResult, OCRMetadata]  # the dedicated request only ran as a fallback
    assert service.get_metadata_status(resp.job_id).metadata_status == MetadataStatus.COMPLETED


def test_missing_or_empty_metadata_falls_back(setup):
    for combined in ({"pages": PAGES}, {"pages": PAGES, "metadata": {k: ([] if isinstance(v, list) else None) for k, v in META.items()}}):
        setup(combined)
        resp, tasks = upload()
        assert resp.metadata_status == MetadataStatus.PENDING and len(tasks.tasks) == 1


def test_empty_ocr_text_skips_metadata_even_if_model_returns_some(setup):
    setup({"pages": [{"page_number": 1, "text": ""}], "metadata": META})
    resp, tasks = upload()
    assert resp.metadata_status == MetadataStatus.SKIPPED and resp.metadata is None and tasks.tasks == []


def test_groq_ocr_has_no_same_call_metadata_so_background_runs(setup, monkeypatch):
    from services.ocr.groq_client import GroqMetadataError  # noqa: F401
    g = setup({})
    monkeypatch.setattr(km, "_manager", km.GeminiKeyManager([]))  # no Gemini credentials at all

    async def groq(data, mime):
        return {"pages": PAGES, "full_text": PAGES[0]["text"]}

    monkeypatch.setattr(groq_client, "extract_text", groq)
    resp, tasks = upload()
    assert resp.ocr_provider == "groq" and resp.metadata is None
    assert resp.metadata_status == MetadataStatus.PENDING and len(tasks.tasks) == 1


def test_request_schema_and_prompt_cover_all_fifteen_fields():
    assert "metadata" in GeminiOCRResult.model_fields
    for field in OCRMetadata.model_fields:
        assert field in gemini_client.COMBINED_PROMPT
    assert "NEVER translate" in gemini_client.COMBINED_PROMPT and "STRICT RULES" in gemini_client.COMBINED_PROMPT
    assert gemini_client.COMBINED_PROMPT.startswith(gemini_client.OCR_PROMPT)


def test_metadata_from_dict_rules():
    assert metadata_from_dict({"metadata": META}).fir_number == "5799028240086"
    assert metadata_from_dict({}) is None and metadata_from_dict({"metadata": "x"}) is None
    assert metadata_from_dict({"metadata": {"accused_names": 5}}) is None


def test_http_upload_response_contains_metadata_directly(setup):
    setup({"pages": PAGES, "full_text": PAGES[0]["text"], "metadata": META})
    client = TestClient(app)
    r = client.post("/python/ocr/upload", files={"file": ("a.pdf", FakeFile()._data, "application/pdf")})
    body = r.json()
    assert r.status_code == 200 and body["metadata_status"] == "completed"
    assert body["metadata"]["fir_number"] == "5799028240086" and body["metadata"]["accused_names"] == [HINDI_NAME]
    assert body["ocr_data"]["full_text"] == PAGES[0]["text"] and body["job_id"] == body["document_id"]
    got = client.get(f"/python/ocr/{body['job_id']}/metadata").json()
    assert got["metadata_status"] == "completed" and got["metadata"]["district"] == "जमुई"
